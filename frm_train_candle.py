import argparse
# import datetime
import json
import os
from pathlib import Path
import sys
# from random import shuffle
# from time import time

import candle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import graphdrp as bmk
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
# from utils import *
from utils import TestbedDataset, DataLoader

import improve_utils as imp
from improve_utils import improve_globals as ig


fdir = Path(__file__).resolve().parent
# file_path = os.path.dirname(os.path.realpath(__file__))


def train(model, device, train_loader, optimizer, epoch, log_interval):
    """ Training of one epoch (all batches). """
    print("Training on {} samples...".format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.MSELoss()
    avg_loss = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if batch_idx % log_interval == 0:
            print(
                "Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data.x),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    return sum(avg_loss) / len(avg_loss)


def predicting(model, device, loader):
    """ Method to run predictions/inference.
    The same method is in frm_infer.py
    TODO: put this in some utils script because it's also used in inference.
    """
    # TODO: this func assumes that the data contains true labels!
    # It's not always going to be the case.
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print("Make prediction for {} samples...".format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _ = model(data)  # predictions
            total_preds = torch.cat((total_preds, output.cpu()), 0)  # preds to tensor
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)  # labels to tensor
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def launch(model_arch, params):

    # CANDLE known params
    lr = params['learning_rate']
    num_epoch = params['epochs']
    train_batch = params['batch_size']

    # Model specific params
    model_st = model_arch.__name__  # model name (string)
    log_interval = params['log_interval']
    val_batch = params['val_batch']

    # Dir to save the trained (converged) model
    model_file_name = "model.pt"  # TODO: this depends on the DL framework
    model_outdir = Path(params['model_outdir'])
    os.makedirs(model_outdir, exist_ok=True)
    model_path = model_outdir + '/' + model_file_name  # file name of the model

    # -----------------------------
    # Prepare PyG datasets
    train_data_file_name = "train_data"  # TestbedDataset() appends this string with ".pt"
    val_data_file_name = "val_data"
    train_data = TestbedDataset(root=params['train_ml_data_dir'], dataset=train_data_file_name)
    val_data = TestbedDataset(root=params['val_ml_data_dir'], dataset=val_data_file_name)

    # PyTorch dataloaders
    train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)  # Note! Don't shuffle the val_loader

    # -----------------------------
    # CUDA device from env var
    print("CPU/GPU: ", torch.cuda.is_available())
    if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
        # Note! When one or multiple device numbers are passed via CUDA_VISIBLE_DEVICES,
        # the values in python script are reindexed and start from 0.
        print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
        cuda_name = "cuda:0"
    else:
        cuda_name = params['cuda_name']

    # -----------------------------
    # Select CUDA/CPU device and move model to device
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = model_arch().to(device)

    # -----------------------------
    # DL optimizer. TODO: should this be specified with CANDLE/IMPROVE?
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -----------------------------
    # Train
    # -----------------------------
    # Variables to monitor the best model based val data
    best_mse = 1000
    best_pearson = 0
    best_epoch = 0

    # Iterate over epochs
    early_stop_counter = 0
    for epoch in range(num_epoch):
        train_loss = train(model, device, train_loader, optimizer, epoch + 1, log_interval)

        # Predict with val data
        G, P = predicting(model, device, val_loader)  # G (groud truth), P (predictions)
        ret = [imp.rmse(G, P), imp.mse(G, P), imp.pearson(G, P), imp.spearman(G, P)]

        # Save best model
        # TODO:
        # Early stopping should be done the same way for all models.
        # Should this be replaced with a checkpoint??
        if ret[1] < best_mse:
            torch.save(model.state_dict(), model_path)
            # with open(result_file_name, "w") as f:
            #     f.write(",".join(map(str, ret)))
            best_epoch = epoch + 1
            best_rmse = ret[0]
            best_mse = ret[1]
            best_pearson = ret[2]
            best_spearman = ret[3]
            print(f"RMSE improved at epoch {best_epoch}; Best RMSE: {best_mse}; Model: {model_st}")
        else:
            print(f"No improvement since epoch {best_epoch}; Best RMSE: {best_mse}; Model: {model_st}")
            early_stop_counter += 1

        if early_stop_counter == params['patience']:
            print(f"Terminate training (model did not improve on val data for {params['patience']} epochs).")
            continue

    # -----------------------------
    # Load saved (best) model
    # -----------------------------
    del model
    model = model_arch().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # -----------------------------
    # Compute raw predictions
    # -----------------------------
    pred_col_name = params['y_col_name'] + ig.pred_col_name_suffix
    true_col_name = params['y_col_name'] + "_true"
    G_val, P_val = predicting(model, device, val_loader)
    # tp = pd.DataFrame({true_col_name: G_val, pred_col_name: P_val})  # This includes true and predicted values
    pred_df = pd.DataFrame(P_val, columns=[pred_col_name])  # This includes only predicted values

    # -----------------------------
    # Concatenate raw predictions with the cancer and drug ids, and the true values
    # -----------------------------
    RSP_FNAME = "val_response.csv"  # TODO: move to improve_utils? ask Yitan?
    rsp_df = pd.read_csv(Path(params['val_ml_data_dir'])/RSP_FNAME)

    # # Old
    # tp = pd.concat([rsp_df, tp], axis=1)
    # tp = tp.astype({params['y_col_name']: np.float32, true_col_name: np.float32, pred_col_name: np.float32})
    # assert sum(tp[true_col_name] == tp[params['y_col_name']]) == tp.shape[0], \
    #     f"Columns {params['y_col_name']} and {true_col_name} are the ground truth, and thus, should be the same."

    # New
    mm = pd.concat([rsp_df, pred_df], axis=1)
    mm = mm.astype({params['y_col_name']: np.float32, pred_col_name: np.float32})

    # Save the raw predictions on val data
    pred_fname = "val_preds.csv"
    imp.save_preds(mm, params['y_col_name'], model_outdir + '/' + pred_fname)

    # -----------------------------
    # Compute performance scores
    # -----------------------------
    # TODO: Make this a func in improve_utils.py --> calc_scores(y_true, y_pred)
    # Compute scores using the loaded model
    y_true = rsp_df[params['y_col_name']].values
    mse_val = imp.mse(y_true, P_val)
    rmse_val = imp.rmse(y_true, P_val)
    pcc_val = imp.pearson(y_true, P_val)
    scc_val = imp.spearman(y_true, P_val)
    r2_val = imp.r_square(y_true, P_val)
    val_scores = {"val_loss": float(mse_val),
                  "rmse": float(rmse_val),
                  "pcc": float(pcc_val),
                  "scc": float(scc_val),
                  "r2": float(r2_val)}

    # Performance scores for Supervisor HPO
    print("\nIMPROVE_RESULT val_loss:\t{}\n".format(mse_val))
    with open(model_outdir + '/' + "val_scores.json", "w", encoding="utf-8") as f:
        json.dump(val_scores, f, ensure_ascii=False, indent=4)

    print("Validation scores:\n\t{}".format(val_scores))
    return val_scores


def run(params):
    print("In Run Function:\n")
    # args = candle.ArgumentStruct(**gParameters)
    modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][params['modeling']]

    # Call launch() with specific model arch and args with all HPs
    # TODO: do we really need launch (especially that we already have run)?
    scores = launch(modeling, params)

    # Supervisor HPO
    print("\nIMPROVE_RESULT val_loss:\t{}\n".format(scores["val_loss"]))
    with open(Path(params['output_dir']) / "val_scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    return scores


def initialize_parameters():
    # def initialize_parameters(args):
    """ Initialize the parameters for the GraphDRP benchmark. """
    print("Initializing parameters\n")
    # TODO: do we have to create graphdrp.py? Certain models have such file
    # already exists (i.e., MODEL_NAME.py). We want to little changes to the
    # original code. Can we call it some other name? candle_benchmark_def.py?

    graphdrp_bmk = bmk.BenchmarkGraphDRP(
        filepath=bmk.file_path,
        defmodel="graphdrp_default_model.txt",
        framework="pytorch",
        prog="GraphDRP",
        desc="CANDLE compliant training GraphDRP",
    )

    gParameters = candle.finalize_parameters(graphdrp_bmk)

    return gParameters


def parse_args(args):
    """ Parse input args. """

    parser = argparse.ArgumentParser(description="Train model")

    # Args common to train and infer scripts
    parser.add_argument(
        "--model_arch",
        type=int,
        default=0,
        required=False,
        help="Integer. 0: GINConvNet, 1: GATNet, 2: GAT_GCN, 3: GCNNet")
    parser.add_argument(
        "--y_col_name",
        type=str,
        default="auc",
        required=False,
        help="Drug sensitivity score to use as the target variable (e.g., IC50, AUC).")
    parser.add_argument(
        "--cuda_name",
        type=str,
        default="cuda:0",
        required=False,
        help="Cuda device (e.g.: cuda:0, cuda:1.")

    # Args specific to train script
    parser.add_argument(
        "--train_ml_data_dir",
        type=str,
        required=True,
        help="...")
    parser.add_argument(
        "--val_ml_data_dir",
        type=str,
        required=True,
        help="...")
    parser.add_argument(
        "--model_outdir",
        type=str,
        required=True,
        help="Output dir to store/checkpoint the trained model.")

    # DL hyperparameters
#    parser.add_argument(
#        "--epochs",
#        type=int,
#        default=10,
#        required=False,
#        help="Number of epochs.")
#    parser.add_argument(
#        "--learning_rate",
#        type=float,
#        default=0.0001,
#        required=False,
#        help="Learning rate.")
#    parser.add_argument(
#        "--batch_size",
#        type=int,
#        default=256,
#        required=False,
#        help="Input batch size for training.")
    parser.add_argument(
        "--val_batch",
        type=int,
        default=256,
        required=False,
        help="Input batch size for validation.")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        required=False,
        help="Interval for saving o/p.")
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        required=False,
        help="Number of epochs with no improvement after which training will be stopped.")

    args = parser.parse_args(args)
    return args


def main():
    # import ipdb; ipdb.set_trace()

    # Using CANDLE
    # TODO: how should we utilize CANDLE here?
    gParameters = initialize_parameters()
    # print(gParameters)
    # args = candle.ArgumentStruct(**gParameters)
    # modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][args.modeling]
    # scores = launch(modeling, gParameters)

    # Without CANDLE
    # args = parse_args(args)
    model_arch = [GINConvNet, GATNet, GAT_GCN, GCNNet][gParameters['model_arch']]
    scores = launch(model_arch, gParameters)

    print("\nFinished training.")


if __name__ == "__main__":
    main()
