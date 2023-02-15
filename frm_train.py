import argparse
import datetime
import json
import os
from pathlib import Path
import sys
from random import shuffle
from time import time
import json

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
from utils import *


def train(model, device, train_loader, optimizer, epoch, log_interval):
    """ Training function at each epoch. """
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
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print("Make prediction for {} samples...".format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _ = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def launch(modeling, args):

    # import ipdb; ipdb.set_trace()
    fdir = Path(__file__).resolve().parent

    # CANDLE known params
    lr = args.learning_rate
    num_epoch = args.epochs
    train_batch = args.batch_size

    # Model specific params
    model_st = modeling.__name__
    log_interval = args.log_interval
    val_batch = args.val_batch
    test_batch = args.test_batch

    # -----------------------------
    # Create output dir
    # TODO:
    # We need to determine where to dump the following outputs.
    # 1. The trained (converged) model
    # 2. Performance scores (as required by HPO Supervisor)
    # 3. Raw predictions of the val data (would be nice but probably not mandatory) 
    # Currently we use: improve_data_dir/models/DATASET_NAME/split_ID
    # outdir = Path(args.output_dir)
    # IMPROVE_DATADIR = fdir/"improve_data_dir"  # This should already exist

    # Fetch data (if needed)
    # ftp_origin = f"https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/GraphDRP/data_processed/{val_scheme}/processed"
    # data_file_list = ["train_data.pt", "val_data.pt", "test_data.pt"]

    # for f in data_file_list:
    #     candle.get_file(fname=f,
    #                     origin=os.path.join(ftp_origin, f.strip()),
    #                     unpack=False, md5_hash=None,
    #                     cache_subdir=args.cache_subdir )

    # _data_dir = os.path.split(args.cache_subdir)[0]
    # root = os.getenv('CANDLE_DATA_DIR') + '/' + _data_dir

    # -----------------------------
    # Dirs of train and val data
    root_train_data = fdir/args.train_ml_datadir
    root_val_data = fdir/args.val_ml_datadir

    # -----------------------------
    # Prepare PyG datasets
    DATA_FILE_NAME = "data"  # TestbedDataset() appends this string with ".pt"
    train_data = TestbedDataset(root=root_train_data, dataset=DATA_FILE_NAME)
    val_data = TestbedDataset(root=root_val_data, dataset=DATA_FILE_NAME)

    # PyTorch dataloaders
    train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)  # Note! Don't shuffle the val_loader

    # CUDA device from env var
    print("CPU/GPU: ", torch.cuda.is_available())
    if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
        # Note! When one or multiple device numbers are passed via CUDA_VISIBLE_DEVICES,
        # the values in python script are reindexed and start from 0.
        print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
        cuda_name = "cuda:0"
    else:
        cuda_name = args.cuda_name

    # CUDA/CPU device and optimizer
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # DL optimizer. TODO: should this be specified with CANDLE/IMPROVE?

    # Vars to monitor best model based val data
    best_mse = 1000
    best_pearson = 0
    best_epoch = 0

    # Dir to save the trained (converged) model
    # import ipdb; ipdb.set_trace()
    model_outdir = fdir/args.model_outdir
    os.makedirs(model_outdir, exist_ok=True)
    model_file_name = model_outdir/"model.pt"

    # Iterate over epochs
    # import ipdb; ipdb.set_trace()
    for epoch in range(num_epoch):
        train_loss = train(model, device, train_loader, optimizer, epoch + 1, log_interval)

        # Predict with val data
        G, P = predicting(model, device, val_loader) # G (groud truth), P (predictions)
        ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P)]

        # Save best model
        # TODO:
        # Early stopping should be done the same way for all models.
        # Should this be replaced with a checkpoint??
        if ret[1] < best_mse:
            torch.save(model.state_dict(), model_file_name)
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

    # Load the best model (as determined based val data)
    # TODO:
    # What should be the output so that we know exactly all the specific attributes
    # of this training run (e.g., data source, data split)?
    del model
    model = modeling().to(device)
    model.load_state_dict(torch.load(model_file_name))
    model.eval()

    # Compute raw predictions for val data
    G_val, P_val = predicting(model, device, val_loader)
    pred = pd.DataFrame({"True": G_val, "Pred": P_val})

    # -----------------------------
    # Concat raw predictions with the cancer and drug ids, and the true values
    # TODO:
    # Should this be a standard in CANDLE/IMPROVE?
    RSP_FNAME = "rsp.csv"
    rsp_df = pd.read_csv(root_val_data/RSP_FNAME)
    pred = pd.concat([rsp_df, pred], axis=1)
    pred = pred.astype({"AUC": np.float32, "True": np.float32, "Pred": np.float32})
    assert sum(pred["True"] == pred["AUC"]) == pred.shape[0], \
        "Columns 'AUC' and 'True' are the ground truth, and thus, should be the same."

    # Save the raw predictions on val data
    pred_fname = "val_preds.csv"
    pred.to_csv(model_outdir/pred_fname, index=False)

    # -----------------------------
    # Get performance scores for val data
    # TODO:
    # Should this be a standard in CANDLE/IMPROVE?
    # Here performance scores/metrics are computed using functions defined in
    # this repo. Consider to use function defined by the framework (e.g., CANDLE)
    # so that all DRP models use same methods to compute scores.
    ## Method 1 - compute scores using the loaded model and val data
    mse_val = mse(G_val, P_val)
    rmse_val = rmse(G_val, P_val)
    pcc_val = pearson(G_val, P_val)
    scc_val = spearman(G_val, P_val)
    val_scores = {"val_loss": float(mse_val),
                  "rmse": float(rmse_val),
                  "pcc": float(pcc_val),
                  "scc": float(scc_val)}
    ## Method 2 - get the scores that were ealier computed (in for loop)
    # val_scores = {"val_loss": float(best_mse),
    #               "rmse": float(best_rmse),
    #               "pcc": float(best_pearson),
    #               "scc": float(best_spearman)}

    # Performance scores for Supervisor HPO
    # import ipdb; ipdb.set_trace()
    print("\nIMPROVE_RESULT val_loss:\t{}\n".format(mse_val))
    with open(model_outdir/"val_scores.json", "w", encoding="utf-8") as f:
        json.dump(val_scores, f, ensure_ascii=False, indent=4)

    print("Scores:\n\t{}".format(val_scores))
    return val_scores


def run(gParameters):
    print("In Run Function:\n")
    args = candle.ArgumentStruct(**gParameters)
    modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][args.modeling]

    # Call launch() with specific model arch and args with all HPs
    scores = launch(modeling, args)

    # Supervisor HPO
    print("\nIMPROVE_RESULT val_loss:\t{}\n".format(scores["val_loss"]))
    with open(Path(args.output_dir) / "scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    return scores


def initialize_parameters():
    """ Initialize the parameters for the GraphDRP benchmark. """
    print("Initializing parameters\n")
    graphdrp_bmk = bmk.BenchmarkGraphDRP(
        filepath=bmk.file_path,
        defmodel="graphdrp_default_model.txt",
        # defmodel="graphdrp_model_candle.txt",
        framework="pytorch",
        prog="GraphDRP",
        desc="CANDLE compliant GraphDRP",
    )
    gParameters = candle.finalize_parameters(graphdrp_bmk)
    return gParameters


def main():
    gParameters = initialize_parameters()
    print(gParameters)
    scores = run(gParameters)
    print("Done training.")


if __name__ == "__main__":
    main()
