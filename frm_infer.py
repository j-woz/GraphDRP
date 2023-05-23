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
# from utils import *
from utils import TestbedDataset, DataLoader

import improve_utils as imp
from improve_utils import improve_globals as ig


fdir = Path(__file__).resolve().parent


def predicting(model, device, loader):
    """ Method to run predictions/inference.
    The same method is in frm_train.py
    TODO: put this in some utils script.
    """
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


def launch(model_arch, args):

    # Model specific params
    test_batch = args.test_batch

    # Output dir name structure: train_dataset-test_datast
    infer_outdir = Path(args.infer_outdir)
    os.makedirs(infer_outdir, exist_ok=True)

    # -----------------------------
    # Prepare PyG datasets
    test_data_file_name = "test_data"  # TestbedDataset() appends this string with ".pt"
    test_data = TestbedDataset(root=args.test_ml_data_dir, dataset=test_data_file_name)

    # PyTorch dataloaders
    test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)  # Note! Don't shuffle the test_loader

    # -----------------------------
    # CUDA device from env var
    print("CPU/GPU: ", torch.cuda.is_available())
    if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
        # Note! When one or multiple device numbers are passed via CUDA_VISIBLE_DEVICES,
        # the values in python script are reindexed and start from 0.
        print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
        cuda_name = "cuda:0"
    else:
        cuda_name = args.cuda_name

    # -----------------------------
    # Select CUDA/CPU device and move model to device
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = model_arch().to(device)

    # -----------------------------
    # Load the best model (as determined based val data)
    model_path = Path(args.model_dir)/"model.pt"
    model.load_state_dict(torch.load(model_path))
    model.eval()
 

    # -----------------------------
    # Compute raw predictions
    # -----------------------------
    pred_col_name = args.y_col_name + ig.pred_col_name_suffix
    true_col_name = args.y_col_name + "_true"
    # G_test, P_test = predicting(model, device, test_loader)  # G (groud truth), P (predictions)
    # tp = pd.DataFrame({true_col_name: G_test, pred_col_name: P_test})  # This includes true and predicted values
    pred_df = pd.DataFrame(P_test, columns=[pred_col_name])  # This includes only predicted values


    # -----------------------------
    # Concat raw predictions with the cancer and drug ids, and the true values
    # -----------------------------
    RSP_FNAME = "test_response.csv"  # TODO: move to improve_utils? ask Yitan?
    rsp_df = pd.read_csv(Path(args.test_ml_data_dir)/RSP_FNAME)

    # # Old
    # tp = pd.concat([rsp_df, tp], axis=1)
    # tp = tp.astype({args.y_col_name: np.float32, true_col_name: np.float32, pred_col_name: np.float32})
    # assert sum(tp[true_col_name] == tp[args.y_col_name]) == tp.shape[0], \
    #     f"Columns {args.y_col_name} and {true_col_name} are the ground truth, and thus, should be the same."

    # New
    mm = pd.concat([rsp_df, pred_df], axis=1)
    mm = mm.astype({args.y_col_name: np.float32, pred_col_name: np.float32})

    # Save the raw predictions on val data
    # pred_fname = "test_preds.csv"
    pred_fname = "test_preds.csv"
    imp.save_preds(mm, args.y_col_name, infer_outdir/pred_fname)


    # -----------------------------
    # Compute performance scores
    # -----------------------------
    # TODO: Make this a func in improve_utils.py --> calc_scores(y_true, y_pred)
    # Make this a func in improve_utils.py --> calc_scores(y_true, y_pred)
    y_true = rsp_df[args.y_col_name].values
    mse_test = imp.mse(y_true, P_test)
    rmse_test = imp.rmse(y_true, P_test)
    pcc_test = imp.pearson(y_true, P_test)
    scc_test = imp.spearman(y_true, P_test)
    r2_test = imp.r_square(y_true, P_test)
    test_scores = {"test_loss": float(mse_test),
              "rmse": float(rmse_test),
              "pcc": float(pcc_test),
              "scc": float(scc_test),
              "r2": float(r2_test)}

    with open(infer_outdir/"test_scores.json", "w", encoding="utf-8") as f:
        json.dump(test_scores, f, ensure_ascii=False, indent=4)

    print("Inference scores:\n\t{}".format(test_scores))
    return test_scores


def parse_args(args):
    """ Parse input args. """
    parser = argparse.ArgumentParser(description="Train model")

    # Args common train and infer scripts
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

    # Args specific to infer script
    parser.add_argument(
        "--test_ml_data_dir",
        type=str,
        required=True,
        help="...")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Dir of the stored/checkpointed model.")
    parser.add_argument(
        "--infer_outdir",
        type=str,
        required=True,
        help="Inference results outdir.")

    # DL hyperparameters
    parser.add_argument(
        "--test_batch",
        type=int,
        default=256,
        required=False,
        help="Input batch size for testing.")

    args = parser.parse_args(args)
    return args


def main(args):
    args = parse_args(args)
    model_arch = [GINConvNet, GATNet, GAT_GCN, GCNNet][args.model_arch]
    scores = launch(model_arch, args)
    print("\nFinished inference.")


if __name__ == "__main__":
    main(sys.argv[1:])
