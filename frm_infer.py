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

    # Model specific params
    test_batch = args.test_batch

    # -----------------------------
    # Create output dir for inference results
    IMPROVE_DATADIR = fdir/"improve_data_dir"
    INFER_DIR = IMPROVE_DATADIR/"infer"

    # Outputdir name structure: train_dataset-test_datast
    import ipdb; ipdb.set_trace()
    # print(args.model_dir)
    # print(args.src)
    # print(args.test_ml_datadir)
    infer_outdir = INFER_DIR / f"{str(args.model_dir).split(os.sep)[2]}-{args.src}"  # source dataset
    os.makedirs(infer_outdir, exist_ok=True)

    # -----------------------------
    # Test dataset
    ML_DATADIR = IMPROVE_DATADIR/"ml_data"
    root_test_data = ML_DATADIR/f"data.{args.src}/{args.test_ml_datadir}"

    # -----------------------------
    # Prepare PyG datasets
    DATA_FILE_NAME = "data"  # TestbedDataset() appends this string with ".pt"
    test_data = TestbedDataset(root=root_test_data, dataset=DATA_FILE_NAME)

    # PyTorch dataloaders
    # Note! Don't shuffle the val_loader
    test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)

    # CUDA device from env var
    print("CPU/GPU: ", torch.cuda.is_available())
    if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
        # Note! When one or multiple device numbers are passed via CUDA_VISIBLE_DEVICES,
        # the values in python script are reindexed and start from 0.
        print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
        cuda_name = "cuda:0"
    else:
        cuda_name = args.cuda_name

    # Load the best model (as determined based val data)
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)
    model_file_name = Path(args.model_dir)/"model.pt"
    model.load_state_dict(torch.load(model_file_name))
    model.eval()

    # Compute raw predictions for val data
    # import ipdb; ipdb.set_trace()
    G_test, P_test = predicting(model, device, test_loader)
    pred = pd.DataFrame({"True": G_test, "Pred": P_test})

    # Concat raw predictions with the cancer and drug ids, and the true values
    RSP_FNAME = "rsp.csv"
    rsp_df = pd.read_csv(root_test_data/RSP_FNAME)
    pred = pd.concat([rsp_df, pred], axis=1)
    pred = pred.astype({"AUC": np.float32, "True": np.float32, "Pred": np.float32})
    assert sum(pred["True"] == pred["AUC"]) == pred.shape[0], \
        "Columns 'AUC' and 'True' are the ground truth, and thus, should be the same."

    # Save the raw predictions on val data
    pred_fname = "test_preds.csv"
    pred.to_csv(infer_outdir/pred_fname, index=False)

    # Get performance scores for val data
    # TODO:
    # Here performance scores/metrics are computed using functions defined in
    # this repo. Consider to use function defined by the framework (e.g., CANDLE)
    # so that all DRP models use same methods to compute scores.
    ## Method 1 - compute scores using the loaded model and val data
    mse_test = mse(G_test, P_test)
    rmse_test = rmse(G_test, P_test)
    pcc_test = pearson(G_test, P_test)
    scc_test = spearman(G_test, P_test)
    test_scores = {"mse": float(mse_test),
                  "rmse": float(rmse_test),
                  "pcc": float(pcc_test),
                  "scc": float(scc_test)}
    ## Method 2 - get the scores that were ealier computed (in for loop)
    # val_scores = {"val_loss": float(best_mse),
    #               "rmse": float(best_rmse),
    #               "pcc": float(best_pearson),
    #               "scc": float(best_spearman)}

    # Performance scores for Supervisor HPO
    with open(infer_outdir/"test_scores.json", "w", encoding="utf-8") as f:
        json.dump(test_scores, f, ensure_ascii=False, indent=4)

    # timer.display_timer()
    print("Scores:\n\t{}".format(test_scores))
    return test_scores


def run(gParameters):
    print("In Run Function:\n")
    args = candle.ArgumentStruct(**gParameters)
    modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][args.modeling]

    # Call launch() with specific model arch and args with all HPs
    scores = launch(modeling, args)

    # Supervisor HPO
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
    print("Done inference.")


if __name__ == "__main__":
    main()
