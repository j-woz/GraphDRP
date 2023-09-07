"""Functionality for Infering on a Trained GraphDRP Model."""

from pathlib import Path
import os
import json
import warnings

import numpy as np
import pandas as pd

from typing import Dict

import torch
from torch_geometric.data import DataLoader

from improve import framework as frm

from improve.torch_utils import TestbedDataset
from improve.metrics import compute_metrics
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet

from graphdrp_train_improve import (
    gdrp_data_conf,
    gdrp_model_conf,
    determine_device,
    build_PT_data_loader,
    evaluate_model,
    store_predictions_df,
    compute_performace_scores,
)

filepath = Path(__file__).resolve().parent

gdrp_test_conf = [
    {"name": "test_data_df",
     "default": frm.SUPPRESS,
     "type": str,
     "help": "Data frame with original test response data."
    },
]

req_infer_args = ["model_arch", "model_outdir", "test_ml_data_dir", "test_data_processed"]


def check_data_available(params: Dict) -> frm.DataPathDict:
    """
    Sweep the expected input paths and check that files needed in testing are available.

    :param Dict params: Dictionary of parameters read

    :return: Path to directories requested stored in dictionary with str key str and Path value.
    :rtype: DataPathDict
    """
    # Expected
    # test_ml_data_dir / processed / test_data_processed
    # test_data_processed --> it has pt extension and is located inside a 'processed' folder
    # Make sure that the test data exists
    itestpath = Path(params["test_ml_data_dir"]) / "processed"
    if itestpath.exists() == False:
        raise Exception(f"ERROR ! Processed testing data folder {itestpath} not found.\n")
    itest = itestpath / params["test_data_processed"]
    if itest.exists() == False:
        raise Exception(f"ERROR ! Processed testing data {itest} not found.\n")

    # Check if validation data frame exists
    itestdfpath = None
    if "test_data_df" in params:
        itestdfpath = Path(params["test_ml_data_dir"]) / params["test_data_df"]
        if itestdfpath.exists() == False:
            itestdfpath = None
            message = (
                       f"Data frame with original testing response data: {params['test_data_df']} not found." \
                       + " Will continue but will only store partial (available) data frame.\n"
            )
            warnings.warn(message, RuntimeWarning)
    else:
        message = (
                   f"Data frame with original testing response data not specified (no 'test_data_df' keyword)." \
                       + " Will continue but will only store partial (available) data frame.\n"
            )
        warnings.warn(message, RuntimeWarning)

    # Create output directory. Do not complain if it exists.
    opath = Path(params["model_outdir"])
    os.makedirs(opath, exist_ok=True)
    modelpath = opath / params["model_params"]
    if modelpath.exists() == False:
        raise Exception(f"ERROR ! Trained model {modelpath} not found.\n")
    fname = f"test_{params['model_eval_suffix']}.csv"
    if itestdfpath is None:
        fname = f"test_{params['model_eval_suffix']}_partial.csv"
    predpath = opath / fname
    fname = f"test_{params['json_scores_suffix']}.json"
    scorespath = opath / fname

    # Return in DataPathDict structure
    inputdtd = {"test": itestpath, "model": modelpath, "df": itestdfpath}
    outputdtd = {"pred": predpath, "scores": scorespath}

    return inputdtd, outputdtd


def run(params):
    """Execute specified model inference.

    Parameters
    ----------
    params: python dictionary
        A dictionary of Candle keywords and parsed values.
    """
    # ------------------------------------------------------
    # Check data availability and create output directory
    # ------------------------------------------------------
    # [Req]
    indtd, outdtd = check_data_available(params)
    # indtd is dictionary with input_description: path components
    # outdtd is dictionary with output_description: path components

    # -----------------------------
    # [Req]
    # Determine CUDA/CPU device and configure CUDA device if available
    device = determine_device(params["cuda_name"])

    # -----------------------------
    # Prepare PyTorch dataloaders
    # Note! Don't shuffle the test_loader or results will be corrupted
    test_loader = build_PT_data_loader(params["test_ml_data_dir"],
                                        params["test_data_processed"],
                                        params["test_batch"],
                                        shuffle=False)

    # -----------------------------
    # Load the saved model
    # Compute predictions
    # (groud truth), (predictions)
    test_true, test_pred = evaluate_model(params["model_arch"], device, indtd["model"], test_loader)

    # Store predictions in data frame
    # Attempt to concat predictions with the cancer and drug ids, and the true values
    # If data frame found, then y_true is read from data frame and returned
    # Otherwise, only a partial data frame is stored (with test_true and test_pred)
    # and y_true is equal to pytorch loaded test_true
    y_true = store_predictions_df(params, indtd, outdtd, test_true, test_pred)
    # Compute performance scores
    metrics = ["mse", "rmse", "pcc", "scc", "r2"]
    test_scores = compute_performace_scores(y_true, test_pred, metrics, outdtd, "test")

    return test_scores


def main():
    params = frm.initialize_parameters(filepath,
                                       default_model="graphdrp_default_model.txt",
                                       additional_definitions = gdrp_model_conf + gdrp_data_conf + gdrp_test_conf,
                                       required = req_infer_args,
                                      )
    run(params)
    print("\nFinished inference of GraphDRP model.")


if __name__ == "__main__":
    main()
