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

from graphdrp_preprocess_improve import gdrp_data_conf
from graphdrp_train_improve import gdrp_model_conf, save_preds, str2Class

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
    # Infer parameters
    test_batch = params["test_batch"]

    # -----------------------------
    # Prepare PyG datasets
    test_data_file_name = params["test_data_processed"]
    if test_data_file_name.endswith(".pt"):
        test_data_file_name = test_data_file_name[:-3] # TestbedDataset() appends this string with ".pt"
    test_data = TestbedDataset(root=indtd["test"], dataset=test_data_file_name)

    # PyTorch dataloaders
    test_loader = DataLoader(test_data, batch_size=test_batch,
                             shuffle=False)  # Note! Don't shuffle the test_loader or results will be corrupted

    # -----------------------------
    # [Req]
    # Determine CUDA/CPU device and configure CUDA device if available
    cuda_avail = torch.cuda.is_available()
    print("CPU/GPU: ", cuda_avail)
    if cuda_avail:  # GPU available
        # -----------------------------
        # CUDA device from env var
        cuda_env_visible = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_env_visible is not None:
            # Note! When one or multiple device numbers are passed via CUDA_VISIBLE_DEVICES,
            # the values in python script are reindexed and start from 0.
            print("CUDA_VISIBLE_DEVICES: ", cuda_env_visible)
            cuda_name = "cuda:0"
        else:
            cuda_name = params["cuda_name"]
        device = cuda_name
    else:
        device = "cpu"

    # -----------------------------
    # Move model to device
    model = str2Class(params["model_arch"]).to(device)

    # -----------------------------
    model.load_state_dict(torch.load(indtd["model"]))
    model.eval()

    # -----------------------------
    # Compute raw predictions
    # -----------------------------
    pred_col_name = params["y_col_name"] + params["pred_col_name_suffix"]
    true_col_name = params["y_col_name"] + "_true"
    test_true, test_pred = frm.predicting(model, device, test_loader)  # (groud truth), (predictions)

    # -----------------------------
    # Attempt to concat raw predictions with the cancer and drug ids, and the true values
    if indtd["df"] is not None:
        rsp_df = pd.read_csv(indtd["df"])

        pred_df = pd.DataFrame(test_pred, columns=[pred_col_name])  # This includes only predicted values

        mm = pd.concat([rsp_df, pred_df], axis=1)
        mm = mm.astype({params["y_col_name"]: np.float32, pred_col_name: np.float32})

        # Save the raw predictions on val data
        # Note that there is no guarantee that the results effectively correspond to
        # this pre-processing parameters or the specified data frame
        # since the data is being read from a processed pt file (no from original data frame)
        save_preds(mm,
               params["canc_col_name"],
               params["drug_col_name"],
               params["y_col_name"],
               params["pred_col_name_suffix"],
               outdtd["pred"],
        )
        y_true = rsp_df[params["y_col_name"]].values
    else: # Save only ground truth and predictions since cancer and drug ids are not available
        df_ = pd.DataFrame({true_col_name: test_true, pred_col_name: test_pred})  # This includes true and predicted values
        # Save preds df
        df_.to_csv(outdtd["pred"], index=False)
        y_true = test_true

    # -----------------------------
    # Compute performance scores
    # -----------------------------
    metrics = ["mse", "rmse", "pcc", "scc", "r2"]
    test_scores = compute_metrics(y_true, test_pred, metrics)
    test_scores["test_loss"] = test_scores["mse"]

    with open(outdtd["scores"], "w", encoding="utf-8") as f:
        json.dump(test_scores, f, ensure_ascii=False, indent=4)

    print("Inference scores:\n\t{}".format(test_scores))
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
