"""Functionality for Infering on a Trained GraphDRP Model."""

from pathlib import Path
import os
import json

import numpy as np
import pandas as pd

from typing import Dict

import torch
from torch_geometric.data import DataLoader

from improve import framework as frm

from improve.utils import TestbedDataset
from improve.metrics import compute_metrics
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet

from graphdrp_train_improve import gdrp_model_conf, gdrp_data_conf, save_preds, str2Class

filepath = Path(__file__).resolve().parent


req_infer_args = ["model_arch", "model_outdir", "test_ml_data_dir", "test_data"]


def check_data_available(params: Dict) -> frm.DataPathDict:
    """
    Sweep the expected input paths and check that files needed in testing are available.

    :param Dict params: Dictionary of parameters read

    :return: Path to directories requested stored in dictionary with str key str and Path value.
    :rtype: DataPathDict
    """
    # Expected
    # test_mld_data_dir / test_data
    # Make sure that the test data exists
    itestpath = Path(params["test_ml_data_dir"]) / params["test_data"]
    if itestpath.exists() == False:
        raise Exception(f"ERROR ! Testing data {intestpath} not found.\n")

    # Create output directory. Do not complain if it exists.
    opath = Path(params["model_outdir"])
    os.makedirs(opath, exist_ok=True)
    modelpath = opath / params["model_params"]
    if modelpath.exists() == False:
        raise Exception(f"ERROR ! Trained model {modelpath} not found.\n")
    fname = f"test_{params['model_eval_suffix']}.csv"
    predpath = opath / fname
    fname = f"test_{params['json_scores_suffix']}.json"
    scorespath = opath / fname

    # Return in DataPathDict structure
    inputdtd = {"test": itestpath, "model": modelpath}
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
    #test_data_file_name = "test_data"  # TestbedDataset() appends this string with ".pt"
    ## test_data = TestbedDataset(root=args.test_ml_data_dir, dataset=test_data_file_name)
    ##test_ml_data_dir_complete = params["ml_data_dir"] / params["test_ml_data_dir"]
    #test_ml_data_dir_complete = params["test_ml_data_dir"]
    #test_data = TestbedDataset(root=test_ml_data_dir_complete, dataset=test_data_file_name)
    test_data = TestbedDataset(root=params["test_ml_data_dir"], dataset="test_data")

    # PyTorch dataloaders
    test_loader = DataLoader(test_data, batch_size=test_batch,
                             shuffle=False)  # Note! Don't shuffle the test_loader

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
    pred_df = pd.DataFrame(test_pred, columns=[pred_col_name])  # This includes only predicted values

    # -----------------------------
    # Concat raw predictions with the cancer and drug ids, and the true values
    # -----------------------------
    rsp_df = pd.read_csv(indtd["test"])
    mm = pd.concat([rsp_df, pred_df], axis=1)
    mm = mm.astype({params["y_col_name"]: np.float32, pred_col_name: np.float32})

    # Save the raw predictions on val data
    save_preds(mm,
               params["canc_col_name"],
               params["drug_col_name"],
               params["y_col_name"],
               params["pred_col_name_suffix"],
               outdtd["pred"],
              )

    # -----------------------------
    # Compute performance scores
    # -----------------------------
    metrics = ["mse", "rmse", "pcc", "scc", "r2"]
    y_true = rsp_df[params["y_col_name"]].values
    test_scores = compute_metrics(y_true, test_pred, metrics)
    test_scores["test_loss"] = test_scores["mse"]

    with open(outdtd["scores"], "w", encoding="utf-8") as f:
        json.dump(test_scores, f, ensure_ascii=False, indent=4)

    print("Inference scores:\n\t{}".format(test_scores))
    return test_scores


def main():
    params = frm.initialize_parameters(filepath,
                                       default_model="graphdrp_default_model.txt",
                                       additional_definitions = gdrp_model_conf + gdrp_data_conf,
                                       required = req_infer_args,
                                      )
    run(params)
    print("\nFinished inference of GraphDRP model.")


if __name__ == "__main__":
    main()
