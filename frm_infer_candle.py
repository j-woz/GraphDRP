import os
import json

import numpy as np
import pandas as pd

from pprint import pprint

import torch
from pathlib import Path

import frm
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
import candle_improve_utils as improve_utils
from utils import TestbedDataset, DataLoader

frm.required.extend(["test_ml_data_dir", "y_col_name"])


def str2Class(str):
    return globals()[str]()

def run(params):
    """Execute specified model inference.

    Parameters
    ----------
    params: python dictionary
        A dictionary of Candle keywords and parsed values.
    """
    # Model specific params
    test_batch = params["batch_size"]  # args.test_batch ?

    # Output dir name structure: train_dataset-test_datast
    infer_outdir = Path(params["output_dir"])  # args.infer_outdir
    #infer_outdir = params["output_dir"]
    os.makedirs(infer_outdir, exist_ok=True)

    # -----------------------------
    # Prepare PyG datasets
    test_data_file_name = "test_data"  # TestbedDataset() appends this string with ".pt"
    # test_data = TestbedDataset(root=args.test_ml_data_dir, dataset=test_data_file_name)
    #test_ml_data_dir_complete = params["ml_data_dir"] / params["test_ml_data_dir"]
    test_ml_data_dir_complete = params["test_ml_data_dir"]
    test_data = TestbedDataset(root=test_ml_data_dir_complete, dataset=test_data_file_name)

    # PyTorch dataloaders
    test_loader = DataLoader(test_data, batch_size=test_batch,
                             shuffle=False)  # Note! Don't shuffle the test_loader

    # -----------------------------
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
    # Load the best model (as determined based val data)
    # model_path = Path(args.model_dir)/"model.pt"
    model_path = Path(params["ckpt_directory"] + "/" + params["model_params"])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # -----------------------------
    # Compute raw predictions
    # -----------------------------
    pred_col_name = params["y_col_name"] + params["pred_col_name_suffix"]
    true_col_name = params["y_col_name"] + "_true"
    G_test, P_test = frm.predicting(model, device, test_loader)  # G (groud truth), P (predictions)
    # tp = pd.DataFrame({true_col_name: G_test, pred_col_name: P_test})  # This includes true and predicted values
    pred_df = pd.DataFrame(P_test, columns=[pred_col_name])  # This includes only predicted values

    # -----------------------------
    # Concat raw predictions with the cancer and drug ids, and the true values
    # -----------------------------
    # RSP_FNAME = "test_response.csv"  # TODO: move to improve_utils? ask Yitan?
    # rsp_df = pd.read_csv(Path(args.test_ml_data_dir)/RSP_FNAME)
    rsp_df = pd.read_csv(Path(params["test_ml_data_dir"] + params["response_data"]))

    # # Old
    # tp = pd.concat([rsp_df, tp], axis=1)
    # tp = tp.astype({args.y_col_name: np.float32, true_col_name: np.float32, pred_col_name: np.float32})
    # assert sum(tp[true_col_name] == tp[args.y_col_name]) == tp.shape[0], \
    #     f"Columns {args.y_col_name} and {true_col_name} are the ground truth, and thus, should be the same."

    # New
    mm = pd.concat([rsp_df, pred_df], axis=1)
    mm = mm.astype({params["y_col_name"]: np.float32, pred_col_name: np.float32})

    # Save the raw predictions on val data
    # pred_fname = "test_preds.csv"
    improve_utils.save_preds(mm, params, params["out_file_path"])

    # -----------------------------
    # Compute performance scores
    # -----------------------------
    # TODO: Make this a func in improve_utils.py --> calc_scores(y_true, y_pred)
    # Make this a func in improve_utils.py --> calc_scores(y_true, y_pred)
    y_true = rsp_df[params["y_col_name"]].values
    metrics = ["mse", "rmse", "pcc", "scc", "r2"]
    test_scores = improve_utils.compute_metrics(y_true, P_test, metrics)
    test_scores["test_loss"] = test_scores["mse"]

    # out_json = "test_scores.json"
    with open(infer_outdir / params["out_json"], "w", encoding="utf-8") as f:
        json.dump(test_scores, f, ensure_ascii=False, indent=4)

    print("Inference scores:\n\t{}".format(test_scores))
    return test_scores


def main():
    params = frm.initialize_parameters()
    # Add infer parameter
    params["out_file_path"] = params["output_dir"] + params["pred_fname"]
    pprint(params)
    run(params)
    print("\nFinished inference.")


if __name__ == "__main__":
    main()
