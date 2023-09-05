"""Functionality for Training one GraphDRP Model."""

from pathlib import Path
import os

import json

import numpy as np
import pandas as pd

from typing import Dict, Union

import torch
from torch_geometric.data import DataLoader

from improve import framework as frm
from candle import build_pytorch_optimizer, get_pytorch_function, keras_default_config, CandleCkptPyTorch

from improve.utils import TestbedDataset
from improve.metrics import compute_metrics
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet

filepath = Path(__file__).resolve().parent

# Params that are specific to graphDRP model
gdrp_model_conf = [
    {"name": "model_arch",
     "default": "GINConvNet",
     "choices": ["GINConvNet", "GATNet", "GAT_GCN", "GCNNet"],
     "type": str,
     "help": "Model architecture to run."},
    {"name": "log_interval",
     "action": "store",
     "type": int,
     "help": "Interval for saving o/p"},
    {"name": "cuda_name",  # TODO: how should we control this?
     "action": "store",
     "type": str,
     "help": "Cuda device (e.g.: cuda:0, cuda:1."},
]

gdrp_data_conf = [
    {"name": "canc_col_name",
     "default": "improve_sample_id",
     "type": str,
     "help": "Column name that contains the cancer sample ids.",
    },
    {"name": "drug_col_name",
     "default": "improve_chem_id",
     "type": str,
     "help": "Column name that contains the drug ids.",
    },
]

frm.additional_definitions.extend(gdrp_model_conf + gdrp_data_conf)


# Params that are required to be specified in order to execute the train script (no default values)
# TODO: consider setting req_train_args elsewhere and then loading it into this script. Similarly we can specify
# req_preprocess_args and req_infer_args
# req_train_args = ["train_ml_data_dir", "val_ml_data_dir", "y_col_name"]  # there is default for y_col_name (auc)
req_train_args = ["model_arch", "model_outdir", "train_ml_data_dir", "val_ml_data_dir", "train_data", "val_data"]
frm.required.extend(req_train_args)


def str2Class(str):
    """Get model class from model name (str)."""
    return globals()[str]()


def check_data_available(params: Dict) -> frm.DataPathDict:
    """
    Sweep the expected input paths and check that files needed training are available.

    :param Dict params: Dictionary of parameters read

    :return: Path to directories requested stored in dictionary with str key str and Path value.
    :rtype: DataPathDict
    """
    # Expected
    # train_mld_data_dir / train_data
    # val_mld_data_dir / val_data
    # Make sure that the train data exists
    itrainpath = Path(params["train_ml_data_dir"]) / params["train_data"]
    if itrainpath.exists() == False:
        raise Exception(f"ERROR ! Training data {itrainpath} not found.\n")
    # Make sure that the train data exists
    ivalpath = Path(params["val_ml_data_dir"]) / params["val_data"]
    if ivalpath.exists() == False:
        raise Exception(f"ERROR ! Validation data {ivalpath} not found.\n")

    # Create output directory. Do not complain if it exists.
    opath = Path(params["model_outdir"])
    os.makedirs(opath, exist_ok=True)
    modelpath = opath / params["model_params"]
    fname = f"val_{params['model_eval_suffix']}.csv"
    predpath = opath / fname
    fname = f"val_{params['json_scores_suffix']}.json"
    scorespath = opath / fname

    # Return in DataPathDict structure
    inputdtd = {"train": itrainpath, "val": ivalpath}
    outputdtd = {"model": modelpath, "pred": predpath, "scores": scorespath}

    return inputdtd, outputdtd


def train(model, device, train_loader, optimizer, loss_fn, epoch, log_interval):
    """ Training of one epoch (all batches). """
    print("Training on {} samples...".format(len(train_loader.dataset)))
    model.train()
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


def save_preds(df: pd.DataFrame,
               canc_col_name: str,
               drug_col_name: str,
               y_col_name: str,
               pred_col_name_suffix: str,
               outpath: Union[str, Path],
               round_decimals: int = 4) -> None:
    """ Save model predictions.
    This function throws errors if the dataframe does not include the expected
    columns: canc_col_name, drug_col_name, y_col_name, y_col_name + "_pred"

    Args:
        df (pd.DataFrame): Pandas data frame with model predictions
        canc_col_name (str): Column name that contains the cancer sample ids.
        drug_col_name (str): Column name that contains the drug ids.
        y_col_name (str): drug response col name (e.g., IC50, AUC)
        pred_col_name_suffix (str): Suffix to identoy column of predictions made by model.
        outpath (str or PosixPath): outdir to save the model predictions df.
        round (int): round response values.

    Returns:
        None
    """
    # Check that the 4 columns exist
    assert canc_col_name in df.columns, f"{canc_col_name} was not found in columns."
    assert drug_col_name in df.columns, f"{drug_col_name} was not found in columns."
    assert y_col_name in df.columns, f"{y_col_name} was not found in columns."
    pred_col_name = y_col_name + f"{pred_col_name_suffix}"
    assert pred_col_name in df.columns, f"{pred_col_name} was not found in columns."

    # Round
    df = df.round({y_col_name: round_decimals, pred_col_name: round_decimals})

    # Save preds df
    df.to_csv(outpath, index=False)
    return None


def run(params):
    """ Execute specified model training.

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
    # indtd is dictionary with stage: path components
    # outdtd is dictionary with output_type: path components

    # -----------------------------
    # Train parameters
    lr = params["learning_rate"]
    num_epoch = params["epochs"]
    train_batch = params["batch_size"]
    log_interval = params["log_interval"]
    val_batch = params["val_batch"]

    # -----------------------------
    # Prepare PyG datasets
    #train_data_file_name = f"train_{params['x_data_suffix']}" # TestbedDataset() appends this string with ".pt"
    #val_data_file_name = f"val_{params['x_data_suffix']}"
    #train_data_file_name = params["train_data"]
    #if train_data_file_name.endswith(".pt"):
    #    train_data_file_name = train_data_file_name[:-3] # TestbedDataset() appends this string with ".pt"
    #val_data_file_name = params["val_data"]
    #if val_data_file_name.endswith(".pt"):
    #    val_data_file_name = val_data_file_name[:-3] # TestbedDataset() appends this string with ".pt"
    #train_data = TestbedDataset(root=params["train_ml_data_dir"], dataset=train_data_file_name)
    #val_data = TestbedDataset(root=params["val_ml_data_dir"], dataset=val_data_file_name)
    train_data = TestbedDataset(root=params["train_ml_data_dir"], dataset="train_data")
    val_data = TestbedDataset(root=params["val_ml_data_dir"], dataset="val_data")

    # PyTorch dataloaders
    train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)  # Note! Don't shuffle the val_loader

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
    # [Req]
    # DL optimizer and loss
    keras_defaults = keras_default_config()
    optimizer = build_pytorch_optimizer(
        model=model,
        optimizer=params["optimizer"],
        lr=lr,
        kerasDefaults=keras_defaults
    )  # Note! Specified in frm_default_model.txt
    loss_fn = get_pytorch_function(params["loss"])

    # -----------------------------
    # Checkpointing
    # -----------------------------
    if params["ckpt_directory"] is None:
        params["ckpt_directory"] = params["model_outdir"]
    ckpt = CandleCkptPyTorch(params)
    ckpt.set_model({"model": model, "optimizer": optimizer})
    J = ckpt.restart(model)
    if J is not None:
        initial_epoch = J["epoch"]
        print("restarting from ckpt: initial_epoch: %i" % initial_epoch)

    # -----------------------------
    # Train
    # -----------------------------
    metrics = ["mse", "rmse", "pcc", "scc"]

    # Settings for early stop and best model settings
    best_score = np.inf
    best_epoch = -1
    early_stop_counter = 0  # define early-stop counter
    early_stop_metric = "mse"  # metric for early stop

    # Iterate over epochs
    for epoch in range(num_epoch):
        train_loss = train(model, device, train_loader, optimizer, loss_fn, epoch + 1, log_interval)
        ckpt.ckpt_epoch(epoch, train_loss) # checkpoints the best model by default

        # Predict with val data
        val_true, val_pred = frm.predicting(model, device, val_loader)  # val_true (groud truth), val_pred (predictions)
        val_scores = compute_metrics(val_true, val_pred, metrics)

        # For early stop
        if val_scores[early_stop_metric] < best_score:
            #torch.save(model.state_dict(), model_path)
            torch.save(model.state_dict(), outdtd["model"])
            best_epoch = epoch + 1
            best_score = val_scores[early_stop_metric]
            print(f"{early_stop_metric} improved at epoch {best_epoch}; Best \
                  {early_stop_metric}: {best_score}; Model: {params['model_arch']}")
            early_stop_counter = 0  # zero the early-stop counter if the model improved after the epoch
        else:
            print(f"No improvement since epoch {best_epoch}; Best \
                  {early_stop_metric}: {best_score}; Model: {params['model_arch']}")
            early_stop_counter += 1  # increment the counter if the model was not improved after the epoch

        if early_stop_counter == params["patience"]:
            print(f"Terminate training (model did not improve on val data for {params['patience']} epochs).")
            continue

    # -----------------------------
    # Load the (best) saved model (as determined based on val data)
    del model
    model = str2Class(params["model_arch"]).to(device)
    #model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(outdtd["model"]))
    model.eval()

    # -----------------------------
    # [Req]
    # Compute raw predictions
    pred_col_name = params["y_col_name"] + params["pred_col_name_suffix"]
    true_col_name = params["y_col_name"] + "_true"
    val_true, val_pred = frm.predicting(model, device, val_loader)  # G (groud truth), P (predictions)
    # tp = pd.DataFrame({true_col_name: G_test, pred_col_name: P_test})  # This includes true and predicted values
    pred_df = pd.DataFrame(val_pred, columns=[pred_col_name])  # This includes only predicted values

    # -----------------------------
    # [Req]
    # Concat raw predictions with the cancer and drug ids, and the true values
    # RSP_FNAME = "val_response.csv"
    # rsp_df = pd.read_csv(Path(args.test_ml_data_dir)/RSP_FNAME)
    #rsp_df = pd.read_csv(params["val_ml_data_dir"] + params["val_y_data_file_name"])
    rsp_df = pd.read_csv(indtd["val"])
    mm = pd.concat([rsp_df, pred_df], axis=1)
    mm = mm.astype({params["y_col_name"]: np.float32, pred_col_name: np.float32})

    # Save the raw predictions on val data
    # pred_fname = "test_preds.csv"
    #params["val_pred_file_path"] = model_outdir/params["val_pred_file_name"]
    # print(params["val_pred_file_path"])
    #improve_utils.save_preds(mm, params, params["val_pred_file_path"])
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
    val_scores = compute_metrics(y_true, val_pred, metrics)
    val_scores["val_loss"] = val_scores["mse"]

    # Performance scores for Supervisor HPO
    print("\nIMPROVE_RESULT val_loss:\t{}\n".format(val_scores["mse"]))
    #with open(model_outdir/params["json_val_scores"], "w", encoding="utf-8") as f:
    #    json.dump(val_scores, f, ensure_ascii=False, indent=4)
    with open(outdtd["scores"], "w", encoding="utf-8") as f:
        json.dump(val_scores, f, ensure_ascii=False, indent=4)

    print("Validation scores:\n\t{}".format(val_scores))
    return val_scores


def main():
    params = frm.initialize_parameters(filepath, default_model="graphdrp_default_model.txt")
    run(params)
    print("\nFinished training of graphDRP model.")


if __name__ == "__main__":
    main()
