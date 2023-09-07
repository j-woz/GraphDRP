"""Functionality for Training a GraphDRP Model."""

from pathlib import Path
import os
import json
import warnings

import numpy as np
import pandas as pd

from typing import Dict, Union

import torch
from torch_geometric.data import DataLoader

from improve import framework as frm
from candle import build_pytorch_optimizer, get_pytorch_function, keras_default_config, CandleCkptPyTorch

from improve.torch_utils import TestbedDataset
from improve.metrics import compute_metrics
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet

from graphdrp_preprocess_improve import gdrp_data_conf

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

gdrp_train_conf = [
    {"name": "val_data_df",
     "default": frm.SUPPRESS,
     "type": str,
     "help": "Data frame with original validation response data."
    },
]

req_train_args = ["model_arch", "model_outdir", "train_ml_data_dir", "val_ml_data_dir", "train_data_processed", "val_data_processed"]


def str2Class(str):
    """Get model class from model name (str)."""
    return globals()[str]()


def check_data_available(params: Dict) -> frm.DataPathDict:
    """
    Sweep the expected input paths and check that files needed in training are available.

    :param Dict params: Dictionary of parameters read

    :return: Path to directories requested stored in dictionary with str key str and Path value.
    :rtype: DataPathDict
    """
    # Expected
    # train_ml_data_dir / processed / train_data_processed
    # train_data_processed --> it has pt extension and is located inside a 'processed' folder
    # train_ml_data_dir / processed / train_data_processed
    # train_data_processed --> it has pt extension and is located inside a 'processed' folder
    # Make sure that the train data exists
    itrainpath = Path(params["train_ml_data_dir"]) / "processed"
    if itrainpath.exists() == False:
        raise Exception(f"ERROR ! Processed training data folder {itrainpath} not found.\n")
    itrain = itrainpath / params["train_data_processed"]
    if itrain.exists() == False:
        raise Exception(f"ERROR ! Processed training data {itrain} not found.\n")
    # Make sure that the val data exists
    ivalpath = Path(params["val_ml_data_dir"]) / "processed"
    if ivalpath.exists() == False:
        raise Exception(f"ERROR ! Processed validation data folder {ivalpath} not found.\n")
    ival = ivalpath / params["val_data_processed"]
    if ival.exists() == False:
        raise Exception(f"ERROR ! Processed validation data {ival} not found.\n")
    # Check if validation data frame exists
    ivaldfpath = None
    if "val_data_df" in params:
        ivaldfpath = Path(params["val_ml_data_dir"]) / params["val_data_df"]
        if ivaldfpath.exists() == False:
            ivaldfpath = None
            message = (
                       f"Data frame with original validation response data: {params['val_data_df']} not found." \
                       + " Will continue but will only store partial (available) data frame.\n"
            )
            warnings.warn(message, RuntimeWarning)
    else:
        message = (
                   f"Data frame with original validation response data not specified (no 'val_data_df' keyword)." \
                       + " Will continue but will only store partial (available) data frame.\n"
            )
        warnings.warn(message, RuntimeWarning)

    # Create output directory. Do not complain if it exists.
    opath = Path(params["model_outdir"])
    os.makedirs(opath, exist_ok=True)
    modelpath = opath / params["model_params"]
    fname = f"val_{params['model_eval_suffix']}.csv"
    if ivaldfpath is None:
        fname = f"val_{params['model_eval_suffix']}_partial.csv"
    predpath = opath / fname
    fname = f"val_{params['json_scores_suffix']}.json"
    scorespath = opath / fname

    # Return in DataPathDict structure
    inputdtd = {"train": itrainpath, "val": ivalpath, "df": ivaldfpath}
    outputdtd = {"model": modelpath, "pred": predpath, "scores": scorespath}

    return inputdtd, outputdtd


class Trainer:
    def __init__(self, params, device, modelpath, metrics=None):
        # -----------------------------
        # Create and move model to device
        self.model_arch = params["model_arch"]
        self.model = str2Class(self.model_arch).to(device)
        self.params = params
        self.device = device
        self.modelpath = modelpath
        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = ["mse", "rmse", "pcc", "scc"]

    def setup_train(self,):
        # Construct DL optimizer and loss
        keras_defaults = keras_default_config()
        self.optimizer = build_pytorch_optimizer(
            model=self.model,
            optimizer=self.params["optimizer"],
            lr=self.params["learning_rate"],
            kerasDefaults=keras_defaults
        )
        self.loss_fn = get_pytorch_function(self.params["loss"])
        # Train parameters
        self.num_epoch = self.params["epochs"]
        self.log_interval = self.params["log_interval"]
        self.patience = self.params["patience"]


    def config_checkpointing(self, ckpt_directory):
        self.params["ckpt_directory"] = ckpt_directory
        initial_epoch = 0
        self.ckpt = CandleCkptPyTorch(self.params)
        self.ckpt.set_model({"model": self.model, "optimizer": self.optimizer})
        J = self.ckpt.restart(self.model)
        if J is not None:
            initial_epoch = J["epoch"]
            print("restarting from ckpt: initial_epoch: %i" % initial_epoch)

        return initial_epoch

    def train(self, train_loader, epoch):
        print("Training on {} samples...".format(len(train_loader.dataset)))
        self.model.train()
        avg_loss = []
        for batch_idx, data in enumerate(train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output, _ = self.model(data)
            loss = self.loss_fn(output, data.y.view(-1, 1).float().to(self.device))
            loss.backward()
            self.optimizer.step()
            avg_loss.append(loss.item())
            if batch_idx % self.log_interval == 0:
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

    def execute_train(self, train_loader, val_loader, initial_epoch):

        # Settings for early stop and best model settings
        best_score = np.inf
        best_epoch = -1
        early_stop_counter = 0  # define early-stop counter
        early_stop_metric = "mse"  # metric for early stop

        # Iterate over epochs
        for epoch in range(initial_epoch, self.num_epoch):
            train_loss = self.train(train_loader, epoch + 1)
            self.ckpt.ckpt_epoch(epoch, train_loss) # checkpoints the best model by default

            # Predict with val data
            val_true, val_pred = frm.predicting(self.model, self.device, val_loader)  # val_true (groud truth), val_pred (predictions)
            val_scores = compute_metrics(val_true, val_pred, self.metrics)

            # For early stop
            if val_scores[early_stop_metric] < best_score:
                torch.save(self.model.state_dict(), self.modelpath)
                best_epoch = epoch + 1
                best_score = val_scores[early_stop_metric]
                print(f"{early_stop_metric} improved at epoch {best_epoch}; Best \
                  {early_stop_metric}: {best_score}; Model: {self.model_arch}")
                early_stop_counter = 0  # zero the early-stop counter if the model improved after the epoch
            else:
                print(f"No improvement since epoch {best_epoch}; Best \
                  {early_stop_metric}: {best_score}; Model: {self.model_arch}")
                early_stop_counter += 1  # increment the counter if the model was not improved after the epoch

            if early_stop_counter == self.patience:
                print(f"Terminate training (model did not improve on val data for {self.patience} epochs).")
                continue


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


def determine_device(cuda_name_from_params):
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
            cuda_name = cuda_name_from_params
        device = cuda_name
    else:
        device = "cpu"

    return device


def build_PT_data_loader(datadir: str, datafname: str, batch: int, shuffle: bool):
    data_file_name = datafname
    if data_file_name.endswith(".pt"):
        data_file_name = data_file_name[:-3] # TestbedDataset() appends this string with ".pt"

    dataset = TestbedDataset(root=datadir, dataset=data_file_name) # TestbedDataset() requires strings

    # PyTorch dataloader
    loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle)
    return loader


def evaluate_model(model_arch, device, modelpath, val_loader):
    model = str2Class(model_arch).to(device)
    model.load_state_dict(torch.load(modelpath))
    model.eval()
    val_true, val_pred = frm.predicting(model, device, val_loader)  # (groud truth), (predictions)

    return val_true, val_pred


def store_predictions_df(params, indtd, outdtd, y_true, y_pred):
    pred_col_name = params["y_col_name"] + params["pred_col_name_suffix"]
    true_col_name = params["y_col_name"] + "_true"
    # -----------------------------
    # Attempt to concat raw predictions with the cancer and drug ids, and the true values
    if indtd["df"] is not None:
        rsp_df = pd.read_csv(indtd["df"])

        pred_df = pd.DataFrame(y_pred, columns=[pred_col_name])  # This includes only predicted values

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
        y_true_return = rsp_df[params["y_col_name"]].values # Read from data frame
        print("Stored orig drug, cell and evaluation in: ", outdtd["pred"])
    else: # Save only ground truth and predictions since cancer and drug ids are not available
        df_ = pd.DataFrame({true_col_name: y_true, pred_col_name: y_pred})  # This includes true and predicted values
        # Save preds df
        df_.to_csv(outdtd["pred"], index=False)
        y_true_return = y_true
        print("Stored only evaluation in: ", outdtd["pred"])

    return y_true_return


def compute_performace_scores(y_true, y_pred, metrics, outdtd, stage):
    scores = compute_metrics(y_true, y_pred, metrics)
    key = f"{stage}_loss"
    scores[key] = scores["mse"]

    with open(outdtd["scores"], "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    # Performance scores for Supervisor HPO
    if stage == "val":
        print("\nIMPROVE_RESULT val_loss:\t{}\n".format(scores["mse"]))
        print("Validation scores:\n\t{}".format(scores))
    elif stage == "test":
        print("Inference scores:\n\t{}".format(scores))
    return scores


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
    # indtd is dictionary with input_description: path components
    # outdtd is dictionary with output_description: path components

    # -----------------------------
    # [Req]
    # Determine CUDA/CPU device and configure CUDA device if available
    device = determine_device(params["cuda_name"])

    # -------------------------------------
    # Create Trainer object and setup train
    # -------------------------------------
    trobj = Trainer(params, device, outdtd["model"])
    trobj.setup_train()
    # -----------------------------
    # Set checkpointing
    # -----------------------------
    if params["ckpt_directory"] is None:
        params["ckpt_directory"] = params["model_outdir"]
    initial_epoch = trobj.config_checkpointing(params["ckpt_directory"])

    # -----------------------------
    # Prepare PyTorch dataloaders
    train_loader = build_PT_data_loader(params["train_ml_data_dir"],
                                        params["train_data_processed"],
                                        params["batch_size"],
                                        shuffle=True)

    # Note! Don't shuffle the val_loader or results will be corrupted
    val_loader = build_PT_data_loader(params["val_ml_data_dir"],
                                        params["val_data_processed"],
                                        params["val_batch"],
                                        shuffle=False)

    # -----------------------------
    # Train
    # -----------------------------
    trobj.execute_train(train_loader, val_loader, initial_epoch)

    # -----------------------------
    # Load the (best) saved model (as determined based on val data)
    # Compute predictions
    # (groud truth), (predictions)
    val_true, val_pred = evaluate_model(params["model_arch"], device, outdtd["model"], val_loader)

    # Store predictions in data frame
    # Attempt to concat predictions with the cancer and drug ids, and the true values
    # If data frame found, then y_true is read from data frame and returned
    # Otherwise, only a partial data frame is stored (with val_true and val_pred)
    # and y_true is equal to pytorch loaded val_true
    y_true = store_predictions_df(params, indtd, outdtd, val_true, val_pred)
    # Compute performance scores
    metrics = ["mse", "rmse", "pcc", "scc", "r2"]
    val_scores = compute_performace_scores(y_true, val_pred, metrics, outdtd, "val")

    return val_scores


def main():
    params = frm.initialize_parameters(filepath,
                                       default_model="graphdrp_default_model.txt",
                                       additional_definitions = gdrp_model_conf + gdrp_data_conf + gdrp_train_conf,
                                       required = req_train_args,
                                      )
    run(params)
    print("\nFinished training GraphDRP model.")


if __name__ == "__main__":
    main()
