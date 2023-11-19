""" Functionality for Training a GraphDRP Model. """

import os
import json
import warnings
from pathlib import Path
from pprint import pformat
from typing import Dict, Union

import numpy as np
import pandas as pd

import torch
from torch_geometric.data import DataLoader

# IMPROVE/CANDLE imports
from improve import framework as frm
from improve.metrics import compute_metrics
from improve import drug_resp_pred as drp
from candle import build_pytorch_optimizer, get_pytorch_function, keras_default_config, CandleCkptPyTorch

# Model-specific imports
# from improve.torch_utils import TestbedDataset  # ap
from model_utils.torch_utils import TestbedDataset  # ap
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet

from model_utils.classlogger import Logger, get_print_func

# from graphdrp_preprocess_improve import gdrp_data_conf  # ap
from graphdrp_preprocess_improve import model_preproc_params, drp_preproc_params, preprocess_params  # ap

filepath = Path(__file__).resolve().parent

# Params that are specific to graphDRP model
# gdrp_model_conf = [
model_train_params = [
    {"name": "model_arch",
     "default": "GINConvNet",
     "choices": ["GINConvNet", "GATNet", "GAT_GCN", "GCNNet"],
     "type": str,
     "help": "Model architecture to run."},
    {"name": "log_interval",
     "action": "store",
     "type": int,
     "help": "Interval for saving o/p"},
    {"name": "cuda_name",  # TODO: how we should control this?
     "action": "store",
     "type": str,
     "help": "Cuda device (e.g.: cuda:0, cuda:1."},
]

# gdrp_train_conf = [
drp_train_params = [
    {"name": "val_data_df",  # TODO: app or frm level?
     "default": frm.SUPPRESS,
     "type": str,
     "help": "Data frame with original validation response data."
    },
]

# train_params = model_train_params + drp_train_params  # ap

req_train_args = ["model_arch", "model_outdir",
                  "train_ml_data_dir", "val_ml_data_dir",
                  # "train_data_processed", "val_data_processed"]
                  # "train_ml_data_fname", "val_ml_data_fname"]
                  ]


def str2Class(str):
    """Get model class from model name (str)."""
    return globals()[str]()


def check_data_available(params: Dict) -> frm.DataPathDict:
    """
    Sweep the expected input paths and check that files needed in training are available.

    :param Dict params: Dictionary of CANDLE/IMPROVE parameters read.

    :return: Path to directories requested stored in dictionary with str key str and Path value.
    :rtype: DataPathDict
    """
    # Expected
    # train_ml_data_dir / processed / train_data_processed
    # train_data_processed --> it has pt extension and is located inside a 'processed' folder
    # train_ml_data_dir / processed / train_data_processed
    # train_data_processed --> it has pt extension and is located inside a 'processed' folder

    # import pdb; pdb.set_trace()

    # Make sure that the train data exists
    itrainpath = Path(params["train_ml_data_dir"]) / "processed"
    if itrainpath.exists() == False:
        raise Exception(f"ERROR ! Processed training data folder {itrainpath} not found.\n")

    # itrain = itrainpath / params["train_data_processed"]
    itrain = itrainpath / (params["train_ml_data_fname"] + ".pt")
    if itrain.exists() == False:
        raise Exception(f"ERROR ! Processed training data {itrain} not found.\n")

    # Make sure that the val data exists
    ivalpath = Path(params["val_ml_data_dir"]) / "processed"
    if ivalpath.exists() == False:
        raise Exception(f"ERROR ! Processed validation data folder {ivalpath} not found.\n")

    # ival = ivalpath / params["val_data_processed"]
    ival = ivalpath / (params["val_ml_data_fname"] + ".pt")
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

    # import pdb; pdb.set_trace()

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


def check_train_data_available(params: Dict) -> frm.DataPathDict:
    """
    Sweep the expected input paths and check that files needed in training are available.

    :param Dict params: Dictionary of CANDLE/IMPROVE parameters read.

    :return: Path to directories requested stored in dictionary with str key str and Path value.
    :rtype: DataPathDict
    """
    # Expected
    # train_ml_data_dir / processed / train_data_processed
    # train_data_processed --> it has pt extension and is located inside a 'processed' folder
    # train_ml_data_dir / processed / train_data_processed
    # train_data_processed --> it has pt extension and is located inside a 'processed' folder

    # import pdb; pdb.set_trace()

    # Make sure that the train data exists
    itrainpath = Path(params["train_ml_data_dir"])
    if itrainpath.exists() == False:
        raise Exception(f"ERROR ! Processed training data folder {itrainpath} not found.\n")

    # Make sure that the val data exists
    ivalpath = Path(params["val_ml_data_dir"])
    if ivalpath.exists() == False:
        raise Exception(f"ERROR ! Processed validation data folder {ivalpath} not found.\n")

    train_data_fname = frm.build_ml_data_name(params, stage="train",
                                              data_format=params["data_format"])
    itrain = itrainpath / train_data_fname
    if itrain.exists() == False:
        raise Exception(f"ERROR ! Processed training data {itrain} not found.\n")

    val_data_fname = frm.build_ml_data_name(params, stage="val",
                                            data_format=params["data_format"])
    ival = ivalpath / val_data_fname
    if ival.exists() == False:
        raise Exception(f"ERROR ! Processed validation data {ival} not found.\n")

    # [Req] Save y data dataframe for the current stage
    train_data_df = f"train_{params['y_data_suffix']}.csv"
    val_data_df = f"val_{params['y_data_suffix']}.csv"
    train_data_df_path = itrainpath / f"train_{params['y_data_suffix']}.csv"
    val_data_df_path = ivalpath /f"val_{params['y_data_suffix']}.csv"

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

    # import pdb; pdb.set_trace()

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


def config_checkpointing(params: Dict, model, optimizer):
    """Configure CANDLE checkpointing. Reads last saved state if checkpoints exist.

    :params str ckpt_directory: String with path to directory for storing the CANDLE checkpointing for the model being trained.

    :return: Number of training iterations already run (this may be > 0 if reading from checkpointing).
    :rtype: int
    """
    # params["ckpt_directory"] = ckpt_directory
    initial_epoch = 0
    # TODO. This creates directory self.params["ckpt_directory"]
    # import pdb; pdb.set_trace()
    ckpt = CandleCkptPyTorch(params)
    ckpt.set_model({"model": model, "optimizer": optimizer})
    J = ckpt.restart(model)
    if J is not None:
        initial_epoch = J["epoch"]
        print("restarting from ckpt: initial_epoch: %i" % initial_epoch)
    return ckpt, initial_epoch


def train_tmp(model, device, train_loader, optimizer, epoch, log_interval):
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



def train_epoch(model, device, train_loader, optimizer, loss_fn, epoch,
                log_interval, verbose=True):
# def train_epoch(train_loader, epoch):
    """Execute a training epoch (i.e. one pass through training set).

    :params DataLoader train_loader: PyTorch data loader with training data.
    :params int epoch: Current training epoch (for display purposes only).

    :return: Average loss for executed training epoch.
    :rtype: float
    """
    print("Training on {} samples...".format(len(train_loader.dataset)))
    model.train()
    # Below is the train() from the original GraphDRP model
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


def train_graphdrp(params: Dict):
    """User-defined training function that runs on each distributed worker process.

    This function typically contains logic for loading the model,
    loading the dataset, training the model, saving checkpoints,
    and logging metrics.
    """
    # Model, Loss, Optimizer
    # __init__()
    model_arch = params["model_arch"]
    model = str2Class(model_arch).to(device)
    device = determine_device(params["cuda_name"])
    # setup_train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # loss_fn = get_pytorch_function(params["loss"])
    loss_fn = torch.nn.MSELoss()

    # Data (Prepare PyTorch dataloaders)
    # -----------------------------
    # Prepare PyTorch dataloaders
    print("\nTraining data:")
    print(f"train_ml_data_dir: {params['train_ml_data_dir']}")
    print(f"train_ml_data_fname: {params['train_ml_data_fname']}")
    print(f"batch_size: {params['batch_size']}")
    # import pdb; pdb.set_trace()
    train_loader = build_GraphDRP_dataloader(
        params["train_ml_data_dir"],
        # params["train_data_processed"],
        params["train_ml_data_fname"],
        params["batch_size"],
        shuffle=True)

    # Note! Don't shuffle the val_loader or results will be corrupted
    print("\nVal data:")
    print(f"val_ml_data_dir: {params['val_ml_data_dir']}")
    print(f"val_ml_data_fname: {params['val_ml_data_fname']}")
    print(f"val_batch: {params['val_batch']}")
    # import pdb; pdb.set_trace()
    val_loader = build_GraphDRP_dataloader(
        params["val_ml_data_dir"],
        params["val_ml_data_fname"],
        params["val_batch"],
        shuffle=False)

    # Train. Prep settings.
    if params["ckpt_directory"] is None:
        params["ckpt_directory"] = params["model_outdir"]
    # import ipdb; ipdb.set_trace()
    # initial_epoch = trobj.config_checkpointing(params["ckpt_directory"])
    initial_epoch = config_checkpointing(params, model, optimizer)

    num_epoch = params["epochs"]
    log_interval = params["log_interval"]
    patience = params["patience"]
    model.train()

    # Settings for early stop and best model settings
    best_score = np.inf
    best_epoch = -1
    early_stop_counter = 0  # define early-stop counter
    # early_stop_metric = "mse"  # metric for early stop

    # Train. Iterate over epochs.
    for epoch in range(initial_epoch, num_epoch):
        train_loss = train(train_loader, epoch + 1)
        ckpt.ckpt_epoch(epoch, train_loss) # checkpoints the best model by default

        # Predict with val data
        val_true, val_pred = predicting(model, device, val_loader)  # val_true (groud truth), val_pred (predictions)
        val_scores = compute_metrics(val_true, val_pred, metrics)

        # For early stop
        if val_scores[early_stop_metric] < best_score:
            torch.save(model.state_dict(), modelpath)
            best_epoch = epoch + 1
            best_score = val_scores[early_stop_metric]
            print(f"{early_stop_metric} improved at epoch {best_epoch}; Best \
              {early_stop_metric}: {best_score}; Model: {model_arch}")
            early_stop_counter = 0  # zero the early-stop counter if the model improved after the epoch
        else:
            print(f"No improvement since epoch {best_epoch}; Best \
              {early_stop_metric}: {best_score}; Model: {model_arch}")
            early_stop_counter += 1  # increment the counter if the model was not improved after the epoch

        if early_stop_counter == patience:
            print(f"Terminate training (model did not improve on val data for {self.patience} epochs).")
            continue
    return model



# TODO. Should each model implement this class? What about keras or TF models?
# TODO. Consider moving this to ./model_utils
class Trainer:
    """Class to define a PyTorch interface for training models."""
    def __init__(self, params, device, modelpath, metrics=None):
        """Initialize a Trainer object.

        :params Dict params: Dictionary of CANDLE/IMPROVE parameters read.
        :params str device: String with PyTorch format describing device available for training.
        :params Path modelpath: Path to store model. Currently this is complementary
                to checkpointing, i.e. models are saved directly and also with CANDLE
                checkpointing. This redundacy should be re-evaluated.
        :params List metrics: List of strings specifying the functions to evaluate the
                model, e.g. "mse", "pcc", etc. Default: None which is converted to:
                ["mse", "rmse", "pcc", "scc"].
        """
        # -----------------------------
        # Create and move model to device
        self.params = params
        self.model_arch = params["model_arch"]
        self.model = str2Class(self.model_arch).to(device)
        self.device = device
        self.modelpath = modelpath
        if metrics is not None:  # TODO: not sure if need this.
            self.metrics = metrics
        else:
            self.metrics = ["mse", "rmse", "pcc", "scc"]

    def setup_train(self,):
        """Configure the Trainer object.

        This function constructs the optimizer and loss and extract
        trainer parameters for number of epochs, interval of logging
        model evaluation and patience for early stopping.
        """
        # Construct DL optimizer and loss
        keras_defaults = keras_default_config()  # TODO: keras?
        # TODO: build_pytorch_optimizer() supports a limited number of optimizers.
        # What if the model needs a different optimizer?
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
        """Configure CANDLE checkpointing. Reads last saved state if checkpoints exist.

        :params str ckpt_directory: String with path to directory for storing the CANDLE checkpointing for the model being trained.

        :return: Number of training iterations already run (this may be > 0 if reading from checkpointing).
        :rtype: int
        """
        self.params["ckpt_directory"] = ckpt_directory
        initial_epoch = 0
        # TODO. This creates directory self.params["ckpt_directory"]
        import pdb; pdb.set_trace()
        self.ckpt = CandleCkptPyTorch(self.params)
        self.ckpt.set_model({"model": self.model, "optimizer": self.optimizer})
        J = self.ckpt.restart(self.model)
        if J is not None:
            initial_epoch = J["epoch"]
            print("restarting from ckpt: initial_epoch: %i" % initial_epoch)

        return initial_epoch

    def train(self, train_loader, epoch):
        """ Execute a training epoch (i.e. one pass through training set).

        :params DataLoader train_loader: PyTorch data loader with training data.
        :params int epoch: Current training epoch (for display purposes only).

        :return: Average loss for executed training epoch.
        :rtype: float
        """
        print("Training on {} samples...".format(len(train_loader.dataset)))
        self.model.train()
        # Below is the train() from the original GraphDRP model
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

    def execute_train(self, train_loader, val_loader, initial_epoch, early_stop_metric="mse"):
        """ Execute a training loop, i.e. perform multiple passes through training set,
        updating model parameters, storing them (directly and via CANDLE checkpointing),
        evaluating model and enacting early stopping (if applicable, i.e. if for a
        `patience` number of iterations the selected metric does not improve when evaluating
        over the validation set).

        :params DataLoader train_loader: PyTorch data loader with training data.
        :params DataLoader val_loader: PyTorch data loader with validation data. Used for early stopping.
        :params int initial_epoch: Number of training iterations already run (this may be > 0 if reading from checkpointing).
        """

        # Settings for early stop and best model settings
        best_score = np.inf
        best_epoch = -1
        early_stop_counter = 0  # define early-stop counter
        # early_stop_metric = "mse"  # metric for early stop

        # Iterate over epochs
        for epoch in range(initial_epoch, self.num_epoch):
            train_loss = self.train(train_loader, epoch + 1)
            self.ckpt.ckpt_epoch(epoch, train_loss) # checkpoints the best model by default

            # Predict with val data
            val_true, val_pred = predicting(self.model, self.device, val_loader)  # val_true (groud truth), val_pred (predictions)
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


# TODO. Consider moving this to ./improve/drug_resp_pred
def save_preds(df: pd.DataFrame,
               # canc_col_name: str,
               # drug_col_name: str,
               y_col_name: str,
               pred_col_name_suffix: str,
               outpath: Union[str, Path],
               round_decimals: int=4) -> None:
    """ Save model predictions.

    This function throws errors if the dataframe does not include the expected
    columns: canc_col_name, drug_col_name, y_col_name, y_col_name + "_pred"

    :params pd.DataFrame df: Pandas data frame with model predictions
    :params str canc_col_name: Column name that contains the cancer sample ids.
    :params str drug_col_name: Column name that contains the drug ids.
    :params str y_col_name: drug response col name (e.g., IC50, AUC)
    :params str pred_col_name_suffix: Suffix to identoy column of predictions made by model.
    :params str or PosixPath outpath: outdir to save the model predictions df.
    :params int round: round response values.

    :return: ?.
    :rtype: None
    """
    # Check that the 4 columns exist
    # assert canc_col_name in df.columns, f"{canc_col_name} was not found in columns."
    # assert drug_col_name in df.columns, f"{drug_col_name} was not found in columns."
    assert y_col_name in df.columns, f"{y_col_name} was not found in columns."
    pred_col_name = y_col_name + f"{pred_col_name_suffix}"
    assert pred_col_name in df.columns, f"{pred_col_name} was not found in columns."

    # Round
    df = df.round({y_col_name: round_decimals, pred_col_name: round_decimals})

    # Save preds df
    df.to_csv(outpath, index=False)
    return None


# TODO. model;
def determine_device(cuda_name_from_params):
    """Determine device to run PyTorch functions.

    PyTorch functions can run on CPU or on GPU. In the latter case, it
    also takes into account the GPU devices requested for the run.

    :params str cuda_name_from_params: GPUs specified for the run.

    :return: Device available for running PyTorch functionality.
    :rtype: str
    """
    cuda_avail = torch.cuda.is_available()
    print("GPU available: ", cuda_avail)
    if cuda_avail:  # GPU available
        # -----------------------------
        # CUDA device from env var
        cuda_env_visible = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_env_visible is not None:
            # Note! When one or multiple device numbers are passed via
            # CUDA_VISIBLE_DEVICES, the values in python script are reindexed
            # and start from 0.
            print("CUDA_VISIBLE_DEVICES: ", cuda_env_visible)
            cuda_name = "cuda:0"
        else:
            cuda_name = cuda_name_from_params
        device = cuda_name
    else:
        device = "cpu"

    return device


# TODO. model-specific?
def build_PT_data_loader(
        datadir: str,
        datafname: str,
        batch: int,
        shuffle: bool):
    """ Build a PyTorch data loader.

    :params str datadir: Directory where `processed` folder containing
            processed data can be found.
    :params str datafname: Name of PyTorch processed data to read.
    :params int batch: Batch size for data loader.
    :params bool shuffle: Flag to specify if data is to be shuffled when
            applying data loader.

    :return: PyTorch data loader constructed.
    :rtype: DataLoader
    """
    data_file_name = datafname
    if data_file_name.endswith(".pt"):
        data_file_name = data_file_name[:-3] # TestbedDataset() appends this string with ".pt"
    dataset = TestbedDataset(root=datadir, dataset=data_file_name) # TestbedDataset() requires strings
    loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle) # PyTorch dataloader
    return loader


# TODO. model-specific?
def build_GraphDRP_dataloader(
        data_dir: str,
        data_fname: str,
        batch_size: int,
        shuffle: bool):
    """ Build a PyTorch data loader.

    :params str datadir: Directory where `processed` folder containing
            processed data can be found.
    :params str datafname: Name of PyTorch processed data to read.
    :params int batch: Batch size for data loader.
    :params bool shuffle: Flag to specify if data is to be shuffled when
            applying data loader.

    :return: PyTorch data loader constructed.
    :rtype: DataLoader
    """
    if data_fname.endswith(".pt"):
        data_fname = data_fname[:-3] # TestbedDataset() appends this string with ".pt"
    dataset = TestbedDataset(root=data_dir, dataset=data_fname) # TestbedDataset() requires strings
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)  # PyTorch dataloader
    return loader


# TODO. model-specific?
# TODO. Should each model implement this class? What about keras or TF models?
# TODO. Consider moving this to ./model_utils
# def evaluate_model(model_arch, device, modelpath, data_loader):
def evaluate_model(params, device, modelpath, data_loader):
    """Load the model and perform predictions using given model.

    :params str model_arch: Name of model architecture to use.
    :params str device: Device to use for evaluating PyTorch model.
    :params Path modelpath: Path containing model parameters.
    :params DataLoader data_loader: PyTorch data loader with data to
            use for evaluation.

    :return: Arrays with ground truth and model predictions.
    :rtype: np.array
    """
    # TODO. consider to create func load_model() or load_graphdrp()
    # Load model
    # model = load_model(params)
    # model = str2Class(model_arch).to(device)
    model = str2Class(params["model_arch"]).to(device)
    model.load_state_dict(torch.load(modelpath))
    model.eval()
    # Compute predictions
    # val_true, val_pred = predict(model, data_loader=val_loader, device=device)
    val_true, val_pred = predicting(model, device, data_loader)  # (groud truth), (predictions)

    return val_true, val_pred


# TODO. Consider moving this to ./improve/drug_resp_pred
# def store_predictions_df(params, indtd, outdtd, y_true, y_pred):
def store_predictions_df(params, indtd, outdtd, y_true, y_pred):
    """Store predictions with accompanying data frame.

    This allows to trace original data evaluated (e.g. drug and cell
    combinations) if corresponding data frame is available, in which case
    the whole structure as well as the model predictions are stored. If
    the data frame is not available, only ground truth (read from the
    PyTorch processed data) and model predictions are stored. The ground
    truth available (from data frame or PyTorch data) is returned for
    further evaluations.

    :params Dict params: Dictionary of CANDLE/IMPROVE parameters read.
    :params Dict indtd: Dictionary specifying paths of input data.
    :params Dict outdtd: Dictionary specifying paths for ouput data.
    :params array y_true: Ground truth.
    :params array y_pred: Model predictions.

    :return: Arrays with ground truth. This may have been read from an
             input data frame or from a processed PyTorch data file.
    :rtype: np.array
    """
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


# def store_preds_df(params, y_true, y_pred):
def store_preds_df(params: Dict, y_true, y_pred, stage: str):
    """Store predictions with accompanying data frame.

    This allows to trace original data evaluated (e.g. drug and cell
    combinations) if corresponding data frame is available, in which case
    the whole structure as well as the model predictions are stored. If
    the data frame is not available, only ground truth (read from the
    PyTorch processed data) and model predictions are stored. The ground
    truth available (from data frame or PyTorch data) is returned for
    further evaluations.

    ap: construct output file name as follows:
            
            [stage]_[params['y_data_suffix']]_

    :params Dict params: Dictionary of CANDLE/IMPROVE parameters read.
    :params Dict indtd: Dictionary specifying paths of input data.
    :params Dict outdtd: Dictionary specifying paths for ouput data.
    :params array y_true: Ground truth.
    :params array y_pred: Model predictions.

    :return: Arrays with ground truth. This may have been read from an
             input data frame or from a processed PyTorch data file.
    :rtype: np.array
    """
    # Check dimensions
    assert len(y_true) == len(y_pred), f"length of y_true ({len(y_true)}) and y_pred ({len(y_pred)}) don't match"
    print(len(y_true))
    print(len(y_pred))

    # Define column names
    pred_col_name = params["y_col_name"] + params["pred_col_name_suffix"]
    true_col_name = params["y_col_name"] + "_true"

    # -----------------------------
    # Attempt to concatenate raw predictions with y dataframe (e.g., df that
    # contains cancer ids, drug ids, and the true response values)
    ydf_fname = f"{stage}_{params['y_data_suffix']}.csv"  # TODO. f"{stage}_{params['y_data_stage_fname_suffix']}.csv"  
    ydf_fpath = Path(params["ml_data_outdir"]) / ydf_fname

    # if indtd["df"] is not None:
    if ydf_fpath.exists() is not None:
        # rsp_df = pd.read_csv(indtd["df"])
        rsp_df = pd.read_csv(ydf_fpath)
        print(rsp_df.shape)

        # Check dimensions
        assert len(y_true) == rsp_df.shape[0], f"length of y_true ({len(y_true)}) and the loaded file ({ydf_fpath} --> {rsp_df.shape[0]}) don't match"

        import ipdb; ipdb.set_trace()
        pred_df = pd.DataFrame(y_pred, columns=[pred_col_name])  # This includes only predicted values

        mm = pd.concat([rsp_df, pred_df], axis=1)
        mm = mm.astype({params["y_col_name"]: np.float32, pred_col_name: np.float32})

        # Save the raw predictions
        # TODO. check comment below!
        # Note that there is no guarantee that the results effectively correspond
        # to this pre-processing parameters or the specified data frame since the
        # data is being read from a processed pt file (not from original data frame)
        # save_preds(mm,
        #        params["canc_col_name"],
        #        params["drug_col_name"],
        #        params["y_col_name"],
        #        params["pred_col_name_suffix"],
        #        outdtd["pred"],
        # )
        save_preds(mm,
                   # params["canc_col_name"],
                   # params["drug_col_name"],
                   y_col_name=params["y_col_name"],
                   pred_col_name_suffix=params["pred_col_name_suffix"],
                   # outdtd["pred"],
                   outpath=ydf_fpath)
        y_true_return = rsp_df[params["y_col_name"]].values # Read from data frame
        print("Stored orig drug, cell and evaluation in: ", outdtd["pred"])
    else: # Save only ground truth and predictions since cancer and drug ids are not available
        df_ = pd.DataFrame({true_col_name: y_true, pred_col_name: y_pred})  # This includes true and predicted values
        # Save preds df
        df_.to_csv(outdtd["pred"], index=False)
        y_true_return = y_true
        print("Stored only evaluation in: ", outdtd["pred"])

    return y_true_return


# TODO. Consider moving this to ./improve/...
def compute_performace_scores(y_true, y_pred, metrics, outdtd, stage):
    """Evaluate predictions according to specified metrics.

    Metrics are evaluated. Scores are stored in specified path and returned.

    :params array y_true: Array with ground truth values.
    :params array y_pred: Array with model predictions.
    :params listr metrics: List of strings with metrics to evaluate.
    :params Dict outdtd: Dictionary with path to store scores.
    :params str stage: String specified if evaluation is with respect to
            validation or testing set.

    :return: Python dictionary with metrics evaluated and corresponding scores.
    :rtype: dict
    """
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


# TODO: While the implementation of this func is model-specific, we may want
# to require that all models have this func defined for their models. Also,
# we need to decide where this func should be located.
def predicting(model, device, loader):
    """ Method to make predictions/inference.
    This is used in *train.py and *infer.py

    Parameters
    ----------
    model : pytorch model
        Model to evaluate.
    device : string
        Identifier for hardware that will be used to evaluate model.
    loader : pytorch data loader.
        Object to load data to evaluate.

    Returns
    -------
    total_labels: numpy array
        Array with ground truth.
    total_preds: numpy array
        Array with inferred outputs.
    """
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print("Make prediction for {} samples...".format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _ = model(data)
            # Is this computationally efficient?
            total_preds = torch.cat((total_preds, output.cpu()), 0)  # preds to tensor
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)  # labels to tensor
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def run(params):
    """ Execute specified model training.

    :params: Dict params: A dictionary of CANDLE/IMPROVE keywords and parsed values.

    :return: List of floats evaluating model predictions according to
             specified metrics.
    :rtype: float list
    """
    # import pdb; pdb.set_trace()

    # [Req] Create output dir. TODO. frm.create_dir(params)?
    opath = Path(params["model_outdir"])
    os.makedirs(opath, exist_ok=True)
    modelpath = opath / params["model_params"]
    # modelpath = opath / params["model_file_name"]

    # lg = Logger(opath/'train_improve.log')
    # print_fn = get_print_func(lg.logger)
    # print_fn(f'File path: {filepath}')
    # print_fn(f'\n{pformat(params)}')
    print_fn = print

    # ------------------------------------------------------
    # Check data availability and create output directory
    # ------------------------------------------------------
    # [Req]
    # import pdb; pdb.set_trace()
    # indtd, outdtd = check_data_available(params)
    # indtd is dictionary with input_description: path components
    # outdtd is dictionary with output_description: path components

    # indtd, outdtd = check_train_data_available(params)

    # ------------------------------------------------------
    # [Req] Build paths
    # -----------------
    # import pdb; pdb.set_trace()
    train_data_fname = frm.build_ml_data_name(params, stage="train",
                                              data_format=params["data_format"])
    val_data_fname = frm.build_ml_data_name(params, stage="val",
                                            data_format=params["data_format"])
    # GraphDRP -- remove data_format
    train_data_fname = train_data_fname.split(params["data_format"])[0]
    val_data_fname = val_data_fname.split(params["data_format"])[0]
    # train_data_fname = params["train_ml_data_fname"]
    # val_data_fname = params["val_ml_data_fname"]
    print(train_data_fname)
    print(val_data_fname)
    # params = frm.build_paths(params)  # paths to raw data

    # Create outdir for ML data (to save preprocessed data)
    # preprocess_outdir = frm.create_ml_data_outdir(params)
    # preprocess_outdir = params["ml_data_outdir"]  # ml_data_dir
    # print_fn(f"preprocess_outdir {preprocess_outdir}")
    # -----------------

    # -----------------------------
    # Prepare PyTorch dataloaders
    # -----------------------------
    print_fn("\nTraining data:")
    print_fn(f"train_ml_data_dir: {params['train_ml_data_dir']}")
    # print_fn(f"train_data_processed: {params['train_data_processed']}")
    # print_fn(f"train_ml_data_fname: {params['train_ml_data_fname']}")
    print_fn(f"batch_size: {params['batch_size']}")
    # import pdb; pdb.set_trace()
    train_loader = build_GraphDRP_dataloader(params["train_ml_data_dir"],
                                             # params["train_data_processed"],
                                             train_data_fname,
                                             params["batch_size"],
                                             shuffle=True)

    # Note! Don't shuffle the val_loader or results will be corrupted
    print_fn("\nVal data:")
    print_fn(f"val_ml_data_dir: {params['val_ml_data_dir']}")
    # print_fn(f"val_data_processed: {params['val_data_processed']}")
    # print_fn(f"val_ml_data_fname: {params['val_ml_data_fname']}")
    print_fn(f"val_batch: {params['val_batch']}")
    # import pdb; pdb.set_trace()
    val_loader = build_GraphDRP_dataloader(params["val_ml_data_dir"],
                                           val_data_fname,
                                           params["val_batch"],
                                           shuffle=False)

    # -----------------------------
    # Determine CUDA/CPU device and configure CUDA device if available
    # -----------------------------
    # import pdb; pdb.set_trace()
    device = determine_device(params["cuda_name"])

    # -------------------------------------
    # Prepare GraphDRP (PyTorch) model
    # -------------------------------------
    # Model, Loss, Optimizer
    model_arch = params["model_arch"]
    model = str2Class(model_arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_fn = torch.nn.MSELoss()

    # -----------------------------
    # Train. Prep settings.
    # -----------------------------
    # [Req] Set checkpointing
    # import pdb; pdb.set_trace()
    print(f"model_outdir {params['model_outdir']}")
    print(f"ckpt_directory: {params['ckpt_directory']}")
    # TODO: why nested dirs are created: params["ckpt_directory"]/params["ckpt_directory"]
    # params["output_dir"] = params["model_outdir"]
    if params["ckpt_directory"] is None:
        params["ckpt_directory"] = params["model_outdir"]
        # params["ckpt_directory"] = "ckpt_graphdrp"  # TODO: why nested dirs are created: params["ckpt_directory"]/params["ckpt_directory"]
    # initial_epoch = trobj.config_checkpointing(params["ckpt_directory"])
    ckpt_obj, initial_epoch = config_checkpointing(params, model, optimizer)

    num_epoch = params["epochs"]
    log_interval = params["log_interval"]
    patience = params["patience"]

    # Settings for early stop and best model settings
    best_score = np.inf
    best_epoch = -1
    early_stop_counter = 0  # define early-stop counter
    early_stop_metric = "mse"  # metric for early stop

    # -----------------------------
    # Train. Iterate over epochs.
    # -----------------------------
    # import pdb; pdb.set_trace()
    # this is from check_data_available()

    metrics = ["mse", "rmse", "pcc", "scc", "r2"]

    print_fn(f"Epochs: {initial_epoch} to {num_epoch}")
    for epoch in range(initial_epoch, num_epoch):
        # import ipdb; ipdb.set_trace()
        # Train epoch and ckechpoint model
        train_loss = train_epoch(model, device, train_loader, optimizer, loss_fn, epoch + 1, log_interval)
        ckpt_obj.ckpt_epoch(epoch, train_loss) # checkpoints the best model by default

        # Predict with val data
        val_true, val_pred = predicting(model, device, val_loader)  # val_true (groud truth), val_pred (predictions)
        val_scores = compute_metrics(val_true, val_pred, metrics)

        # For early stop
        print_fn(f"{early_stop_metric}, {val_scores[early_stop_metric]}")
        if val_scores[early_stop_metric] < best_score:
            torch.save(model.state_dict(), modelpath)
            best_epoch = epoch + 1
            best_score = val_scores[early_stop_metric]
            print_fn(f"{early_stop_metric} improved at epoch {best_epoch};  \
                     Best {early_stop_metric}: {best_score};  Model: {model_arch}")
            early_stop_counter = 0  # zero the early-stop counter if the model improved after the epoch
        else:
            print_fn(f"No improvement since epoch {best_epoch};  \
                     Best {early_stop_metric}: {best_score};  Model: {model_arch}")
            early_stop_counter += 1  # increment the counter if the model was not improved after the epoch

        if early_stop_counter == patience:
            print_fn(f"Terminate training (model did not improve on val data for {params['patience']} epochs).")
            print_fn(f"Best epoch: {best_epoch};  Best score ({early_stop_metric}): {best_score}")
            break

    # -------------------------------------
    # Create Trainer object and setup train
    # -------------------------------------

    # # import pdb; pdb.set_trace()
    # # this is from check_data_available()
    # opath = Path(params["model_outdir"])
    # os.makedirs(opath, exist_ok=True)
    # modelpath = opath / params["model_params"]

    # # trobj = Trainer(params=params, device=device, modelpath=outdtd["model"])
    # trobj = Trainer(params=params, device=device, modelpath=modelpath)
    # trobj.setup_train()


    # # -----------------------------
    # # [Req] Set checkpointing
    # # -----------------------------
    # if params["ckpt_directory"] is None:
    #     params["ckpt_directory"] = params["model_outdir"]
    # import ipdb; ipdb.set_trace()
    # initial_epoch = trobj.config_checkpointing(params["ckpt_directory"])

    # -----------------------------
    # Prepare PyTorch dataloaders
    # train_loader = build_PT_data_loader(params["train_ml_data_dir"],
    #                                     params["train_data_processed"],
    #                                     params["batch_size"],
    #                                     shuffle=True)

    # # Note! Don't shuffle the val_loader or results will be corrupted
    # val_loader = build_PT_data_loader(params["val_ml_data_dir"],
    #                                   params["val_data_processed"],
    #                                   params["val_batch"],
    #                                   shuffle=False)

    # # -----------------------------
    # # Train
    # # -----------------------------
    # import ipdb; ipdb.set_trace()
    # trobj.execute_train(train_loader, val_loader, initial_epoch)

    # -----------------------------
    # [Req] Load best model and cal preds
    # -----------------------------
    # Load the (best) saved model (as determined based on val data)
    # Compute predictions
    # (groud truth), (predictions)
    # val_true, val_pred = evaluate_model(params["model_arch"], device, outdtd["model"], val_loader)
    # import ipdb; ipdb.set_trace()
    # TODO. consider separate evaluate_model() into:
    # 1) load_model() or load_graphdrp()
    # 2) model_predict()
    # model = load_model(params)
    # val_true, val_pred = predict(model, device, data_loader=val_loader)
    val_true, val_pred = evaluate_model(params, device, modelpath, val_loader)

    # -----------------------------
    # [Req] Save raw preds in df
    # -----------------------------
    # Store predictions in data frame
    # import ipdb; ipdb.set_trace()
    # Attempt to concat predictions with the cancer and drug ids, and the true values
    # If data frame found, then y_true is read from data frame and returned
    # Otherwise, only a partial data frame is stored (with val_true and val_pred)
    # and y_true is equal to pytorch loaded val_true
    # y_true = store_predictions_df(params, indtd, outdtd, val_true, val_pred)
    # y_true = store_preds_df(params, val_true, val_pred, stage="val")
    y_true = store_preds_df(params, y_true=val_true, y_pred=val_pred, stage="val")

    # -----------------------------
    # Compute performance scores
    # -----------------------------
    metrics = ["mse", "rmse", "pcc", "scc", "r2"]
    val_scores = compute_performace_scores(y_true, val_pred, metrics, outdtd, "val")

    return val_scores


def main():
    # additional_definitions = gdrp_model_conf + gdrp_data_conf + gdrp_train_conf
    additional_definitions = model_train_params + \
                             model_preproc_params + \
                             drp_train_params
    params = frm.initialize_parameters(
        filepath,
        default_model="graphdrp_default_model.txt",
        additional_definitions=additional_definitions,
        required=req_train_args,
    )
    val_scores = run(params)
    print_fn("\nFinished training GraphDRP model.")


if __name__ == "__main__":
    main()
