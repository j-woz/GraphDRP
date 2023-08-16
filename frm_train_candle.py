import os
import json
import sys

import numpy as np
import pandas as pd

from pprint import pprint

import torch
from pathlib import Path

# from utils import TestbedDataset, DataLoader
from utils import TestbedDataset
from torch_geometric.data import DataLoader
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet

import frm
import candle_improve_utils as improve_utils
from candle import build_pytorch_optimizer, get_pytorch_function, keras_default_config

# Params that are required to be specified in order to execute the train script (no default values)
# TODO: consider setting req_train_args elsewhere and then loading it into this script. Similarly we can specify
# req_preprocess_args and req_infer_args
# req_train_args = ["train_ml_data_dir", "val_ml_data_dir", "y_col_name"]  # there is default for y_col_name (auc)
req_train_args = ["train_ml_data_dir", "val_ml_data_dir"]
frm.required.extend(req_train_args)


def str2Class(str):
    #return getattr(sys.modules[__name__], str)
    return globals()[str]()


def train(model, device, train_loader, optimizer, loss_fn, epoch, log_interval):
    """ Training of one epoch (all batches). """
    print("Training on {} samples...".format(len(train_loader.dataset)))
    model.train()
    # loss_fn = nn.MSELoss()
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


def run(params):
    """ Execute specified model training.

    Parameters
    ----------
    params: python dictionary
        A dictionary of Candle keywords and parsed values.
    """
    # -----------------------------
    # Train parameters
    lr = params["learning_rate"]
    num_epoch = params["epochs"]
    train_batch = params["batch_size"]
    log_interval = params["log_interval"]
    val_batch = params["val_batch"]

    # -----------------------------
    # [Req]
    # Prepare model output
    model_outdir = Path(params["ckpt_directory"])
    # model_outdir = Path(params["model_outdir"])
    os.makedirs(model_outdir, exist_ok=True)
    model_path = model_outdir / params["model_params"] # TODO: consider changing "model_params" to more descriptive, e.g., model_file_name

    # -----------------------------
    # Prepare PyG datasets
    train_data_file_name = "train_data"  # TestbedDataset() appends this string with ".pt"
    val_data_file_name = "val_data"
    train_data = TestbedDataset(root=params["train_ml_data_dir"], dataset=train_data_file_name)
    val_data = TestbedDataset(root=params["val_ml_data_dir"], dataset=val_data_file_name)

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
    # [Req?]
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

        # Predict with val data
        val_true, val_pred = frm.predicting(model, device, val_loader)  # val_true (groud truth), val_pred (predictions)
        val_scores = improve_utils.compute_metrics(val_true, val_pred, metrics)

        # Checkpoint the (best) model
        # TODO (todo-ap):
        # Can we use CANDLE checkpoint for this?
        if val_scores[early_stop_metric] < best_score:
            torch.save(model.state_dict(), model_path)
            # with open(result_file_name, "w") as f:
            #     f.write(",".join(map(str, ret)))
            best_epoch = epoch + 1
            best_score = val_scores[early_stop_metric]
            # rmse_for_best_mse = val_scores["rmse"]
            # pearson_for_best_mse = val_scores["pcc"]
            # spearman_for_best_mse = val_scores["scc"]
            print(f"{early_stop_metric} improved at epoch {best_epoch}; Best \
                  {early_stop_metric}: {best_score}; Model: {params['model_arch']}")
            early_stop_counter = 0  # zero the early-stop counter if the model imroved after the epoch
        else:
            print(f"No improvement since epoch {best_epoch}; Best \
                  {early_stop_metric}: {best_score}; Model: {params['model_arch']}")
            early_stop_counter += 1  # increment the counter if the model was not improved after the epoch

        if early_stop_counter == params["patience"]:
            print(f"Terminate training (model did not improve on val data for {params['patience']} epochs).")
            continue

    # -----------------------------
    # Load the (best) saved model (as determined based val data)
    del model
    model = str2Class(params["model_arch"]).to(device)
    model.load_state_dict(torch.load(model_path))
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
    # TODO (todo-ap): consider to change "val_response_data" to "val_y_data"
    rsp_df = pd.read_csv(Path(params["val_ml_data_dir"] + params["val_response_data"]))

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
    print(params["val_pred_file_path"])   # used in frm_train*.py
    print(params["test_pred_file_path"])  # used in frm_train*.py
    # TODO (todo-ap): params "val_pred_file_path" and "test_pred_file_path" are not properly defined.
    params["val_pred_file_path"] = params["model_outdir"] + params["val_pred_fname"]
    print(params["val_pred_file_path"])
    improve_utils.save_preds(mm, params, params["val_pred_file_path"])

    # -----------------------------
    # Compute performance scores
    # -----------------------------
    metrics = ["mse", "rmse", "pcc", "scc", "r2"]
    y_true = rsp_df[params["y_col_name"]].values
    val_scores = improve_utils.compute_metrics(y_true, val_pred, metrics)
    val_scores["val_loss"] = val_scores["mse"]

    # Performance scores for Supervisor HPO
    print("\nIMPROVE_RESULT val_loss:\t{}\n".format(val_scores["mse"]))
    with open(model_outdir / params["json_val_scores"], "w", encoding="utf-8") as f:
        json.dump(val_scores, f, ensure_ascii=False, indent=4)

    print("Validation scores:\n\t{}".format(val_scores))
    return val_scores


def main():
    params = frm.initialize_parameters(default_model="frm_default_model.txt")
    # Add infer parameter
    # params["out_file_path"] = params["output_dir"] + params["val_pred_fname"]  # TODO (Q-ap): why define here and not inside run()??
    pprint(params)
    run(params)
    print("\nFinished training.")


if __name__ == "__main__":
    main()
