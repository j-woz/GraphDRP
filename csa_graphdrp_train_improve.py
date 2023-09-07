"""Functionality for Training in Cross-Study Analysis (CSA) for GraphDRP Model."""

from pathlib import Path
import os


import json
import sys

import numpy as np
import pandas as pd


import torch
from torch_geometric.data import DataLoader

from improve import csa
from candle import build_pytorch_optimizer, get_pytorch_function, keras_default_config, CandleCkptPyTorch

from improve.torch_utils import TestbedDataset
from improve.metrics import compute_metrics
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet

from graphdrp_train_improve import str2Class

filepath = Path(__file__).resolve().parent



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


def run(params):
    """Execute graphDRP CSA model training.

    Parameters
    ----------
    params: python dictionary
        A dictionary of Candle/Improve keywords and parsed values.
    """
    # ------------------------------------------
    # Check/Construct output directory structure
    # ------------------------------------------
    totrainq, _ = csa.directory_tree_from_parameters(params, step = "train")
    # Each element of the queue contains a tuple ((source, target, split_id), ipath, opath)
    print(totrainq)

    # -----------------------------
    # Train ML data for every split
    # -----------------------------
    while totrainq:
        elem = totrainq.popleft() # This is (DataSplit, InputSplitPath, OutputSplitPath)
        # -----------------------------------------------
        # Prepare checkpoint path to follow CSA structure
        ckptpath = elem[2] / "ckpts"
        os.makedirs(ckptpath, exist_ok=True)
        params["ckpt_directory"] = ckptpath # Needed to build checkpointing
        # model_path = elem[2] / params["model_params"]
        # -----------------------------------------------
        # Recover datasets from CSA structure
        xpath = elem[1] / "processed"
        if xpath.exists() == False:
            raise Exception(f"ERROR ! Feature data path '{xpath}' not found.\n")
        train_data_file_name = f"train_{params['x_data_suffix']}" # TestbedDataset() appends this string with ".pt"
        val_data_file_name = f"val_{params['x_data_suffix']}"

########## Use graphdrp_train_improve......


def main():
    params = csa.frm.initialize_parameters(filepath,
                                       default_model="csa_graphdrp_default_model.txt",
                                       additional_definitions = csa.csa_conf,
                                       required = csa.req_csa_args,
                                      )

    params = csa.initialize_parameters(filepath,
                                       default_model="csa_graphdrp_default_model.txt",
                                       additional_definitions = gdrp_data_conf,
                                       required = required_csa,
                                       topop = not_used_from_model,
                                      )

    run(params)
    print("\nFinished CSA GraphDRP training.")


if __name__ == "__main__":
    main()
