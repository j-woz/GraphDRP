import argparse
import datetime
import os
import sys
from random import shuffle
from time import time

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


class Timer:
    """
    Measure runtime.
    """

    def __init__(self):
        self.start = time()

    def timer_end(self):
        self.end = time()
        time_diff = self.end - self.start
        return time_diff

    def display_timer(self, print_fn=print):
        time_diff = self.timer_end()
        if (time_diff) // 3600 > 0:
            print_fn("Runtime: {:.1f} hrs".format((time_diff) / 3600))
        else:
            print_fn("Runtime: {:.1f} mins".format((time_diff) / 60))


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, log_interval):
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

    # ap
    timer = Timer()
    if args.set == "mix":
        set_str = "_mix"
        val_scheme = "mixed"
    elif args.set == "cell":
        set_str = "_cell_blind"
        val_scheme = "cell_blind"
    elif args.set == "drug":
        set_str = "_blind"
        val_scheme = "drug_blind"

    # ap
    from pathlib import Path

    fdir = Path(__file__).resolve().parent
    if args.gout is not None:
        outdir = fdir / args.gout
    else:
        outdir = fdir / "results"
    os.makedirs(outdir, exist_ok=True)

    print("Learning rate: ", args.learning_rate)
    print("Epochs: ", args.epochs)

    model_st = modeling.__name__
    dataset = "GDSC"
    train_losses = []
    val_losses = []
    val_pearsons = []
    print("\nrunning on ", model_st + "_" + dataset)

    root = args.root
    if args.tr_file is None:
        processed_data_file_train = (
            root + "/processed/" + dataset + "_train" + set_str + ".pt"
        )
    else:
        processed_data_file_train = root + "/processed/" + args.tr_file + ".pt"

    if args.vl_file is None:
        processed_data_file_val = (
            root + "/processed/" + dataset + "_val" + set_str + ".pt"
        )
    else:
        processed_data_file_val = root + "/processed/" + args.vl_file + ".pt"

    if args.tr_file is None:
        processed_data_file_test = (
            root + "/processed/" + dataset + "_test" + set_str + ".pt"
        )
    else:
        processed_data_file_test = root + "/processed/" + args.te_file + ".pt"

    # import pdb; pdb.set_trace()
    if (
        (not os.path.isfile(processed_data_file_train))
        or (not os.path.isfile(processed_data_file_val))
        or (not os.path.isfile(processed_data_file_test))
    ):
        print("please run create_data.py to prepare data in pytorch format!")
    else:
        train_data = TestbedDataset(root=root, dataset=args.tr_file)
        val_data = TestbedDataset(root=root, dataset=args.vl_file)
        test_data = TestbedDataset(root=root, dataset=args.te_file)

        # currently all set to batch_size, may change.
        train_batch = args.batch_size
        val_batch = args.batch_size
        test_batch = args.batch_size

        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
        print("CPU/GPU: ", torch.cuda.is_available())

        # training the model
        device = torch.device(args.cuda_name if torch.cuda.is_available() else "cpu")
        print("Device: ", device)
        model = modeling().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        best_mse = 1000
        best_pearson = 1
        best_epoch = -1

        model_file_name = outdir / (
            "model_" + model_st + "_" + dataset + "_" + val_scheme + ".model"
        )
        result_file_name = outdir / (
            "result_" + model_st + "_" + dataset + "_" + val_scheme + ".csv"
        )
        loss_fig_name = str(
            outdir / ("model_" + model_st + "_" + dataset + "_" + val_scheme + "_loss")
        )
        pearson_fig_name = str(
            outdir
            / ("model_" + model_st + "_" + dataset + "_" + val_scheme + "_pearson")
        )
        for epoch in range(args.epochs):
            train_loss = train(
                model, device, train_loader, optimizer, epoch + 1, args.log_interval
            )
            G, P = predicting(model, device, val_loader)
            ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P)]

            G_test, P_test = predicting(model, device, test_loader)
            ret_test = [
                rmse(G_test, P_test),
                mse(G_test, P_test),
                pearson(G_test, P_test),
                spearman(G_test, P_test),
            ]

            train_losses.append(train_loss)
            val_losses.append(ret[1])
            val_pearsons.append(ret[2])

            if ret[1] < best_mse:  # ap: is it early stopping on the mse of train set??
                torch.save(model.state_dict(), model_file_name)
                with open(result_file_name, "w") as f:
                    f.write(",".join(map(str, ret_test)))
                best_epoch = epoch + 1
                best_mse = ret[1]
                best_pearson = ret[2]
                print(
                    " rmse improved at epoch ",
                    best_epoch,
                    "; best_mse:",
                    best_mse,
                    model_st,
                    dataset,
                )
            else:
                print(
                    " no improvement since epoch ",
                    best_epoch,
                    "; best_mse, best pearson:",
                    best_mse,
                    best_pearson,
                    model_st,
                    dataset,
                )
        draw_loss(train_losses, val_losses, loss_fig_name)
        draw_pearson(val_pearsons, pearson_fig_name)

        # ap: Add code to create dir for results

        # ap: Add to drop raw predictions
        G_test, P_test = predicting(model, device, test_loader)
        preds = pd.DataFrame({"True": G_test, "Pred": P_test})
        preds_file_name = f"preds_{val_scheme}_{model_st}_{dataset}.csv"
        preds.to_csv(outdir / preds_file_name, index=False)

        # ap: Add code to calc and dump scores
        # ret = [rmse(G_test, P_test), mse(G_test, P_test), pearson(G_test, P_test), spearman(G_test, P_test)]
        ccp_scr = pearson(G_test, P_test)
        rmse_scr = rmse(G_test, P_test)
        scores = {"ccp": ccp_scr, "rmse": rmse_scr}
        import json

        with open(
            outdir / f"scores_{val_scheme}_{model_st}_{dataset}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)

        timer.display_timer()
        print(scores)
        print("Done.")
        return scores


def run(gParameters):
    print("In Run Function:\n")
    args = candle.ArgumentStruct(**gParameters)
    modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][args.modeling]

    # call the launch function with specific model and args with all hyperparameters
    scores = launch(modeling, args)
    return scores

def initialize_parameters():
    print("Initializing parameters\n")

    """Initialize the parameters for the GraphDRP benchmark"""
    graphdrp_bench = bmk.BenchmarkGraphDRP(
        bmk.file_path,
        "graphdrp_default_model.txt",
        "pytorch",
        prog="graphdrp",
        desc="Graph DRP",
    )

    gParameters = candle.finalize_parameters(graphdrp_bench)
    return gParameters


def main():
    gParameters = initialize_parameters()
    print(gParameters)
    run(gParameters)


if __name__ == "__main__":
    main()
