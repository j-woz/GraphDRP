import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *
import datetime
import argparse

import sklearn
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

data_sources = ["ccle", "ctrp", "gcsi", "gdsc1", "gdsc2"]


from time import time
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
        if (time_diff)//3600 > 0:
            print_fn("Runtime: {:.1f} hrs".format( (time_diff)/3600) )
        else:
            print_fn("Runtime: {:.1f} mins".format( (time_diff)/60) )


def calc_mae(y_true, y_pred):
    return sklearn.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)

def calc_r2(y_true, y_pred):
    return sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred)

def calc_pcc(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

def calc_scc(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, log_interval):
    # print('Training on {} samples...'.format(len(train_loader.dataset)))
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
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data.x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return sum(avg_loss) / len(avg_loss)


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    # print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _ = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def launch(modeling, train_batch, val_batch, test_batch, lr, num_epoch, log_interval,
         cuda_name, args):
    # import pdb; pdb.set_trace()

    timer = Timer()

    # ap
    from pathlib import Path
    fdir = Path(__file__).resolve().parent
    # if args.gout is not None:
    #     outdir = fdir/args.gout
    # else:
    #     outdir = fdir/f"results.csg.{args.src}.split_{args.split}"
    outdir = fdir/f"results.csg.{args.src}"
    os.makedirs(outdir, exist_ok=True)
    # outdir = outdir/f"split_{args.split}"
    # os.makedirs(outdir, exist_ok=True)

    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)

    model_st = modeling.__name__
    dataset = 'GDSC'
    train_losses = []
    val_losses = []
    val_pearsons = []
    print('\nrunning on ', model_st + '_' + dataset)

    # import pdb; pdb.set_trace()
    # root = args.root
    datadir = fdir/args.datadir/f"data.{args.src}"
    root = str(datadir/f"data_split_{args.split}")

    processed_data_file_train = root + '/processed/' + args.train_data + '.pt'
    processed_data_file_val = root + '/processed/' + args.val_data + '.pt'
    processed_data_file_test = root + '/processed/' + args.test_data + '.pt'

    # import pdb; pdb.set_trace()
    if ((not os.path.isfile(processed_data_file_train))
            or (not os.path.isfile(processed_data_file_val))
            or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        # import pdb; pdb.set_trace()
        train_data = TestbedDataset(root=root, dataset=args.train_data)
        val_data = TestbedDataset(root=root, dataset=args.val_data)
        test_data = TestbedDataset(root=root, dataset=args.test_data)

        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
        val_loader   = DataLoader(val_data, batch_size=val_batch, shuffle=False)
        test_loader  = DataLoader(test_data, batch_size=test_batch, shuffle=False)
        print("CPU/GPU: ", torch.cuda.is_available())

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        print("Device", device)
        model = modeling().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_mse = 1000
        best_pearson = 1
        best_epoch = -1

        model_file_name = outdir/('model_' + model_st + f'_split_{args.split}' + '.model')

        for epoch in range(num_epoch):
            # import pdb; pdb.set_trace()
            train_loss = train(model, device, train_loader, optimizer, epoch + 1, log_interval)
            G, P = predicting(model, device, val_loader)
            ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P)]

            G_test, P_test = predicting(model, device, test_loader)
            ret_test = [
                rmse(G_test, P_test),
                mse(G_test, P_test),
                pearson(G_test, P_test),
                spearman(G_test, P_test)
            ]

            if ret[1] < best_mse:
                torch.save(model.state_dict(), model_file_name)
                # with open(result_file_name, 'w') as f:
                #     f.write(','.join(map(str, ret_test)))
                best_epoch = epoch + 1
                best_mse = ret[1]
                best_pearson = ret[2]
                print('\tRMSE improved at epoch: ', best_epoch, '; Best_mse:', round(best_mse, 7))
            else:
                print('\tNo improvement since epoch: ', best_epoch,
                      '; Best_mse:', round(best_mse, 7), ', Best pearson:', round(best_pearson, 7))

        # Drop raw predictions
        # import pdb; pdb.set_trace()
        G_test, P_test = predicting(model, device, test_loader)
        pred = pd.DataFrame({"True": G_test, "Pred": P_test})
        te_data = pd.read_csv(datadir/f'data_split_{args.split}'/'test_rsp.csv')
        pred = pd.concat([te_data, pred], axis=1)
        pred = pred.astype({"AUC": np.float32, "True": np.float32, "Pred": np.float32})
        assert sum(pred["True"] == pred["AUC"]) == pred.shape[0], "Columns 'AUC' and 'True' are the ground truth."
        pred_fname = f"{args.src}_{args.src}_split_{args.split}.csv"
        pred.to_csv(outdir/pred_fname, index=False)

        scores = {"scc": calc_scc(G_test, P_test), "pcc": calc_pcc(G_test, P_test), "r2": calc_r2(G_test, P_test)}
        print(scores)

        # Drop preds on target set
        trg_studies = [s for s in data_sources if s not in args.src]
        # trg_studies = ["ccle", "gcsi"]
        for trg_study in trg_studies:
            print(f"\nTarget study {trg_study}")
            trg_study_dir = fdir/f"data/data.{trg_study}"
            trg_data = TestbedDataset(root=trg_study_dir/f"data_split_{args.split}", dataset="all_data")
            trg_loader  = DataLoader(trg_data, batch_size=test_batch, shuffle=False)

            G_test, P_test = predicting(model, device, trg_loader)
            pred = pd.DataFrame({"True": G_test, "Pred": P_test})
            te_df = pd.read_csv(trg_study_dir/f'data_split_{args.split}'/'all_rsp.csv')
            pred = pd.concat([te_df, pred], axis=1)
            pred = pred.astype({"AUC": np.float32, "True": np.float32, "Pred": np.float32})
            assert sum(pred["True"] == pred["AUC"]) == pred.shape[0], "Columns 'AUC' and 'True' are the ground truth."
            pred_fname = f"{args.src}_{trg_study}_split_{args.split}.csv"
            pred.to_csv(outdir/pred_fname, index=False)

            scores = {"scc": calc_scc(G_test, P_test), "pcc": calc_pcc(G_test, P_test), "r2": calc_r2(G_test, P_test)}
            print(scores)

        # ccp_scr = pearson(G_test, P_test)
        # rmse_scr = rmse(G_test, P_test)
        # scores = {"ccp": ccp_scr, "rmse": rmse_scr}
        # import json
        # with open(outdir/f"scores_{model_st}_split_{split}.json", "w", encoding="utf-8") as f:
        #     json.dump(scores, f, ensure_ascii=False, indent=4)
        # print(scores)

        timer.display_timer()
        print("Done.")

# def run(gParameters):


def initialize_parameters():
    print("Initializing parameters\n")

    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument(
        '--model',
        type=int,
        required=False,
        default=0,
        help='0: GINConvNet, 1: GATNet, 2: GAT_GCN, 3: GCNNet')
    parser.add_argument(
        '--train_batch',
        type=int,
        required=False,
        default=1024,
        help='Batch size training set')
    parser.add_argument(
        '--val_batch',
        type=int,
        required=False,
        default=1024,
        help='Batch size validation set')
    parser.add_argument(
        '--test_batch',
        type=int,
        required=False,
        default=1024,
        help='Batch size test set')
    parser.add_argument(
        '--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument(
        '--num_epoch', type=int, required=False, default=300, help='Number of epoch')
    parser.add_argument(
        '--log_interval', type=int, required=False, default=20, help='Log interval')
    parser.add_argument(
        '--cuda_name', type=str, required=False, default="cuda:0", help='Cuda')
    parser.add_argument("--set", type=str, choices=["mix", "cell", "drug"], help="Validation scheme.")

    parser.add_argument('--datadir', required=False, default="data", type=str,
                        help='Relative path to the cross-study data files.')
    # parser.add_argument('--root', required=False, default="data", type=str,
    #                     help='Path to processed .pt files (default: data).')
    parser.add_argument('--gout', default=None, type=str,
                        help="Global outdir to dump all the resusts.")
    parser.add_argument('--train_data', required=False, default=None, type=str,
                        help='Train data path (default: None).')
    parser.add_argument('--val_data', required=False, default=None, type=str,
                        help='Val data path (default: None).')
    parser.add_argument('--test_data', required=False, default=None, type=str,
                        help='Test data path (default: None).')

    # Args for cross-study analysis
    parser.add_argument('--split', required=False, default=0, type=int,
                        help='Split id.')
    parser.add_argument('--src', required=True, default="gdsc1", choices=data_sources,
                        help='Source name.')
    args = parser.parse_args()

    modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][args.model]
    train_batch = args.train_batch
    val_batch = args.val_batch
    test_batch = args.test_batch
    lr = args.lr
    num_epoch = args.num_epoch
    log_interval = args.log_interval
    cuda_name = args.cuda_name
    print("In Run Function:\n")
    launch(modeling, train_batch, val_batch, test_batch, lr, num_epoch, log_interval,
         cuda_name, args)


def main():
    gParameters = initialize_parameters()
    # run(gParameters)


if __name__ == "__main__":
    main()
