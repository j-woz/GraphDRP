import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import json
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr, sem

from improve.metrics import compute_metrics

fdir = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', required=False, default='LGBM', type=str,
                    help='Name of the model.')
parser.add_argument('--y_col_name', required=False, default='auc', type=str,
                    help='Y col name.')
args = parser.parse_args()

y_col_name = args.y_col_name

main_dir_path = Path("/lambda_stor/data/apartin/projects/IMPROVE/pan-models/GraphDRP")
infer_dir_name = "infer"
infer_dir_path = main_dir_path/y_col_name/infer_dir_name
dirs = list(infer_dir_path.glob("*-*")); print(dirs)
# print(split_files)

# model_name = args.model_name
# model_name = "GraphDRP_01"
# model_name = "GraphDRP_02"
# model_name = "GraphDRP_03"

# datadir = fdir/f"results.{model_name}"
# outdir = fdir/f"scores.{model_name}"
# os.makedirs(outdir, exist_ok=True)


data_sources = ["ccle", "ctrp", "gcsi", "gdsc1", "gdsc2"]
trg_name = "AUC"
round_digits = 3

def calc_mae(y_true, y_pred):
    return sklearn.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)

def calc_r2(y_true, y_pred):
    return sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred)

def calc_pcc(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

def calc_scc(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]

scores_names = {"mae": calc_mae,
                "r2": calc_r2,
                "pcc": calc_pcc,
                "scc": calc_scc}

# ====================
# Aggregate raw scores
# ====================

scores_file_name = "test_scores.json"
preds_file_name = "test_y_data_predicted.csv"
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]  

dct = {}

for dir_path in dirs:
    print("\nExperiment:", dir_path)
    # import ipdb; ipdb. set_trace()
    src = str(dir_path.name).split("-")[0]
    trg = str(dir_path.name).split("-")[1]
    split_dirs = list((dir_path).glob(f"split_*"))

    import ipdb; ipdb. set_trace()
    for split_dir in split_dirs:
        # # Load scores
        # scores_file_path = split_dir/scores_file_name
        # with open(scores_file_path) as json_file:
        #     scores = json.load(json_file)

        split = int(split_dir.name.split("split_")[1])

        # Load predictions
        preds_file_path = split_dir/preds_file_name
        df = pd.read_csv(preds_file_path, sep=",")
        # df = pd.read_csv(resdir/fname, sep="\t")

        # for sc_name, sc_func in scores_names.items():
        y_true = df[f"{y_col_name}_true"].values
        y_pred = df[f"{y_col_name}_pred"].values
        # ---
        scores = compute_metrics(y_true, y_pred, metrics_list)

        dct[split] = scores
        df = pd.DataFrame(dct)
        df = df.T.reset_index().rename(columns={"index": "split"})

        # TODO. continue

        # sc_value = sc_func(y_true=y_true, y_pred=y_pred)
        # # scores[trg][sc_name].append(sc_value)
        # import ipdb; ipdb. set_trace()
        # scores[trg].append(sc_value)
        
    import ipdb; ipdb. set_trace()
    kk = pd.DataFrame(dct)

# ---------------------
# Data source study
for sc_name, sc_func in scores_names.items():
    print("\nMetric:", sc_name)
    for src in data_sources:
        print("\n\tSource study:", src)
        # resdir = fdir/f"results.csa.{src}"
        # resdir = datadir/f"results.csa.{src}"
        resdir = datadir
        scores = {}

        # Data test study
        for trg in data_sources:
            print("\tTraget study:", trg)
            if trg not in scores:
                # scores[trg] = {sc: [] for sc in scores_names}
                scores[trg] = []

            # Data split
            for split in range(10):
                fname = f"{src}_{trg}_split_{split}.csv"
                df = pd.read_csv(resdir/fname, sep=",")
                # df = pd.read_csv(resdir/fname, sep="\t")

                # for sc_name, sc_func in scores_names.items():
                y_true = df["True"].values
                y_pred = df["Pred"].values
                sc_value = sc_func(y_true=y_true, y_pred=y_pred)
                # scores[trg][sc_name].append(sc_value)
                scores[trg].append(sc_value)

        with open(outdir/f"{sc_name}_{src}_scores_raw.json", "w") as json_file:
            json.dump(scores, json_file)
del scores
# ---------------------



# ====================
# Generate tables
# ====================
# import ipdb; ipdb.set_trace(context=5)
# Data source study
for sc_name in scores_names.keys():
    print("\nMetric:", sc_name)

    mean_df = {}
    err_df = {}
    for src in data_sources:
        print("\tSource study:", src)

        with open(outdir/f"{sc_name}_{src}_scores_raw.json") as json_file:
            mean_scores = json.load(json_file)
        err_scores = mean_scores.copy()

        # print(scores)

        for trg in data_sources:
            mean_scores[trg] = round(np.mean(mean_scores[trg]), round_digits)
            err_scores[trg] = round(sem(err_scores[trg]), round_digits)

        # import ipdb; ipdb.set_trace(context=5)
        mean_df[src] = mean_scores
        err_df[src] = err_scores

    # import ipdb; ipdb.set_trace(context=5)
    mean_df = pd.DataFrame(mean_df)
    err_df = pd.DataFrame(err_df)
    mean_df.to_csv(outdir/f"{sc_name}_mean_table.csv", index=True)
    err_df.to_csv(outdir/f"{sc_name}_err_table.csv", index=True)


# ====================
# Summary table
# ====================
# import ipdb; ipdb.set_trace(context=5)
# Data source study
sc_df = []
cols = ["Metric", "Diag", "Off-diag", "Diag_std", "Off-diag_std"]
for i, sc_name in enumerate(scores_names):
    # print("\nMetric:", sc_name)
    sc_item = {}
    sc_item["Metric"] = sc_name.upper()

    mean_df = pd.read_csv(outdir/f"{sc_name}_mean_table.csv")
    n = mean_df.shape[0]

    # Diag
    vv = np.diag(mean_df.iloc[:, 1:].values)
    sc_item["Diag"] = np.round(sum(vv)/n, 3)
    sc_item["Diag_std"] = np.round(np.std(vv), 3)

    # Off-diag
    vv = mean_df.iloc[:, 1:].values
    np.fill_diagonal(vv, 0)
    sc_item["Off-diag"] = np.round(sum(np.ravel(vv)) / (n*n - n), 3)
    sc_item["Off-diag_std"] = np.round(np.std(np.ravel(vv)), 3)

    for ii, dname in enumerate(mean_df.iloc[:, 0].values):
        dname = dname.upper()
        sc_item[dname] = np.round(sum(vv[ii, :] / (n - 1)), 3)
        if i == 0:
            cols.append(dname)

    sc_df.append(sc_item)

sc_df = pd.DataFrame(sc_df, columns=cols)
sc_df.to_csv(outdir/f"summary_table.csv", index=False)
print(sc_df)

print("Finished all")
