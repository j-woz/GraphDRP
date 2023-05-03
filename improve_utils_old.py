import os
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
import matplotlib.pyplot as plt


# These are globals for all models
# TODO:
# Where do these go?
# Solution:
# ... 
import types
imp_globals = types.SimpleNamespace()
imp_globals.raw_data_dir_name = "raw_data"
imp_globals.ml_data_dir_name = "ml_data"
imp_globals.models_dir_name = "models"
imp_globals.infer_dir_name = "infer"
# ---
imp_globals.x_data_dir_name = "x_data"
imp_globals.y_data_dir_name = "y_data"
imp_globals.canc_col_name = "CancID"
imp_globals.drug_col_name = "DrugID"
imp_globals.ge_fname = "ge.parquet"  # cancer feature
imp_globals.smiles_fname = "smiles.csv"  # drug feature
imp_globals.y_file_substr = "rsp"


def load_rsp_data(src_raw_data_dir: str, y_col_name: str="AUC", verbose: bool=True):
    """
    IMPROVE-specific func.
    Read drug response response file and return a datarame with cancer ids,
    drug ids, and drug response values.
    src_raw_data_dir : data dir where the raw DRP data is stored
    y_col_name : Drug sensitivity score/metric (e.g., AUC, IC50)
    """
    # pathlist = list(Path(src_raw_data_dir/y_datadir_name).glob(f"{y_file_substr}*.csv"))  # glob csv files that contain response data
    pathlist = list(Path(src_raw_data_dir/imp_globals.y_data_dir_name).glob(f"{imp_globals.y_file_substr}*.csv"))  # glob csv files that contain response data
    # pathlist = [p for p in pathlist if "full" not in str(p)]  # get the file that contains the full dataset
    pathlist = [p for p in pathlist if "full" in str(p)]  # get the file that contains the full dataset
    rsp_df = pd.read_csv(pathlist[0])  # there should be only one suitable file
    rsp_df = rsp_df[[imp_globals.drug_col_name, imp_globals.canc_col_name, y_col_name]]  # [drug id, cancer id, response]
    # print(rsp_df[[canc_col_name, drug_col_name]].nunique())
    if verbose:
        print(rsp_df[[imp_globals.canc_col_name, imp_globals.drug_col_name]].nunique())
    return rsp_df


def load_ge_data(src_raw_data_dir: str):
    """
    IMPROVE-specific func.
    Read gene expressions file.
    src_raw_data_dir : data dir where the raw DRP data is stored
    """
    ge = read_df(src_raw_data_dir/imp_globals.x_data_dir_name/imp_globals.ge_fname)
    return ge


def load_cnv_data(src_raw_data_dir: str, y_col_name: str="AUC"):
    """
    IMPROVE-specific func.
    Read drug response response file and return a datarame with cancer ids,
    drug ids, and drug response values.
    src_raw_data_dir : data dir where the raw DRP data is stored
    y_col_name : Drug sensitivity score/metric (e.g., AUC, IC50)
    """
    pass
    return None


def get_common_samples(df1: pd.DataFrame, df2: pd.DataFrame, ref_col: str):
    """
    IMPROVE-specific func.
    df1, df2 : dataframes
    ref_col : the ref column to find the common values

    Returns:
        df1, df2

    Example:
        TODO
    """
    # Retain (canc, drug) response samples for which we have omic data
    # TODO: consider making this an IMPROVE func
    common_ids = list(set(df1[ref_col]).intersection(df2[ref_col]))
    # print(df1.shape)
    df1 = df1[ df1[imp_globals.canc_col_name].isin(common_ids) ]
    # print(df1.shape)
    # print(df2.shape)
    df2 = df2[ df2[imp_globals.canc_col_name].isin(common_ids) ]
    # print(df2.shape)
    return df1, df2


def load_smiles_data(src_raw_data_dir: str):
    """
    IMPROVE-specific func.
    Read smiles data.
    src_raw_data_dir : data dir where the raw DRP data is stored
    """
    smi = read_df(src_raw_data_dir/imp_globals.x_data_dir_name/imp_globals.smiles_fname)
    return smi


def get_data_splits(src_raw_data_dir: str, splitdir_name: str,
                    split_file_name: str, rsp_df: pd.DataFrame):
    """
    IMPROVE-specific func.
    Read smiles data.
    src_raw_data_dir : data dir where the raw DRP data is stored
    """
    splitdir = src_raw_data_dir/splitdir_name
    if len(split_file_name) == 1 and split_file_name[0] == "full":
        # Full dataset (take all samples)
        ids = list(range(rsp_df.shape[0]))
    else:
        # Check if the split file exists and load
        ids = []
        for fname in split_file_name:
            assert (splitdir/fname).exists(), "split_file_name not found."
            with open(splitdir/fname) as f:
                ids_ = [int(line.rstrip()) for line in f]
                ids.extend(ids_)

    """
    # Method 1
    splitdir = Path(os.path.join(src_raw_data_dir))/"splits"
    if len(args.split_file_name) == 1 and args.split_file_name[0] == "full":
        # Full dataset (take all samples)
        ids = list(range(rsp_df.shape[0]))
        outdir_name = "full"
    else:
        # Check if the split file exists and load
        ids = []
        split_id_str = []    # e.g. split_5
        split_type_str = []  # e.g. tr, vl, te
        for fname in args.split_file_name:
            assert (splitdir/fname).exists(), "split_file_name not found."
            with open(splitdir/fname) as f:
                # Get the ids
                ids_ = [int(line.rstrip()) for line in f]
                ids.extend(ids_)
                # Get the name
                fname_sep = fname.split("_")
                split_id_str.append("_".join([s for s in fname_sep[:2]]))
                split_type_str.append(fname_sep[2])
        assert len(set(split_id_str)) == 1, "Data splits must be from the same dataset source."
        split_id_str = list(set(split_id_str))[0]
        split_type_str = "_".join([x for x in split_type_str])
        outdir_name = f"{split_id_str}_{split_type_str}"
    ML_DATADIR = main_data_dir/"ml_data"
    root = ML_DATADIR/f"data.{args.source_data_name}"/outdir_name # ML data
    os.makedirs(root, exist_ok=True)
    """

    """
    # Method 2
    splitdir = src_raw_data_dir/args.splitdir_name
    if len(args.split_file_name) == 1 and args.split_file_name[0] == "full":
        # Full dataset (take all samples)
        ids = list(range(rsp_df.shape[0]))
    else:
        # Check if the split file exists and load
        ids = []
        for fname in args.split_file_name:
            assert (splitdir/fname).exists(), "split_file_name not found."
            with open(splitdir/fname) as f:
                ids_ = [int(line.rstrip()) for line in f]
                ids.extend(ids_)
    """
    return ids


def get_subset_df(df: pd.DataFrame, ids: list):
    """ Get a subset of the input dataframe based on row ids."""
    df = df.loc[ids]
    return df








def read_df(fpath: str, sep: str=","):
    """
    IMPROVE-specific func.
    Load a dataframe. Supports csv and parquet files.
    sep : the sepator in the csv file
    """
    # TODO: this func might be available in candle
    assert Path(fpath).exists(), f"File {fpath} was not found."
    if "parquet" in str(fpath):
        df = pd.read_parquet(fpath)
    else:
        df = pd.read_csv(fpath, sep=sep)
    return df


def rmse(y, f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse


def mse(y, f):
    mse = ((y - f)**2).mean(axis=0)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs
