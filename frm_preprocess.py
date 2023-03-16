from collections import OrderedDict
from pathlib import Path
from pubchempy import *
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from utils import *
import argparse
import csv
import h5py
import json, pickle
import math
import matplotlib.pyplot as plt
import networkx as nx
import numbers
import numpy as np
import os
import pandas as pd
import pickle
import random
import sys

import candle


"""
Functions below are used to generate graph molecular structures.
"""

fdir = os.path.dirname(os.path.abspath(__file__)) # parent dir


def atom_features(atom):
    """ (ap) Extract atom features and put into array. """
    # a1 = one_of_k_encoding_unk(atom.GetSymbol(), [
    #         'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
    #         'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
    #         'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
    #         'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
    #     ])
    # a2 = one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # a3 = one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # a4 = one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # a5 = [atom.GetIsAromatic()]
    # arr = np.array(a1 + a2 + a3 + a4 + a5)
    # return arr
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
            'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
            'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
            'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
        ]) + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(atom.GetImplicitValence(),
                                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    """ (ap) Convert SMILES to graph. """
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()  # num atoms in molecule

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()  # return a directed graph
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    # (ap) How is edges list different from edge_index list??
    # It seems that len(edge_index) is twice the size of len(edges)
    return c_size, features, edge_index


"""
The functions below generate datasets for CSG (data from July 2020) - Start
"""

# These are globals for all models
# TODO:
# Where do these go?
raw_datadir_name = "raw_data"
x_datadir_name = "x_data"
y_datadir_name = "y_data"
canc_col_name = "CancID"
drug_col_name = "DrugID"
ge_fname = "ge.parquet"  # cancer feature
smiles_fname = "smiles.csv"  # drug feature
y_file_substr = "rsp"


def read_df(fpath, sep=","):
    """ Load a dataframe. Supports csv and parquet files.
    sep : the sepator in csv file
    """
    assert Path(fpath).exists(), f"File {fpath} was not found."
    if "parquet" in str(fpath):
        df = pd.read_parquet(fpath)
    else:
        df = pd.read_csv(fpath, sep=sep)
    return df


def scale_fea(xdata, scaler_name='stnd', dtype=np.float32, verbose=False):
    """ Returns the scaled dataframe of features. """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    if scaler_name is None:
        if verbose:
            print('Scaler is None (not scaling).')
        return xdata
    
    if scaler_name == 'stnd':
        scaler = StandardScaler()
    elif scaler_name == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_name == 'rbst':
        scaler = RobustScaler()
    else:
        print(f'The specified scaler {scaler_name} is not supported (not scaling).')
        return xdata

    cols = xdata.columns
    return pd.DataFrame(scaler.fit_transform(xdata), columns=cols, dtype=dtype)


def load_rsp_data(src_raw_datadir, y_col_name="AUC"):
    """ Read drug response response file and return a datarame with cancer ids,
    drug ids, and drug response values.
    src_raw_datadir : data dir where the raw DRP data is stored
    y_col_name : Drug sensitivity score/metric (e.g., AUC, IC50)
    """
    # TODO: Should this be a standard CANDLE/IMPROVE function?
    pathlist = list(Path(src_raw_datadir/y_datadir_name).glob(f"{y_file_substr}*.csv"))  # glob csv files that contain response data
    pathlist = [p for p in pathlist if "full" not in str(p)]  # get the file that contains the full dataset
    rsp_df = pd.read_csv(pathlist[0])  # there should be only one suitable file
    rsp_df = rsp_df[[drug_col_name, canc_col_name, y_col_name]]  # [drug id, cancer id, response]
    # print(rsp_df[[canc_col_name, drug_col_name]].nunique())
    return rsp_df


def raw_drp_to_ml_data(args):
    """ Generate a single ML data file from raw DRP data. The raw DRP data is
    defined as IMPROVE doc website. """

    # import ipdb; ipdb.set_trace()

    # Main data dir (e.g., CANDLE_DATADIR, IMPROVE_DATADIR)
    # TODO:
    # What shoud it be and how this should be specified? config_file?
    # Keep in mind the unified interface.
    IMPROVE_DATADIR = fdir/"improve_data_dir"

    # -------------------
    # Specify path for raw DRP data (download data from ftp)
    # E.g., improve_data_dir/raw_data/data.ccle
    RAW_DATADIR = IMPROVE_DATADIR/raw_datadir_name  # contains folders "data.{source_data_name}" with raw DRP data
    src_raw_datadir = RAW_DATADIR/f"data.{args.source_data_name}"  # folder of specific data source with raw DRP data

    # Download raw DRP data (which inludes the data splits)
    # download = True
    download = False
    # TODO:
    # Where this should be specified? config_file?
    if download:
        ftp_origin = f"https://ftp.mcs.anl.gov/pub/candle/public/improve/CSG_data"
        data_file_list = [f"data.{args.source_data_name}.zip"]
        for f in data_file_list:
            candle.get_file(fname=f,
                            origin=os.path.join(ftp_origin, f.strip()),
                            unpack=False, md5_hash=None,
                            cache_subdir=None,
                            datadir=RAW_DATADIR) # cache_subdir=args.cache_subdir

    # -------------------
    # Response data (general func for all models)
    # TODO:
    # This command should be used in preprocess of all models.
    # Should this be a standard in CANDLE/IMPROVE?
    rsp_df = load_rsp_data(src_raw_datadir, y_col_name=args.y_col_name)
    print(rsp_df[[canc_col_name, drug_col_name]].nunique())

    # -------------------
    # Drugs data (general func for all models)
    smi = read_df(src_raw_datadir/x_datadir_name/smiles_fname)

    # Drug featurization for GraphDRP (specific to GraphDRP)
    d_dict = {v: i for i, v in enumerate(smi[drug_col_name].values)}  # drug_dict; len(d_dict): 311
    d_smile = smi["SMILES"].values  # drug_smile
    smile_graph = {}  # smile_graph
    dd = {d_id: s for d_id, s in zip(smi[drug_col_name].values, smi["SMILES"].values)}
    for smile in d_smile:
        g = smile_to_graph(smile)  # g: [c_size, features, edge_index]
        smile_graph[smile] = g

    # print("Unique drugs: {}".format(len(d_dict)))
    # print("Unique smiles: {}".format(len(smile_graph)))

    # -------------------
    # Cancer data (general func for all models)
    # pathlist = list(Path(src_raw_datadir/x_data_subdir).glob(ge_fname))  # glob gene expression files
    # ge = read_df(pathlist[0])
    ge = read_df(src_raw_datadir/x_datadir_name/ge_fname)

    # Use landmark genes (for gene selection)
    # TODO:
    # Eventually, lincs genes will provided with the raw DRP data (check with data curation team).
    # Thus, it will be part of CANDLE/IMPROVE functions.
    use_lincs = True
    if use_lincs:
        # with open(Path(src_raw_datadir)/"../landmark_genes") as f:
        with open(fdir/"landmark_genes") as f:
            genes = [str(line.rstrip()) for line in f]
        genes = ["ge_" + str(g) for g in genes]
        print("Genes count: {}".format(len(set(genes).intersection(set(ge.columns[1:])))))
        genes = list(set(genes).intersection(set(ge.columns[1:])))
        cols = [canc_col_name] + genes
        ge = ge[cols]

    # Scale gene expression data
    # TODO:
    # We might need to save the scaler object (needs to be applied to test/infer data).
    ge_xdata = ge.iloc[:, 1:]
    ge_xdata_scaled = scale_fea(ge_xdata, scaler_name='stnd', dtype=np.float32, verbose=False)
    ge = pd.concat([ge[[canc_col_name]], ge_xdata_scaled], axis=1)

    # Below is omic data preparation for GraphDRP
    # ge = ge.iloc[:, :1000]  # Take subset of cols (genes)
    c_dict = {v: i for i, v in enumerate(ge[canc_col_name].values)}  # cell_dict; len(c_dict): 634
    c_feature = ge.iloc[:, 1:].values  # cell_feature
    cc = {c_id: ge.iloc[i, 1:].values for i, c_id in enumerate(ge[canc_col_name].values)}  # cell_dict; len(c_dict): 634

    # -------------------
    # Get data splits and create folder for saving the generated ML data
    # TODO:
    # Should this be a standard in CANDLE/IMPROVE?

    # Method 1
    # splitdir = Path(os.path.join(src_raw_datadir))/"splits"
    # if len(args.split_file_name) == 1 and args.split_file_name[0] == "full":
    #     # Full dataset (take all samples)
    #     ids = list(range(rsp_df.shape[0]))
    #     outdir_name = "full"
    # else:
    #     # Check if the split file exists and load
    #     ids = []
    #     split_id_str = []    # e.g. split_5
    #     split_type_str = []  # e.g. tr, vl, te
    #     for fname in args.split_file_name:
    #         assert (splitdir/fname).exists(), "split_file_name not found."
    #         with open(splitdir/fname) as f:
    #             # Get the ids
    #             ids_ = [int(line.rstrip()) for line in f]
    #             ids.extend(ids_)
    #             # Get the name
    #             fname_sep = fname.split("_")
    #             split_id_str.append("_".join([s for s in fname_sep[:2]]))
    #             split_type_str.append(fname_sep[2])
    #     assert len(set(split_id_str)) == 1, "Data splits must be from the same dataset source."
    #     split_id_str = list(set(split_id_str))[0]
    #     split_type_str = "_".join([x for x in split_type_str])
    #     outdir_name = f"{split_id_str}_{split_type_str}"
    # ML_DATADIR = IMPROVE_DATADIR/"ml_data"
    # root = ML_DATADIR/f"data.{args.source_data_name}"/outdir_name # ML data
    # os.makedirs(root, exist_ok=True)

    # Method 2
    splitdir = src_raw_datadir/args.splitdir_name
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
    root = fdir/args.outdir
    os.makedirs(root, exist_ok=True)


    # -------------------
    # Folder for saving the generated ML data
    # _data_dir = os.path.split(args.cache_subdir)[0]
    # root = os.getenv('CANDLE_DATA_DIR') + '/' + _data_dir
    # -----
    # ML_DATADIR = IMPROVE_DATADIR/"ml_data"
    # root = ML_DATADIR/f"data.{args.source_data_name}"/outdir_name # ML data
    # -----
    # root = fdir/args.outdir
    # os.makedirs(root, exist_ok=True)


    # -------------------
    # Index data
    # TODO:
    # CANDLE should have this function(?)
    rsp_data = rsp_df.loc[ids]
    rsp_data.to_csv(root/"rsp.csv", index=False)

    # -------------------
    # Extract features and reponse data
    def extract_data_vars(df, d_dict, c_dict, d_smile, c_feature, dd, cc, y_col_name="AUC"):
        """ Get drug and cancer feature data, and response values. """
        xd = []
        xc = []
        y = []
        xd_ = []
        xc_ = []
        nan_rsp_list = []
        miss_cell = []
        miss_drug = []
        meta = []
        # import ipdb; ipdb.set_trace()
        for i in range(df.shape[0]):  # tuples of (drug name, cell id, IC50)
            if i>0 and (i%15000 == 0):
                print(i)
            drug, cell, rsp = df.iloc[i, :].values.tolist()
            if np.isnan(rsp):
                nan_rsp_list.append(rsp)
            # If drug and cell features are available
            if drug in d_dict and cell in c_dict:  # len(drug_dict): 223, len(cell_dict): 990
                xd.append(d_smile[d_dict[drug]])   # xd contains list of smiles
                # xd_.append(dd[drug])   # xd contains list of smiles
                xc.append(c_feature[c_dict[cell]]) # xc contains list of feature vectors
                # xc_.append(cc[cell]) # xc contains list of feature vectors
                y.append(rsp)
                meta.append([drug, cell, rsp])
            elif cell not in c_dict:
                # import ipdb; ipdb.set_trace()
                miss_cell.append(cell)
            elif drug not in d_dict:
                # import ipdb; ipdb.set_trace()
                miss_drug.append(drug)

        # Three arrays of size 191049, as the number of responses
        xd, xc, y = np.asarray(xd), np.asarray(xc), np.asarray(y)
        xd_, xc_ = np.asarray(xd_), np.asarray(xc_)
        meta = pd.DataFrame(meta, columns=[drug_col_name, canc_col_name, y_col_name])

        return xd, xc, y

    xd, xc, y = extract_data_vars(rsp_data, d_dict, c_dict, d_smile, c_feature, dd, cc)
    print("xd ", xd.shape, "xc ", xc.shape, "y_all ", y.shape)

    # -------------------
    # Create and save PyTorch data
    DATA_FILE_NAME = "data"  # TestbedDataset() appends this string with ".pt"
    pt_data = TestbedDataset(
        root=root,
        dataset=DATA_FILE_NAME,
        xd=xd,
        xt=xc,
        y=y,
        smile_graph=smile_graph)
    return root


if __name__ == "__main__":
    fdir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="prepare dataset to train model")
    parser.add_argument(
        "--splitdir_name",
        type=str,
        required=True,
        help="Dir that stores the split files.")
    parser.add_argument(
        "--split_file_name",
        type=str,
        nargs="+",
        required=True,
        help="Split file path in the cross-study analysis.")
    parser.add_argument(
        "--source_data_name",
        type=str,
        required=True,
        help="Data source name.")
    parser.add_argument(
        "--y_col_name",
        type=str,
        required=True,
        help="Drug sensitivity score to use as the target variable.")
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output dir to store the generated ML data files.")

    args = parser.parse_args()
    ml_data_path = raw_drp_to_ml_data(args)
    print(f"\nML data path:\t\n{ml_data_path}")
    print("\nFinished pre-processing.")
