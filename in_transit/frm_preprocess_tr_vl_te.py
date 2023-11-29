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
from sklearn.preprocessing import StandardScaler
import sys

import candle
import improve_utils as imp
from improve_utils import improve_globals as ig


"""
Functions below are used to generate graph molecular structures.
"""

# fdir = os.path.dirname(os.path.abspath(__file__)) # parent dir
fdir = Path(__file__).resolve().parent


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


def raw_data_to_ml_data(args):
    """
    Transform raw drug response prediction (DRP) data to ML data, and save.
    The raw DRP data is defined in the IMPROVE doc website:
    https://jdacs4c-improve.github.io/docs/content/drp_overview.html#raw-drp-data.
    Model-specific func.

    Args:
        args: input arguments

    Returns:
        Path: path to the created ML data file
    """
    # import ipdb; ipdb.set_trace()

    # -------------------
    root = args.outdir
    os.makedirs(root, exist_ok=True)

    # -------------------
    # If we decide to download from ftp
    # TODO:
    # Should the preprocess script take care of data downloading?
    # Where this should be specified? config_file?

    # download = True
    download = False
    if download:
        ftp_origin = f"https://ftp.mcs.anl.gov/pub/candle/public/improve/IMP_data"
        data_file_list = [f"data.{args.source_data_name}.zip"]
        for f in data_file_list:
            candle.get_file(fname=f,
                            origin=os.path.join(ftp_origin, f.strip()),
                            unpack=False, md5_hash=None,
                            cache_subdir=None,
                            datadir=raw_data_dir) # cache_subdir=args.cache_subdir


    # -------------------
    # Response data
    # -------------------

    # Load response data (global func; TODO: use in all models)
    """
    rs = imp.load_single_drug_response_data_v2(
        source=args.train_data_name,
        split_file_name=args.split_file_name,
        y_col_name=args.y_col_name)
    """

    # ~~~~~~~~~~~~~~~~~~
    # Train response data
    rs_tr = imp.load_single_drug_response_data_v2(
        source=args.train_data_name,
        split_file_name=args.train_split_file_name,
        y_col_name=args.y_col_name)

    # Val response data
    rs_vl = imp.load_single_drug_response_data_v2(
        source=args.val_data_name,
        split_file_name=args.val_split_file_name,
        y_col_name=args.y_col_name)

    # Test response data
    rs_te = imp.load_single_drug_response_data_v2(
        source=args.test_data_name,
        split_file_name=args.test_split_file_name,
        y_col_name=args.y_col_name)
    # ~~~~~~~~~~~~~~~~~~

    # -------------------
    # Cancer data
    # -------------------

    # Load omics data (global func; TODO: use in all models)
    ge = imp.load_gene_expression_data(gene_system_identifier="Gene_Symbol")
    ge = ge.reset_index()
    print(ge[ig.canc_col_name].nunique())

    """
    # Retain (canc, drug) response samples for which we have the omic data
    rs, ge = imp.get_common_samples(df1=rs, df2=ge, ref_col=ig.canc_col_name)
    print(rs[[ig.canc_col_name, ig.drug_col_name]].nunique())
    """

    # Retain (canc, drug) response samples for which we have the omic data
    rs_tr, ge_tr = imp.get_common_samples(df1=rs_tr, df2=ge, ref_col=ig.canc_col_name)
    rs_vl, ge_vl = imp.get_common_samples(df1=rs_vl, df2=ge, ref_col=ig.canc_col_name)
    rs_te, ge_te = imp.get_common_samples(df1=rs_te, df2=ge, ref_col=ig.canc_col_name)
    print(rs_tr[[ig.canc_col_name, ig.drug_col_name]].nunique())
    print(rs_vl[[ig.canc_col_name, ig.drug_col_name]].nunique())
    print(rs_te[[ig.canc_col_name, ig.drug_col_name]].nunique())

    # Use landmark genes (gene selection)
    # TODO:
    # Data curataion team.
    # Should the function that load gene expression have an option to load only LINCS genes?
    use_lincs = True
    if use_lincs:
        # with open(Path(src_raw_data_dir)/"../landmark_genes") as f:
        with open(fdir/"landmark_genes") as f:
            genes = [str(line.rstrip()) for line in f]
        # genes = ["ge_" + str(g) for g in genes]  # This is for our legacy data
        print("Genes count: {}".format(len(set(genes).intersection(set(ge.columns[1:])))))
        genes = list(set(genes).intersection(set(ge.columns[1:])))
        cols = [ig.canc_col_name] + genes
        """ ge = ge[cols] """
        ge_tr = ge_tr[cols]
        ge_vl = ge_vl[cols]
        ge_te = ge_te[cols]

    # Scale features
    """
    # TODO:
    # We might need to save the scaler object (needs to be applied to test/infer data).
    ge_x_data = ge.iloc[:, 1:]
    ge_x_data_scaled = scale_fea(ge_x_data, scaler_name="stnd", dtype=np.float32, verbose=False)
    ge = pd.concat([ge[[ig.canc_col_name]], ge_x_data_scaled], axis=1)

    # Below is omic data preparation for GraphDRP
    # ge = ge.iloc[:, :1000]  # Take subset of cols (genes)
    c_dict = {v: i for i, v in enumerate(ge[ig.canc_col_name].values)}  # cell_dict; len(c_dict): 634
    c_feature = ge.iloc[:, 1:].values  # cell_feature
    cc = {c_id: ge.iloc[i, 1:].values for i, c_id in enumerate(ge[ig.canc_col_name].values)}  # cell_dict; len(c_dict): 634
    """
    xtr = ge_tr.iloc[:, 1:]
    xvl = ge_vl.iloc[:, 1:]
    xte = ge_te.iloc[:, 1:]
    xx = pd.concat([xtr, xvl], axis=0)  # Only train and val are create the scaler
    scaler = StandardScaler()
    fea_cols = xx.columns
    scaler.fit(xx)
    xtr_scaled = pd.DataFrame(scaler.transform(xtr), columns=fea_cols, dtype=np.float32)
    xvl_scaled = pd.DataFrame(scaler.transform(xvl), columns=fea_cols, dtype=np.float32)
    xte_scaled = pd.DataFrame(scaler.transform(xte), columns=fea_cols, dtype=np.float32)
    ge_tr = pd.concat([ge_tr[[ig.canc_col_name]], xtr_scaled], axis=1)
    ge_vl = pd.concat([ge_vl[[ig.canc_col_name]], xvl_scaled], axis=1)
    ge_te = pd.concat([ge_te[[ig.canc_col_name]], xte_scaled], axis=1)

    # Below is omic data preparation for GraphDRP
    # ge = ge.iloc[:, :1000]  # Take subset of cols (genes)
    def prep_omic_data(ge):
        c_dict = {v: i for i, v in enumerate(ge[ig.canc_col_name].values)}  # cell_dict; len(c_dict): 634
        c_feature = ge.iloc[:, 1:].values  # cell_feature
        cc = {c_id: ge.iloc[i, 1:].values for i, c_id in enumerate(ge[ig.canc_col_name].values)}  # cell_dict; len(c_dict): 634
        return c_dict, c_feature, cc

    c_dict_tr, c_feature_tr, cc_tr = prep_omic_data(ge=ge_tr)
    c_dict_vl, c_feature_vl, cc_vl = prep_omic_data(ge=ge_vl)
    c_dict_te, c_feature_te, cc_te = prep_omic_data(ge=ge_te)


    # -------------------
    # Drugs data
    # -------------------

    # Load SMILES data (global func; TODO: use in all models)
    smi = imp.load_smiles_data()
    smi = smi.rename(columns={"smiles": "SMILES"})
    # TODO: retain only the relevant drugs??

    # Drug featurization for GraphDRP (specific to GraphDRP)
    d_dict = {v: i for i, v in enumerate(smi[ig.drug_col_name].values)}  # drug_dict; len(d_dict): 311
    d_smile = smi["SMILES"].values  # drug_smile
    smile_graph = {}  # smile_graph
    dd = {d_id: s for d_id, s in zip(smi[ig.drug_col_name].values, smi["SMILES"].values)} # {drug_id: smiles str}
    for smile in d_smile:
        g = smile_to_graph(smile)  # g: [c_size, features, edge_index]
        smile_graph[smile] = g

    # print("Unique drugs: {}".format(len(d_dict)))
    # print("Unique smiles: {}".format(len(smile_graph)))


    # -------------------
    # Save the subset of response data
    # This can be used to merge with model predictions.
    ### rs = imp.get_subset_df(rs, ids)
    ### rs.to_csv(Path(root)/"rsp.csv", index=False)
    """
    rs.to_csv(Path(root)/"response_subset.csv", index=False) ###
    """
    # TODO: consider creating args:
    #   train_y_data_name, val_y_data_name, test_y_data_name
    rs_tr.to_csv(Path(root)/"train_response.csv", index=False) ###
    rs_vl.to_csv(Path(root)/"val_response.csv", index=False) ###
    rs_te.to_csv(Path(root)/"test_response.csv", index=False) ###

    # -------------------
    # Extract features and reponse data
    def extract_data_vars(df, d_dict, c_dict, d_smile, c_feature, dd, cc, y_col_name="AUC"):
        """ Returns drug and cancer feature data, and response values.
        Args:
            df: drug resposne dataframe
            d_dict: dict of {drug_id: drug int}
            c_dict: dict of {canc_id: canc int}
            d_smile: list of smiles
            c_feature: 2-D array of [canc samples, feature vectors]
            dd: dict of {drug_id, smile}
            cc: dict of {canc id: feature vector}
            y_col_name: drug response col name

        Returns:
            xd, xc, y
        """
        xd = []
        xc = []
        y = []
        xd_ = []
        xc_ = []
        nan_rsp_list = []
        miss_cell = []
        miss_drug = []
        meta = []
        for i in range(df.shape[0]):  # tuples of (drug name, cell id, IC50)
            if i > 0 and (i%15000 == 0):
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
                miss_cell.append(cell)
            elif drug not in d_dict:
                miss_drug.append(drug)

        # Three arrays of size 191049, as the number of responses
        xd, xc, y = np.asarray(xd), np.asarray(xc), np.asarray(y)
        xd_, xc_ = np.asarray(xd_), np.asarray(xc_)
        meta = pd.DataFrame(meta, columns=[ig.drug_col_name, ig.canc_col_name, y_col_name])

        return xd, xc, y

    # import ipdb; ipdb.set_trace()
    ### xd, xc, y = extract_data_vars(rsp_data, d_dict, c_dict, d_smile, c_feature, dd, cc)
    """
    df = rs[[ig.drug_col_name, ig.canc_col_name, args.y_col_name]]
    xd, xc, y = extract_data_vars(df, d_dict, c_dict, d_smile, c_feature, dd, cc, args.y_col_name) ### new
    print("xd ", xd.shape, "xc ", xc.shape, "y_all ", y.shape)
    """

    df_tr = rs_tr[[ig.drug_col_name, ig.canc_col_name, args.y_col_name]]
    df_vl = rs_vl[[ig.drug_col_name, ig.canc_col_name, args.y_col_name]]
    df_te = rs_te[[ig.drug_col_name, ig.canc_col_name, args.y_col_name]]
    xd_tr, xc_tr, y_tr = extract_data_vars(df_tr, d_dict, c_dict_tr, d_smile, c_feature_tr, dd, cc_tr, args.y_col_name) ### new
    xd_vl, xc_vl, y_vl = extract_data_vars(df_vl, d_dict, c_dict_vl, d_smile, c_feature_vl, dd, cc_vl, args.y_col_name) ### new
    xd_te, xc_te, y_te = extract_data_vars(df_te, d_dict, c_dict_te, d_smile, c_feature_te, dd, cc_te, args.y_col_name) ### new
    print("xd_tr ", xd_tr.shape, "xc_tr ", xc_tr.shape, "y_tr ", y_tr.shape)
    print("xd_vl ", xd_vl.shape, "xc_vl ", xc_vl.shape, "y_vl ", y_vl.shape)
    print("xd_te ", xd_te.shape, "xc_te ", xc_te.shape, "y_te ", y_tr.shape)

    # -------------------
    # Create and save PyTorch data
    # TODO: should DATA_FILE_NAME be global?
    # DATA_FILE_NAME = "data"  # TestbedDataset() appends this string with ".pt"
    # TODO: should these names be global?
    # TestbedDataset() appends this string with ".pt"
    train_file_name = "train_data"
    val_file_name = "val_data"
    test_file_name = "test_data"

    # Train data
    pt_data = TestbedDataset(
        root=root,
        dataset=train_file_name,
        xd=xd_tr,
        xt=xc_tr,
        y=y_tr,
        smile_graph=smile_graph)

    # Val data
    pt_data = TestbedDataset(
        root=root,
        dataset=val_file_name,
        xd=xd_vl,
        xt=xc_vl,
        y=y_vl,
        smile_graph=smile_graph)

    # Test data
    pt_data = TestbedDataset(
        root=root,
        dataset=test_file_name,
        xd=xd_te,
        xt=xc_te,
        y=y_te,
        smile_graph=smile_graph)

    return root


def parse_args(args):
    """ Parse input args. """

    parser = argparse.ArgumentParser(description="Generate dataset for DL model")

    # IMPROVE required args
    parser.add_argument(
        "--train_data_name",
        type=str,
        required=True,
        help="Data source name.")
    parser.add_argument(
        "--val_data_name",
        type=str,
        default=None,
        required=False,
        help="Data target name (not required for GraphDRP).")
    parser.add_argument(
        "--test_data_name",
        type=str,
        default=None,
        required=False,
        help="Data target name (not required for GraphDRP).")
    # ------------------
    # parser.add_argument(
    #     "--split_file_name",
    #     type=str,
    #     nargs="+",
    #     # required=True,
    #     required=False,
    #     help="The path to the file that contains the split ids (e.g., 'split_0_tr_id', 'split_0_vl_id').")
    # ------------------
    # ~~~~~~~~~~~~~~~~~~
    parser.add_argument(
        "--train_split_file_name",
        type=str,
        nargs="+",
        required=True,
        help="The path to the file that contains the split ids (e.g., 'split_0_tr_id',  'split_0_vl_id').")
    parser.add_argument(
        "--val_split_file_name",
        type=str,
        nargs="+",
        required=True,
        help="The path to the file that contains the split ids (e.g., 'split_0_tr_id',  'split_0_vl_id').")
    parser.add_argument(
        "--test_split_file_name",
        type=str,
        nargs="+",
        required=True,
        help="The path to the file that contains the split ids (e.g., 'split_0_tr_id',  'split_0_vl_id').")
    # ~~~~~~~~~~~~~~~~~~
    parser.add_argument(
        "--y_col_name",
        type=str,
        required=False,
        default="auc",
        help="Drug sensitivity score to use as the target variable (e.g., IC50, AUC).")
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output dir to store the generated ML data files (e.g., 'split_0_tr').")
    # parser.add_argument(
    #     "--receipt",
    #     type=str,
    #     required=False,
    #     help="...")

    args = parser.parse_args(args)
    return args


def main(args):
    # import ipdb; ipdb.set_trace()
    args = parse_args(args)
    ml_data_path = raw_data_to_ml_data(args)
    print(f"\nML data path:\t\n{ml_data_path}")
    print("\nFinished pre-processing (transformed raw DRP data to model input ML data).")
    return ml_data_path


if __name__ == "__main__":
    main(sys.argv[1:])
