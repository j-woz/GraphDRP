""" Preprocess benchmark data (e.g., CSA data) to generate datasets for the
GraphDRP prediction model.

Required outputs
----------------
All the outputs from this preprocessing script are saved in params["ml_data_outdir"].

1. Model input data files.
   This script creates three data files corresponding to train, validation,
   and test data. These data files are used as inputs to the ML/DL model in
   the train and infer scripts. The file format is specified by
   params["data_format"].
   For GraphDRP, the generated files:
        train_data.pt, val_data.pt, test_data.pt

2. Y data files.
   The script creates dataframes with true y values and additional metadata.
   Generated files:
        train_y_data.csv, val_y_data.csv, and test_y_data.csv.
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm
from improve import drug_resp_pred as drp

# Model-specific imports
from model_utils.torch_utils import TestbedDataset

filepath = Path(__file__).resolve().parent # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_preproc_params
# 2. model_preproc_params
# 
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
# Note! This list should not be modified (i.e., no params should added or
# removed from the list.
# 
# There are two types of params in the list: default and required
# default:   default values should be used
# required:  these params must be specified for the model in the param file
app_preproc_params = [
    {"name": "y_data_files", # default
     "type": str,
     "help": "List of files that contain the y (prediction variable) data. \
             Example: [['response.tsv']]",
    },
    {"name": "x_data_canc_files", # required
     "type": str,
     "help": "List of feature files including gene_system_identifer. Examples: \n\
             1) [['cancer_gene_expression.tsv', ['Gene_Symbol']]] \n\
             2) [['cancer_copy_number.tsv', ['Ensembl', 'Entrez']]].",
    },
    {"name": "x_data_drug_files", # required
     "type": str,
     "help": "List of feature files. Examples: \n\
             1) [['drug_SMILES.tsv']] \n\
             2) [['drug_SMILES.tsv'], ['drug_ecfp4_nbits512.tsv']]",
    },
    {"name": "canc_col_name",
     "default": "improve_sample_id", # default
     "type": str,
     "help": "Column name in the y (response) data file that contains the cancer sample ids.",
    },
    {"name": "drug_col_name", # default
     "default": "improve_chem_id",
     "type": str,
     "help": "Column name in the y (response) data file that contains the drug ids.",
    },

]

# 2. Model-specific params (Model: GraphDRP)
# All params in model_preproc_params are optional.
# If no params are required by the model, then it should be an empty list.
model_preproc_params = [
    {"name": "use_lincs",
     "type": frm.str2bool,
     "default": True,
     "help": "Flag to indicate if landmark genes are used for gene selection.",
    },
    {"name": "scaling",
     "type": str,
     "default": "std",
     "choice": ["std", "minmax", "miabs", "robust"],
     "help": "Scaler for gene expression data.",
    },
    {"name": "ge_scaler_fname",
     "type": str,
     "default": "x_data_gene_expression_scaler.gz",
     "help": "File name to save the gene expression scaler object.",
    },
]

# Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
preprocess_params = app_preproc_params + model_preproc_params
# req_preprocess_args = []
# ---------------------


# ------------------------------------------------------------
# Utils to generate molecular graphs from SMILES
# ------------------------------------------------------------
from rdkit import Chem
import networkx as nx

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
    """ Maps inputs not in the allowable set to the last element. """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    """ Convert SMILES to graph. """
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
    # (ap) How does the edges list different from edge_index list??
    # It seems that len(edge_index) is twice the size of len(edges)
    return c_size, features, edge_index
# ------------------------------------------------------------

def gene_selection(df, genes_fpath, canc_col_name):
    """ Takes a dataframe omics data (e.g., gene expression) and retains only
    the genes specified in genes_fpath.
    """
    with open(genes_fpath) as f:
        genes = [str(line.rstrip()) for line in f]
    # genes = ["ge_" + str(g) for g in genes]  # This is for our legacy data
    # print("Genes count: {}".format(len(set(genes).intersection(set(df.columns[1:])))))
    # genes = list(set(genes).intersection(set(df.columns[1:])))
    genes = drp.common_elements(genes, df.columns[1:])
    cols = [canc_col_name] + genes
    return df[cols]


def scale_df(df, scaler_name: str="std", scaler=None, verbose: bool=False):
    """ Returns a dataframe with scaled data.

    It can create a new scaler or use the scaler passed or return the
    data as it is. If `scaler_name` is None, no scaling is applied. If
    `scaler` is None, a new scaler is constructed. If `scaler` is not
    None, and `scaler_name` is not None, the scaler passed is used for
    scaling the data frame.

    Args:
        df: Pandas dataframe to scale.
        scaler_name: Name of scikit learn scaler to apply. Options:
                     ["minabs", "minmax", "std", "none"]. Default: std
                     standard scaling.
        scaler: Scikit object to use, in case it was created already.
                Default: None, create scikit scaling object of
                specified type.
        verbose: Flag specifying if verbose message printing is desired.
                 Default: False, no verbose print.

    Returns:
        pd.Dataframe: dataframe that contains drug response values.
        scaler: Scikit object used for scaling.
    """
    if scaler_name is None or scaler_name == "none":
        if verbose:
            print("Scaler is None (no df scaling).")
        return df, None

    # Scale data
    # Select only numerical columns in data frame
    df_num = df.select_dtypes(include="number")

    if scaler is None: # Create scikit scaler object
        if scaler_name == "std":
            scaler = StandardScaler()
        elif scaler_name == "minmax":
            scaler = MinMaxScaler()
        elif scaler_name == "minabs":
            scaler = MaxAbsScaler()
        elif scaler_name == "robust":
            scaler = RobustScaler()
        else:
            print(f"The specified scaler ({scaler_name}) is not implemented (no df scaling).")
            return df, None

        # Scale data according to new scaler
        df_norm = scaler.fit_transform(df_num)
    else: # Apply passed scikit scaler
        # Scale data according to specified scaler
        df_norm = scaler.transform(df_num)

    # Copy back scaled data to data frame
    df[df_num.columns] = df_norm
    return df, scaler
# ------------------------------------------------------------


# TODO. Remove (not used)
# def check_parameter_consistency(params: Dict):
#     """Minimal validation over parameter set.

#     :params: Dict params: A Python dictionary of CANDLE/IMPROVE keywords and parsed values.
#     """
#     if params["response_file"] not in params["y_data"]:
#         message = (f"ERROR ! {params['response_file']} was not listed in params['y_data']. Not guaranteed that it is available.\n")
#         warnings.warn(message, RuntimeWarning)
#     # if params["cell_file"] not in params["x_data"]:
#     #     message = (f"ERROR ! {params['cell_data']} was not listed in params['x_data']. Not guaranteed that it is available.\n")
#     #     warnings.warn(message, RuntimeWarning)
#     # if params["drug_file"] not in params["x_data"]:
#     #     message = (f"ERROR ! {params['drug_data']} was not listed in params['x_data']. Not guaranteed that it is available.\n")
#     #     warnings.warn(message, RuntimeWarning)


# TODO. Remove (not used)
# def raw_data_available(params: Dict) -> frm.DataPathDict:
#     """
#     Sweep the expected raw data folder and check that files needed for cross-study analysis (CSA) are available.

#     :params: Dict params: Dictionary of parameters read

#     :return: Path to directories requested stored in dictionary with str key str and Path value.
#     :rtype: DataPathDict
#     """
#     # Expected
#     # raw_data -> {splits, x_data, y_data}
#     # Make sure that main path exists
#     mainpath = Path(os.environ["IMPROVE_DATA_DIR"])
#     if mainpath.exists() == False:
#         raise Exception(f"ERROR ! {mainpath} not found.\n")

#     # Make sure that the raw data directory exists
#     inpath = mainpath / params["raw_data_dir"]
#     if inpath.exists() == False:
#         raise Exception(f"ERROR ! {inpath} not found.\n")

#     # Make sure that the data subdirectories exist
#     xpath = frm.check_path_and_files(params["x_data_dir"], params["x_data_files"], inpath)
#     ypath = frm.check_path_and_files(params["y_data_dir"], params["y_data_files"], inpath)
#     spath = frm.check_path_and_files("splits", [], inpath)

#     return {"x_data_path": xpath, "y_data_path": ypath, "splits_path": spath}


# TODO. Remove (not used)
# def check_data_available(params: Dict) -> frm.DataPathDict:
#     """
#     Sweep the expected input paths and check that raw data files needed for preprocessing are available.

#     :params: Dict params: Dictionary of parameters read

#     :return: Path to directories requested stored in dictionary with str key str and Path value.
#     :rtype: DataPathDict
#     """
#     # Check that raw data is available
#     # Expected
#     # raw_data -> {splits, x_data, y_data}
#     ipathd = raw_data_available(params)

#     # Create output directory. Do not complain if it exists.
#     opath = Path(params["model_outdir"]) # this was originally called ml_data
#     os.makedirs(opath, exist_ok=True)

#     # Return in DataPathDict structure
#     inputdtd = {"x_data_path": ipathd["x_data_path"],
#                 "y_data_path": ipathd["y_data_path"],
#                 "splits_path": ipathd["splits_path"]}
#     outputdtd = {"preprocess_path": opath}

#     return inputdtd, outputdtd


def compose_data_arrays(df_response: pd.DataFrame,
                        df_drug: pd.DataFrame,
                        df_cell: pd.DataFrame,
                        drug_col_name: str,
                        canc_col_name: str):
    """ Returns drug and cancer feature data, and response values.

    :params: pd.Dataframe df_response: drug response dataframe. This
             already has been filtered to three columns: drug_id,
             cell_id and drug_response.
    :params: pd.Dataframe df_drug: drug features dataframe.
    :params: pd.Dataframe df_cell: cell features dataframe.
    :params: str drug_col_name: Column name that contains the drug ids.
    :params: str canc_col_name: Column name that contains the cancer sample ids.

    :return: Numpy arrays with drug features, cell features and responses
            xd, xc, y
    :rtype: np.array
    """
    xd = [] # To collect drug features
    xc = [] # To collect cell features
    y = []  # To collect responses

    # To collect missing or corrupted data
    nan_rsp_list = []
    miss_cell = []
    miss_drug = []
    # count_nan_rsp = 0
    # count_miss_cell = 0
    # count_miss_drug = 0

    # Convert to indices for rapid lookup (??)
    df_drug = df_drug.set_index([drug_col_name])
    df_cell = df_cell.set_index([canc_col_name])

    for i in range(df_response.shape[0]):  # tuples of (drug name, cell id, response)
        if i > 0 and (i%15000 == 0):
            print(i)
        drug, cell, rsp = df_response.iloc[i, :].values.tolist()
        if np.isnan(rsp):
            # count_nan_rsp += 1
            nan_rsp_list.append(i)
        # If drug and cell features are available
        try: # Look for drug
            drug_features = df_drug.loc[drug]
        except KeyError: # drug not found
            miss_drug.append(drug)
            # count_miss_drug += 1
        else: # Look for cell
            try:
                cell_features = df_cell.loc[cell]
            except KeyError: # cell not found
                miss_cell.append(cell)
                # count_miss_cell += 1
            else: # Both drug and cell were found
                xd.append(drug_features.values) # xd contains list of drug feature vectors
                xc.append(cell_features.values) # xc contains list of cell feature vectors
                y.append(rsp)

    # print("Number of NaN responses:   ", len(nan_rsp_list))
    # print("Number of drugs not found: ", len(miss_cell))
    # print("Number of cells not found: ", len(miss_drug))

    # # Reset index
    # df_drug = df_drug.reset_index()
    # df_cell = df_cell.reset_index()

    return np.asarray(xd).squeeze(), np.asarray(xc), np.asarray(y)


# [Req]
def run(params: Dict):
    """ Run data preprocessing.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        str: directory name that was used to save the preprocessed (generated)
            ML data files.
    """
    # import pdb; pdb.set_trace()

    # --------------------------------------------
    # Check consistency of parameter specification
    # --------------------------------------------
    # check_parameter_consistency(params)

    # ------------------------------------------------------
    # Check data availability and create output directory
    # ------------------------------------------------------
    # indtd, outdtd = check_data_available(params)
    # indtd is dictionary with input_description: path components
    # outdtd is dictionary with output_description: path components

    # ------------------------------------------------------
    # [Req] Build paths and create output dir
    # ------------------------------------------------------
    # Build paths for raw_data, x_data, y_data, splits
    params = frm.build_paths(params)  

    # Create output dir for model input data (to save preprocessed ML data)
    frm.create_outdir(outdir=params["ml_data_outdir"])

    # ------------------------------------------------------
    # Construct data frames for drug and cell features
    # ------------------------------------------------------
    # df_drug, df_cell_all, smile_graphs = build_common_data(params, indtd)

    # ------------------------------------------------------
    # [Req] Load X data (feature representations)
    # ------------------------------------------------------
    # Use the provided data loaders to load data that is required by the model.
    #
    # Benchmark data includes three dirs: x_data, y_data, splits.
    # The x_data contains files that represent feature information such as
    # cancer representation (e.g., omics) and drug representation (e.g., SMILES).
    #
    # Prediction models utilize various types of feature representations.
    # Drug response prediction (DRP) models generally use omics and drug features.
    #
    # If the model uses omics data types that are provided as part of the benchmark
    # data, then the model must use the provided data loaders to load the data files
    # from the x_data dir.
    print("\nLoads omics data.")
    omics_obj = drp.OmicsLoader(params)
    # print(omics_obj)
    ge = omics_obj.dfs['cancer_gene_expression.tsv'] # return gene expression

    print("\nLoad drugs data.")
    # drugs_obj = drp.DrugsLoader(params)
    # # print(drugs_obj)
    # md = drugs_obj.dfs['drug_mordred.tsv'] # return the Mordred descriptors
    # md = md.reset_index()  # TODO. implement reset_index() inside the loader
    drugs_obj = drp.DrugsLoader(params)
    # print(dd)
    smi = drugs_obj.dfs['drug_SMILES.tsv']  # get only the SMILES data

    # # ------------------------------------------------------
    # # [Req] Load omics data
    # # ---------------------
    # print("\nLoading omics data ...")
    # oo = drp.OmicsLoader(params)
    # # print(oo)
    # ge = oo.dfs['cancer_gene_expression.tsv']  # get only gene expression dataframe
    # # ---------------------

    # ------------------------------------------------------
    # Further preprocess X data
    # ------------------------------------------------------
    # Gene selection (based on LINCS landmark genes)
    if params["use_lincs"]:
        genes_fpath = filepath/"landmark_genes"
        ge = gene_selection(ge, genes_fpath, canc_col_name=params["canc_col_name"])

    # Prefix gene column names with "ge."
    fea_sep = "."
    fea_prefix = "ge"
    ge = ge.rename(columns={fea: f"{fea_prefix}{fea_sep}{fea}" for fea in ge.columns[1:]})

    # Prep molecular graph data for GraphDRP
    smi = smi.reset_index()
    smi.columns = [params["drug_col_name"], "SMILES"]
    drug_smiles = smi["SMILES"].values  # list of smiles
    smiles_graphs = {}  # dict of molecular graphs converted from smiles {smiles: graph}
    for smile in drug_smiles:
        g = smile_to_graph(smile)  # g: [c_size, features, edge_index]
        smiles_graphs[smile] = g

    # # ------------------------------------------------------
    # # [GraphDRP] Prep omics data
    # # ------------------------------------------------------
    # # Gene selection (LINCS landmark genes) for GraphDRP
    # if params["use_lincs"]:
    #     genes_fpath = filepath/"landmark_genes"
    #     ge = gene_selection(ge, genes_fpath, canc_col_name=params["canc_col_name"])

    # # ------------------------------------------------------
    # # [Req] Load drug data
    # # --------------------
    # print("\nLoading drugs data...")
    # dd = drp.DrugsLoader(params)
    # # print(dd)
    # smi = dd.dfs['drug_SMILES.tsv']  # get only the SMILES data
    # # --------------------

    # # ------------------------------------------------------
    # # [GraphDRP] Prep drug features
    # # ------------------------------------------------------
    # # smile_graphs = build_graph_dict_from_smiles_collection(smi["smiles"].values)

    # # Prep molecular graph data for GraphDRP
    # smi = smi.reset_index()
    # smi.columns = [params["drug_col_name"], "SMILES"]
    # drug_smiles = smi["SMILES"].values  # list of smiles
    # smiles_graphs = {}  # dict of molecular graphs converted from smiles {smiles: graph}
    # for smile in drug_smiles:
    #     g = smile_to_graph(smile)  # g: [c_size, features, edge_index]
    #     smiles_graphs[smile] = g

    # ------------------------------------------------------
    # Create feature scaler
    # ------------------------------------------------------
    # Load and combine responses
    print("Create feature scaler.")
    rsp_tr = drp.DrugResponseLoader(params,
                                    split_file=params["train_split_file"],
                                    verbose=False).dfs["response.tsv"]
    rsp_vl = drp.DrugResponseLoader(params,
                                    split_file=params["val_split_file"],
                                    verbose=False).dfs["response.tsv"]
    rsp = pd.concat([rsp_tr, rsp_vl], axis=0)

    # Retian feature rows that are present in the y data (response dataframe)
    # Intersection of omics features, drug features, and responses
    rsp = rsp.merge( ge[params["canc_col_name"]], on=params["canc_col_name"], how="inner")
    rsp = rsp.merge(smi[params["drug_col_name"]], on=params["drug_col_name"], how="inner")
    ge_sub = ge[ge[params["canc_col_name"]].isin(rsp[params["canc_col_name"]])].reset_index(drop=True)
    # smi_sub = smi[smi[params["drug_col_name"]].isin(rsp[params["drug_col_name"]])].reset_index(drop=True)
    # TODO: keep only smiles (smiles_graphs) in smi_sub

    # Scale gene expression
    _, ge_scaler = scale_df(ge_sub, scaler_name=params["scaling"])
    ge_scaler_fpath = Path(params["ml_data_outdir"]) / params["ge_scaler_fname"]
    joblib.dump(ge_scaler, ge_scaler_fpath)
    print("Scaler object for gene expression: ", ge_scaler_fpath)

    # # Scale Mordred descriptors
    # _, md_scaler = scale_df(md_sub, scaler_name=params["scaling"])
    # md_scaler_fpath = Path(params["ml_data_outdir"]) / params["md_scaler_fname"]
    # joblib.dump(md_scaler, md_scaler_fpath)
    # print("Scaler object for Mordred:         ", md_scaler_fpath)

    del rsp, rsp_tr, rsp_vl, ge_sub

    # ------------------------------------------------------
    # [Req] Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------
    # All models must load response data (y data) using DrugResponseLoader().
    # Below, we iterate over the 3 split files (train, val, test) and load
    # response data, filtered by the split ids from the split files.

    # Dict with split files corresponding to the three sets (train, val, and test)
    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}
    # scaler = None

    # # %%%%%%%%%%%%%%%
    # # Create scaler
    # # import ipdb; ipdb.set_trace()
    # # split_file_dev = params["train_split_file"] + \
    # #                  params["val_split_file"] # + \
    #                  # params["test_split_file"]
    # rs_tr = drp.DrugResponseLoader(params,
    #                                split_file=params["train_split_file"],
    #                                verbose=False).dfs["response.tsv"]
    # rs_vl = drp.DrugResponseLoader(params,
    #                                split_file=params["val_split_file"],
    #                                verbose=False).dfs["response.tsv"]
    # rs_dev = pd.concat([rs_tr, rs_vl], axis=0)
    # # print(rs_tr.shape) 
    # # print(rs_vl.shape) 
    # # print("rs_dev", rs_dev.shape) 
    # # print("ge", ,ge.shape) 
    # ydf, df_canc = drp.get_common_samples(df1=rs_dev, df2=ge,
    #                                       ref_col=params["canc_col_name"])
    # # print("ydf", ydf.shape) 
    # # print("df_canc", df_canc.shape) 
    # df_canc, scaler = scale_df(df_canc, scaler_name=params["scaling"])
    # scaler_fpath = Path(params["ml_data_outdir"]) / params["scaler_fname"]
    # joblib.dump(scaler, scaler_fpath)
    # print("Scaler object created and stored in: ", scaler_fpath)
    # # %%%%%%%%%%%%%%%

    for stage, split_file in stages.items():

        # --------------------------------
        # [Req] Load response data
        # --------------------------------
        rsp = drp.DrugResponseLoader(params,
                                     split_file=split_file,
                                     verbose=False).dfs["response.tsv"]

        # --------------------------------
        # Data prep
        # --------------------------------
        # Retain (canc, drug) responses for which both omics and drug features
        # are available.
        # ydf, df_canc = drp.get_common_samples(df1=df_response, df2=ge,
        #                                       ref_col=params["canc_col_name"])
        # print(ydf[[params["canc_col_name"], params["drug_col_name"]]].nunique())

        rsp = rsp.merge( ge[params["canc_col_name"]], on=params["canc_col_name"], how="inner")
        rsp = rsp.merge(smi[params["drug_col_name"]], on=params["drug_col_name"], how="inner")
        ge_sub = ge[ge[params["canc_col_name"]].isin(rsp[params["canc_col_name"]])].reset_index(drop=True)
        smi_sub = smi[smi[params["drug_col_name"]].isin(rsp[params["drug_col_name"]])].reset_index(drop=True)

        # Scale features
        # df_canc, _ = scale_df(df_canc, scaler=scaler)
        ge_sc, _ = scale_df(ge_sub, scaler=ge_scaler) # scale gene expression
        # print("GE mean:", ge_sc.iloc[:,1:].mean(axis=0).mean())
        # print("GE var: ", ge_sc.iloc[:,1:].var(axis=0).mean())

        # # Scale features using training data
        # if stage == "train":
        #     # Scale data
        #     df_canc, scaler = scale_df(df_canc, scaler_name=params["scaling"])
        #     # Store scaler object
        #     if params["scaling"] is not None and params["scaling"] != "none":
        #         scaler_fpath = Path(params["ml_data_outdir"]) / params["scaler_fname"]
        #         joblib.dump(scaler, scaler_fpath)
        #         print("Scaler object created and stored in: ", scaler_fpath)
        # else:
        #     # Use passed scikit scaler object
        #     df_canc, _ = scale_df(df_canc, scaler=scaler)

        # Sub-select desired response column (y_col_name)
        # And reduce response dataframe to 3 columns: drug_id, cell_id and selected drug_response
        # ydf = ydf[[params["drug_col_name"], params["canc_col_name"], params["y_col_name"]]]
        rsp_cut = rsp[[params["drug_col_name"], params["canc_col_name"], params["y_col_name"]]].copy()
        # Further prepare data (model-specific)
        xd, xc, y = compose_data_arrays(rsp_cut, smi, ge_sc, params["drug_col_name"], params["canc_col_name"])
        print(stage.upper(), "data --> xd ", xd.shape, "xc ", xc.shape, "y ", y.shape)

        # -----------------------
        # [Req] Save ML data files in params["ml_data_outdir"]
        # The implementation of this step depends on the model.
        # -----------------------
        # import ipdb; ipdb.set_trace()

        # [Req] Create data name
        # data_fname = frm.build_ml_data_name(params, stage,
        #                                     file_format=params["data_format"])
        data_fname = frm.build_ml_data_name(params, stage)

        # Revmoe data_format because TestbedDataset() appends '.pt' to the
        # file name automatically. This is unique for GraphDRP.
        data_fname = data_fname.split(params["data_format"])[0]

        # Create the ml data and save it as data_fname in params["ml_data_outdir"]
        # Note! In the *train*.py and *infer*.py scripts, functionality should
        # be implemented to load the saved data.
        # -----
        # In GraphDRP, TestbedDataset() is used to create and save the file.
        # TestbedDataset() which inherits from torch_geometric.data.InMemoryDataset
        # automatically creates dir called "processed" inside root and saves the file
        # inside. This results in: [root]/processed/[dataset],
        # e.g., ml_data/processed/train_data.pt
        # -----
        TestbedDataset(root=params["ml_data_outdir"],
                       dataset=data_fname,
                       xd=xd,
                       xt=xc,
                       y=y,
                       smile_graph=smiles_graphs)

        # # [Req] Save y dataframe for the current stage
        # frm.save_stage_ydf(ydf, params, stage)

        # [Req] Save y dataframe for the current stage
        # frm.save_stage_ydf(ydf, params, stage)
        frm.save_stage_ydf(rsp, params, stage)

    return params["ml_data_outdir"]


# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_params
    params = frm.initialize_parameters(
        filepath,
        # default_model="graphdrp_default_model.txt",
        # default_model="graphdrp_params.txt",
        # default_model="params_ws.txt",
        default_model="params_cs.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    ml_data_outdir = run(params)
    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
