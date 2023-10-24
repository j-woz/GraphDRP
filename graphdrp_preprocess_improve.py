""" Functionality for Preprocessing Data for Training a GraphDRP Model. """

import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler

# IMPROVE imports
from improve import framework as frm
# from improve import dataloader as dtl
from improve import drug_resp_pred as drp
# from improve.torch_utils import TestbedDataset
# from improve.rdkit_utils import build_graph_dict_from_smiles_collection

# Model-specific imports
from model_utils.torch_utils import TestbedDataset
# from model_utils.rdkit_utils import build_graph_dict_from_smiles_collection

filepath = Path(__file__).resolve().parent

# General DRP args
# TODO: all these args are specific to the DRP probelm but apply to all DRP models.
# Thus, consider moving this dict somewhere else.
gdrp_data_conf = [
    # {"name": "x_data",
    #  "nargs": "+",
    #  "type": str,
    #  "help": "List of feature files.",
    # },
    # {"name": "y_data",
    #  "nargs": "+",
    #  "type": str,
    #  "help": "List of output files.",
    # },
    {"name": "data_set",
     "type": str,
     "help": "Data set to preprocess.",
    },
    # {"name": "split_id",
    #  "type": int,
    #  "default": 0,
    #  "help": "ID of split to read. This is used to find training/validation/testing \
    #          partitions and read lists of data samples to use for preprocessing.",
    # },
    # {"name": "response_file",
    #  "type": str,
    #  "default": "response.tsv",
    #  "help": "File with response data",
    # },
    # {"name": "cell_file", # TODO. Should this be list?
    #  "type": str,
    #  "default": "cancer_gene_expression.tsv",
    #  "help": "File with cancer feature data",
    # },
    # {"name": "drug_file", # TODO. Should this be list?
    #  "type": str,
    #  "default": "drug_SMILES.tsv",
    #  "help": "File with drug feature data",
    # },
    # {"name": "canc_col_name",
    #  "default": "improve_sample_id",
    #  "type": str,
    #  "help": "Column name that contains the cancer sample ids.",
    # },
    # {"name": "drug_col_name",
    #  "default": "improve_chem_id",
    #  "type": str,
    #  "help": "Column name that contains the drug ids.",
    # },
    # {"name": "gene_system_identifier",
    #  "nargs": "+",
    #  "type": str,
    #  "help": "Gene identifier system to use. Options: 'Entrez', 'Gene_Symbol',\
    #          'Ensembl', 'all', or any list combination.",
    # },
    {"name": "use_lincs",
     "type": frm.str2bool,
     "default": True, # TODO. Should this be False?
     "help": "Flag to indicate if using landmark genes.",
    },

]

req_preprocess_args = [ll["name"] for ll in gdrp_data_conf]

req_preprocess_args.extend(["y_col_name", "model_outdir"])

# TODO: The functions below are general functions relevant to drp.
# 1. check_parameter_consistency()
# 2. raw_data_available()
# 3. check_data_available()
# 4. build_common_data()
# 5. load_response_data()
#
# Thus, should these functions be defined somewhere in ./improve rather than in *_preprocess_*.py?
# How about ./imporve/drug_response_prediction?


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
    # (ap) How is edges list different from edge_index list??
    # It seems that len(edge_index) is twice the size of len(edges)
    return c_size, features, edge_index
# ------------------------------------------------------------

def scale_df(dataf, scaler_name: str="std", scaler=None, verbose=False):
    """ Returns a dataframe with scaled data.

    It can create a new scaler or use the scaler passed or return the
    data as it is. If `scaler_name` is None, no scaling is applied. If
    `scaler` is None, a new scaler is constructed. If `scaler` is not
    None, and `scaler_name` is not None, the scaler passed is used for
    scaling the data frame.

    Args:
        dataf: Pandas dataframe to scale.
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
        return dataf, None

    # Scale data
    # Select only numerical columns in data frame
    df_num = dataf.select_dtypes(include="number")

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
            print(f"The specified scaler {scaler_name} is not implemented (no df scaling).")
            return dataf, None

        # Scale data according to new scaler
        df_norm = scaler.fit_transform(df_num)
    else: # Apply passed scikit scaler
        # Scale data according to specified scaler
        df_norm = scaler.transform(df_num)

    # Copy back scaled data to data frame
    dataf[df_num.columns] = df_norm
    return dataf, scaler
# ------------------------------------------------------------

# TODO: consider moving to ./improve/drug_response_prediction
def check_parameter_consistency(params: Dict):
    """Minimal validation over parameter set.

    :params: Dict params: A Python dictionary of CANDLE/IMPROVE keywords and parsed values.
    """
    # TODO:
    # 1. Why do we need to define response_file var in graphdrp_default_model.txt? I think defining y_data var should be enough.
    # 2. cell_file and drug_file should probably be lists, because certain models use multiple feature types to represent cells and drugs.
    if params["response_file"] not in params["y_data"]:
        message = (f"ERROR ! {params['response_file']} was not listed in params['y_data']. Not guaranteed that it is available.\n")
        warnings.warn(message, RuntimeWarning)
    # TODO. I commented the lines below. Check if need to put them elsewhere.
    # if params["cell_file"] not in params["x_data"]:
    #     message = (f"ERROR ! {params['cell_data']} was not listed in params['x_data']. Not guaranteed that it is available.\n")
    #     warnings.warn(message, RuntimeWarning)
    # if params["drug_file"] not in params["x_data"]:
    #     message = (f"ERROR ! {params['drug_data']} was not listed in params['x_data']. Not guaranteed that it is available.\n")
    #     warnings.warn(message, RuntimeWarning)


# TODO: consider moving to ./improve/data
def raw_data_available(params: Dict) -> frm.DataPathDict:
    """
    Sweep the expected raw data folder and check that files needed for cross-study analysis (CSA) are available.

    :params: Dict params: Dictionary of parameters read

    :return: Path to directories requested stored in dictionary with str key str and Path value.
    :rtype: DataPathDict
    """
    # Expected
    # raw_data -> {splits, x_data, y_data}
    # Make sure that main path exists
    mainpath = Path(os.environ["IMPROVE_DATA_DIR"])
    if mainpath.exists() == False:
        raise Exception(f"ERROR ! {mainpath} not found.\n")

    # Make sure that the raw data directory exists
    inpath = mainpath / params["raw_data_dir"]
    if inpath.exists() == False:
        raise Exception(f"ERROR ! {inpath} not found.\n")

    # Make sure that the data subdirectories exist
    xpath = frm.check_path_and_files(params["x_data_dir"], params["x_data_files"], inpath)
    ypath = frm.check_path_and_files(params["y_data_dir"], params["y_data_files"], inpath)
    spath = frm.check_path_and_files("splits", [], inpath)

    return {"x_data_path": xpath, "y_data_path": ypath, "splits_path": spath}


# TODO: consider moving to ./improve/data
def check_data_available(params: Dict) -> frm.DataPathDict:
    """
    Sweep the expected input paths and check that raw data files needed for preprocessing are available.

    :params: Dict params: Dictionary of parameters read

    :return: Path to directories requested stored in dictionary with str key str and Path value.
    :rtype: DataPathDict
    """
    # Check that raw data is available
    # Expected
    # raw_data -> {splits, x_data, y_data}
    ipathd = raw_data_available(params)

    # Create output directory. Do not complain if it exists.
    # TODO: This was originally called ml_data.
    opath = Path(params["model_outdir"])
    os.makedirs(opath, exist_ok=True)

    # Return in DataPathDict structure
    inputdtd = {"x_data_path": ipathd["x_data_path"],
                "y_data_path": ipathd["y_data_path"],
                "splits_path": ipathd["splits_path"]}
    outputdtd = {"preprocess_path": opath}

    return inputdtd, outputdtd


# # TODO: consider moving to ./improve/drug_response_prediction
# def load_response_data(inpath_dict: frm.DataPathDict,
#         y_file_name: str,
#         source: str,
#         split_id: int,
#         stage: str,
#         canc_col_name = "improve_sample_id",
#         drug_col_name = "improve_chem_id",
#         sep: str = "\t",
#         verbose: bool = True) -> pd.DataFrame:
#     """
#     Returns dataframe with cancer ids, drug ids, and drug response values.
#     Samples from the original drug response file are filtered based on
#     the specified split ids.

#     :params: Dict inpath_dict: Dictionary of paths and info about raw
#              data input directories.
#     :params: str y_file_name: Name of file for reading the y_data.
#     :params: str source: DRP source name.
#     :params: int split_id: Split id. If -1, use all data. Note that this
#              assumes that split_id has been constructed to take into
#              account all the data sources.
#     :params: str stage: Type of partition to read. One of the following:
#              'train', 'val', 'test'.
#     :params: str canc_col_name: Column name that contains the cancer
#              sample ids. Default: "improve_sample_id".
#     :params: str drug_col_name: Column name that contains the drug ids.
#              Default: "improve_chem_id".
#     :params: str sep: Separator used in data file.
#     :params: bool verbose: Flag for verbosity. If True, info about
#              computations is displayed. Default: True.

#     :return: Dataframe that contains single drug response values.
#     :rtype: pd.Dataframe
#     """
#     y_data_file = inpath_dict["y_data"] / y_file_name
#     if y_data_file.exists() == False:
#         raise Exception(f"ERROR ! {y_file_name} file not available.\n")
#     # Read y_data_file
#     df = pd.read_csv(y_data_file, sep=sep)

#     # Get a subset of samples if split_id is different to -1
#     if split_id > -1:
#         # TODO: this should not be encoded like this because other comparison
#         # piplines will have a different split_file_name.
#         # E.g, in learning curve, it will be
#         # f"{source}_split_{split_id}_{stage}_size_{train_size}.txt"
#         # Moreover, we should be able to pass a list of splits.
#         split_file_name = f"{source}_split_{split_id}_{stage}.txt"
#     else:
#         split_file_name = f"{source}_all.txt"
#     insplit = inpath_dict["splits"] / split_file_name
#     if insplit.exists() == False:
#         raise Exception(f"ERROR ! {split_file_name} file not available.\n")
#     ids = pd.read_csv(insplit, header=None)[0].tolist()
#     df = df.loc[ids]

#     df = df.reset_index(drop=True)
#     if verbose:
#         print(f"Data read: {y_file_name}, Filtered by: {split_file_name}")
#         print(f"Shape of constructed response data framework: {df.shape}")
#         print(f"Unique cells:  {df[canc_col_name].nunique()}")
#         print(f"Unique drugs:  {df[drug_col_name].nunique()}")
#     return df


# TODO: consider moving to ./model_utils/...
# xd_tr, xc_tr, y_tr = extract_data_vars(df_tr, d_dict, c_dict_tr, d_smile, c_feature_tr, dd, cc_tr, args.y_col_name)
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

    print("Number of NaN responses:   ", len(nan_rsp_list))
    print("Number of drugs not found: ", len(miss_cell))
    print("Number of cells not found: ", len(miss_drug))

    # # Reset index
    # df_drug = df_drug.reset_index()
    # df_cell = df_cell.reset_index()

    return np.asarray(xd).squeeze(), np.asarray(xc), np.asarray(y)


# TODO: Data loading is done directly in run()
def build_common_data(params: Dict, inputdtd: frm.DataPathDict):
    """Construct common feature data frames.

    :params: Dict params: A Python dictionary of CANDLE/IMPROVE keywords
             and parsed values.
    :params: Dict inputdtd: Path to directories of input data stored in
            dictionary with str key str and Path value.

    :return: drug and cell dataframes and smiles graphs
    :rtype: pd.DataFrame
    """
    fname = [inputdtd["x_data"] / fname for fname in params["x_data_files"]]
    for f in fname:
        if f.exists() == False:
            raise Exception(f"ERROR ! File '{fname}' file not found.\n")

    # -------------------
    # Load drug data
    # -------------------
    # Soft coded for smiles for now
    # fname = inputdtd["x_data"] / params["drug_file"]
    # if fname.exists() == False:
    #     raise Exception(f"ERROR ! Drug data from {fname} file not found.\n")
    # df_drug = dtl.load_drug_data(fname,
    #                              columns=["improve_chem_id", "smiles"],
    #                              )
    df_drug = drp.load_drug_data(fname)

    # TODO: This method converts SMILES into graphs and it's specific to GraphDRP model.
    # I don't think it's a "common" data so we should consider defining it somewhere else.
    smile_graphs = build_graph_dict_from_smiles_collection(df_drug["smiles"].values)

    # -------------------
    # Load cancer data
    # -------------------
    # fname = inputdtd["x_data"] / params["cell_file"]
    # fname = [inputdtd["x_data"] / fname for fname in params["cell_file"]]
    # if fname.exists() == False:
    #     raise Exception(f"ERROR ! Cancer data from {fname} file not found.\n")
    # df_cell_all = drp.load_omics_data(fname,
    #                                   canc_col_name=params["canc_col_name"],
    #                                   gene_system_identifier=params["gene_system_identifier"],
    #                                   )

    df_cell_all = drp.load_omics_data(
        omics_type="gene_expression",
        canc_col_name=params["canc_col_name"],
        gene_system_identifier=params["gene_system_identifier"],
        use_lincs=True
    )

    # Use landmark genes (gene selection)
    if params["use_lincs"]:
        with open(filepath/"landmark_genes") as f: # AWFUL that this is not in data site but in code repo
            genes = [str(line.rstrip()) for line in f]
        # genes = ["ge_" + str(g) for g in genes]  # This is for legacy data
        genes = drp.common_elements(genes, df_cell_all.columns[1:])
        cols = [params["canc_col_name"]] + genes
        df_cell_all = df_cell_all[cols]

    return df_drug, df_cell_all, smile_graphs


# TODO: Functionality from this func was used in run()
def build_stage_dependent_data(params: Dict,
                         inputdtd: frm.DataPathDict,
                         outputdtd: frm.DataPathDict,
                         stage: str,
                         source: str,
                         split_id: int,
                         df_drug: pd.DataFrame,
                         df_cell_all: pd.DataFrame,
                         smile_graphs,
                         scaler,
                         ):
    """Construct feature and ouput arrays according to training stage.

    :params: Dict params: A Python dictionary of CANDLE/IMPROVE keywords
             and parsed values.
    :params: Dict inputdtd: Path to directories of input data stored in
            dictionary with str key str and Path value.
    :params: Dict outputdtd: Path to directories for output data stored
            in dictionary with str key str and Path value.
    :params: str stage: Type of partition to read. One of the following:
             'train', 'val', 'test'.
    :params: str source: DRP source name.
    :params: int split_id: Split id. If -1, use all data. Note that this
             assumes that split_id has been constructed to take into
             account all the data sources.
    :params: pd.Dataframe df_drug: Pandas dataframe with drug features.
    :params: pd.Dataframe df_cell_all: Pandas dataframe with cell features.
    :params: dict smile_graphs: Python dictionary with smiles string as
             key and corresponding graphs as values.
    :params: scikit scaler: Scikit object for scaling data.
    """
    # -----------------------------
    # Load y data according to stage
    # ------------------------------
    df_response = drp.load_response_data(inputdtd,
                                         params["response_file"],
                                         source,
                                         split_id,
                                         # stage,
                                         params["canc_col_name"],
                                         params["drug_col_name"],
                                         )
    # Retain (canc, drug) response samples for which omic data is available
    df_y, df_cell = drp.get_common_samples(df1=df_response,
                                           df2=df_cell_all,
                                           ref_col=params["canc_col_name"])
    print(df_y[[params["canc_col_name"], params["drug_col_name"]]].nunique())

    # Normalize features using training set
    if stage == "train": # Ignore scaler object even if specified
        # Normalize
        df_cell, scaler = drp.scale_df(df_cell, scaler_name=params["scaling"])
        if params["scaling"] is not None and params["scaling"] != "none":
            # Store normalization object
            scaler_fname = outputdtd["preprocess"] / "cell_xdata_scaler.gz"
            joblib.dump(scaler, scaler_fname)
            print("Scaling object created is stored in: ", scaler_fname)
    else:
        # Use passed scikit scaler object
        df_cell, _ = drp.scale_df(df_cell, scaler=scaler)

    # Sub-select desired response column (y_col_name)
    # And reduce response dataframe to 3 columns: drug_id, cell_id and selected drug_response
    df_y = df_y[[params["drug_col_name"], params["canc_col_name"], params["y_col_name"]]]
    # Combine data
    # TODO: is this general func or model-specific?
    xd, xc, y = compose_data_arrays(df_y, df_drug, df_cell, params["drug_col_name"], params["canc_col_name"])
    print("stage ", stage, "--> xd ", xd.shape, "xc ", xc.shape, "y ", y.shape)

    # Save the processed (all) data as PyTorch dataset
    TestbedDataset(root=outputdtd["preprocess"],
                   dataset=stage + "_" + params["data_suffix"],
                   xd=xd,
                   xt=xc,
                   y=y,
                   smile_graph=smile_graphs)

    # Save the subset of y data
    fname = f"{stage}_{params['y_data_suffix']}.csv"
    df_y.to_csv(outputdtd["preprocess"] / fname, index=False)

    return scaler




def run(params):
    """ Execute data pre-processing for GraphDRP model.

    :params: Dict params: A dictionary of CANDLE/IMPROVE keywords and parsed values.
    """
    import pdb; pdb.set_trace()
    # --------------------------------------------
    # Check consistency of parameter specification
    # --------------------------------------------
    # check_parameter_consistency(params)

    # ------------------------------------------------------
    # Check data availability and create output directory
    # ------------------------------------------------------
    # [Req]
    indtd, outdtd = check_data_available(params)
    # indtd is dictionary with input_description: path components
    # outdtd is dictionary with output_description: path components

    # ------------------------------------------------------
    # Build paths [Req]
    # ------------------------------------------------------
    # TODO. This serves the same as check_data_available() but puts the paths
    # into params.
    # Need to decide which method to use!
    params = frm.build_paths(params)

    # ------------------------------------------------------
    # Construct data frames for drug and cell features
    # ------------------------------------------------------
    # [Req]
    # df_drug, df_cell_all, smile_graphs = build_common_data(params, indtd)

    # ------------------------------------------------------
    # Load omics data [Req]
    # If the model uses omics data, then must use the improve
    # lib function to load the needed data. E.g. load_omics_data()
    # ------------------------------------------------------
    # import ipdb; ipdb.set_trace()
    ge = drp.load_omics_data(
        params,
        omics_type="gene_expression",
        canc_col_name=params["canc_col_name"],
        gene_system_identifier=params["gene_system_identifier"],
        use_lincs=True
    )

    # ------------------------------------------------------
    # Load drug data [Req]
    # If the model uses omics data, then must use the improve
    # lib function to load the needed data. E.g. load_smiels_data()
    # ------------------------------------------------------
    # import ipdb; ipdb.set_trace()
    smi = drp.load_smiles_data(params)

    # ------------------------------------------------------
    # Prepare features
    # Drug featurization for GraphDRP (specific to GraphDRP)
    # ------------------------------------------------------
    # import ipdb; ipdb.set_trace()
    # smile_graphs = build_graph_dict_from_smiles_collection(smi["smiles"].values)

    smi = smi.rename(columns={"smiles": "SMILES"})

    # Prep molecular graph data
    drug_smiles = smi["SMILES"].values  # list of smiles
    smiles_graphs = {}  # dict of molecular graphs converted from smiles {smiles: graph}
    for smile in drug_smiles:
        g = smile_to_graph(smile)  # g: [c_size, features, edge_index]
        smiles_graphs[smile] = g

    # -------------------------------------------
    # Construct ML data for every stage
    # -------------------------------------------
    # stages = ["train", "val", "test"]
    # scaler = None
    # for st in stages:
    #     print(f"Building stage: {st}")
    #     source = params["data_set"]
    #     split_id = params["split_id"]
    #     scaler = build_stage_dependent_data(params,
    #                                indtd,
    #                                outdtd,
    #                                st,
    #                                source,
    #                                split_id,
    #                                df_drug,
    #                                df_cell_all,
    #                                smile_graphs,
    #                                scaler,
    #     )

    # import ipdb; ipdb.set_trace()
    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}
    scaler = None
    for stage, split_file_name in stages.items():

        # Load response data
        # import ipdb; ipdb.set_trace()
        # df_response = drp.load_response_data(inputdtd,
        df_response = drp.load_response_data(
            y_data_fpath=params["y_data_path"]/params["y_data_file"],
            source=params["data_set"],
            split_fpath=params["splits_path"]/split_file_name,
            canc_col_name=params["canc_col_name"],
            drug_col_name=params["drug_col_name"],
        )

        # Retain (canc, drug) response samples for which omic data is available
        # df_y, df_cell = drp.get_common_samples(df1=df_response, df2=df_cell_all,
        #                                        ref_col=params["canc_col_name"])
        df_y, df_canc = drp.get_common_samples(df1=df_response, df2=ge,
                                               ref_col=params["canc_col_name"])
        print(df_y[[params["canc_col_name"], params["drug_col_name"]]].nunique())

        # import ipdb; ipdb.set_trace()
        # Normalize features using training set
        if stage == "train": # Ignore scaler object even if specified
            # Normalize
            df_canc, scaler = scale_df(df_canc, scaler_name=params["scaling"])
            if params["scaling"] is not None and params["scaling"] != "none":
                # Store normalization object
                scaler_fname = outdtd["preprocess_path"] / "x_data_gene_expression_scaler.gz"
                joblib.dump(scaler, scaler_fname)
                print("Scaling object created is stored in: ", scaler_fname)
        else:
            # Use passed scikit scaler object
            df_canc, _ = scale_df(df_canc, scaler=scaler)

        # Sub-select desired response column (y_col_name)
        # And reduce response dataframe to 3 columns: drug_id, cell_id and selected drug_response
        df_y = df_y[[params["drug_col_name"], params["canc_col_name"], params["y_col_name"]]]
        # Combine data
        # TODO: is this general func or model-specific?
        # xd, xc, y = compose_data_arrays(df_y, df_drug, df_canc, params["drug_col_name"], params["canc_col_name"])
        xd, xc, y = compose_data_arrays(df_y, smi, df_canc, params["drug_col_name"], params["canc_col_name"])
        print("stage ", stage, "--> xd ", xd.shape, "xc ", xc.shape, "y ", y.shape)

        # Save the processed (all) data as PyTorch dataset
        TestbedDataset(root=outdtd["preprocess_path"],
                       dataset=stage + "_" + params["data_suffix"],
                       xd=xd,
                       xt=xc,
                       y=y,
                       smile_graph=smiles_graphs)

        # Save the subset of y data
        # rs_tr.to_csv(Path(root)/"train_response.csv", index=False) # That's what we originally used
        fname = f"{stage}_{params['y_data_suffix']}.csv"
        df_y.to_csv(outdtd["preprocess_path"] / fname, index=False)

    # return root
    return outdtd["preprocess_path"]


def main():
    params = frm.initialize_parameters(filepath,
                                       default_model="graphdrp_default_model.txt",
                                       additional_definitions = gdrp_data_conf,
                                       required = req_preprocess_args,
                                      )
    ml_data_path = run(params)
    print("\nFinished GraphDRP pre-processing (transformed raw DRP data to model input ML data).")


if __name__ == "__main__":
    main()
