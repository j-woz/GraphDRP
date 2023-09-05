"""Functionality for Pre-processing in Cross-Study Analysis (CSA) for GraphDRP Model."""

from pathlib import Path
import os

from typing import Deque, Dict, List, Tuple, Union

import pandas as pd

from improve import csa
from improve import dataloader as dtl
from graphdrp_train_improve import gdrp_data_conf

filepath = Path(__file__).resolve().parent


csa_gdrp_conf = [
    {"name": "gene_system_identifier",
     "nargs": "+",
     "type": str,
     "help": "Gene identifier system to use. Options: 'Entrez', 'Gene_Symbol', 'Ensembl', 'all', or any list combination.",
    },
    {"name": "use_lincs",
     "type": csa.frm.str2bool,
     "default": True,
     "help": "Flag to indicate if using landmark genes.",
    },
    {"name": "response_file",
     "type": str,
     "default": "response.tsv",
     "help": "File with response data",
    },
    {"name": "cell_file",
     "type": str,
     "default": "cancer_gene_expression.tsv",
     "help": "File with cancer feature data",
    },
    {"name": "drug_file",
     "type": str,
     "default": "drug_SMILES.tsv",
     "help": "File with drug feature data",
    },
]


def load_response_data(inpath_dict: csa.frm.DataPathDict,
        y_file_name: str,
        source: str,
        split_id: int,
        stage: str,
        y_col_name: str = "auc",
        canc_col_name = "improve_sample_id",
        drug_col_name = "improve_chem_id",
        sep: str = "\t",
        verbose: bool = True) -> pd.DataFrame:
    """
    Returns dataframe with cancer ids, drug ids, and drug response values. Samples
    from the original drug response file are filtered based on the specified
    split ids.

    Args:
        inpath_dict: Dictionary of paths and info about raw data input directories.
        y_file_name: Name of file for reading the y_data.
        source: DRP source name.
        split_id : Split id. If -1, use all data. Note that this assumes that split_id has been constructed to take into account all the data sources.
        stage: Type of split to read. One of the following: 'train', 'val', 'test'.
        y_col_name: Name of drug response measure/score (e.g., AUC, IC50)
        canc_col_name: Column name that contains the cancer sample ids. Default: "improve_sample_id".
        drug_col_name: Column name that contains the drug ids. Default: "improve_chem_id".
        sep: Separator used in data file.
        verbose: Flag for verbosity. If True, info about computations is displayed. Default: True.

    Returns:
        pd.Dataframe: dataframe that contains single drug response values.
    """
    y_data_file = inpath_dict["y_data"] / y_file_name
    if y_data_file.exists() == False:
        raise Exception(f"ERROR ! {y_file_name} file not available.\n")
    # Read y_data_file
    df = pd.read_csv(y_data_file, sep=sep)

    # Get a subset of samples if split_id is different to -1
    if split_id > -1:
        split_file_name = f"{source}_split_{split_id}_{stage}.txt"
    else:
        split_file_name = f"{source}_all.txt"
    insplit = inpath_dict["splits"] / split_file_name
    if insplit.exists() == False:
        raise Exception(f"ERROR ! {split_file_name} file not available.\n")
    ids = pd.read_csv(insplit, header=None)[0].tolist()
    df = df.loc[ids]

    df = df.reset_index(drop=True)
    if verbose:
        print(f"Data read: {y_file_name}, Filtered by: {split_file_name}")
        print(f"Shape of constructed response data framework: {df.shape}")
        print(f"Unique cells:  {df[canc_col_name].nunique()}")
        print(f"Unique drugs:  {df[drug_col_name].nunique()}")
    return df


def check_parameter_consistency(params: Dict):
    """Minimal validation over parameter set.

    Parameters
    ----------
    params: python dictionary
        A dictionary of Candle/Improve keywords and parsed values.
    """
    if params["response_file"] not in params["y_data"]:
        message = (f"ERROR ! {params['response_file']} was not listed in params['y_data']. Not guaranteed that it is available.\n")
        warnings.warn(message, RuntimeWarning)
    if params["cell_file"] not in params["x_data"]:
        message = (f"ERROR ! {params['cell_data']} was not listed in params['x_data']. Not guaranteed that it is available.\n")
        warnings.warn(message, RuntimeWarning)
    if params["drug_file"] not in params["x_data"]:
        message = (f"ERROR ! {params['drug_data']} was not listed in params['x_data']. Not guaranteed that it is available.\n")
        warnings.warn(message, RuntimeWarning)


def run(params: Dict):
    """Execute specified data preprocessing.

    Parameters
    ----------
    params: python dictionary
        A dictionary of Candle keywords and parsed values.
    """
    # --------------------------------------------
    # Check consistency of parameter specification
    # --------------------------------------------
    check_parameter_consistency(params)

    # ------------------------------------------
    # Check/Construct output directory structure
    # ------------------------------------------
    tobuildq, inpathd = csa.directory_tree_from_parameters(params, step = "pre-process")
    # inpathd is dictionary with folder_name: path components
    # Each element of the queue contains a tuple ((source, target, split_id), ipath, opath)
    print(tobuildq)

    # -------------------
    # Load cancer data
    # -------------------
    fname = inpathd["x_data"] / params["cell_file"]
    if fname.exists() == False:
        raise Exception(f"ERROR ! Cancer data from {fname} file not found.\n")
    df_x = dtl.load_cell_data(fname,
                          params["canc_col_name"],
                          params["gene_system_identifier"],
                         )

    # Use landmark genes (gene selection)
    if params["use_lincs"]:
        with open(filepath/"landmark_genes") as f: # AWFUL that this is not in data site but in code repo
            genes = [str(line.rstrip()) for line in f]
        # genes = ["ge_" + str(g) for g in genes]  # This is for legacy data
        genes = dtl.common_elements(genes, df_x.columns[1:])
        cols = [params["canc_col_name"]] + genes
        df_x = df_x[cols]

    # -------------------
    # Load drug data
    # -------------------
    # Soft coded for smiles for now
    fname = inpathd["x_data"] / params["drug_file"]
    if fname.exists() == False:
        raise Exception(f"ERROR ! Drug data from {fname} file not found.\n")
    df_x2 = dtl.load_drug_data(fname,
                           columns=["improve_chem_id", "smiles"],
                          )

    # -------------------------------------------
    # Construct ML data for every split and stage
    # -------------------------------------------
    stages = ["train", "val", "test"]
    df_x_cs = {}
    df_y_cs = {}

    while tobuildq:
        elem = tobuildq.popleft() # This is (DataSplit, ISplitPath, OSplitPath)
        for st in stages:
            print(f"Building stage: {st}")
            source = elem[0].data_source
            split_id = elem[0].split_source_index
            if st == "test": # Test data comes from target data
                source = elem[0].data_target
                split_id = elem[0].split_target_index

            # ------------------------------
            # Load y data according to stage
            # ------------------------------
            df_y = load_response_data(inpathd,
                                       params["response_file"],
                                       source,
                                       split_id,
                                       st,
                                       params["y_col_name"],
                                       params["canc_col_name"],
                                       params["drug_col_name"],
                          )
            # Retain (canc, drug) response samples for which omic data is available
            df_y_cs[st], df_x_cs[st] = dtl.get_common_samples(df1=df_y,
                                                          df2=df_x,
                                                          ref_col=params["canc_col_name"])
            print(df_y_cs[st][[params["canc_col_name"], params["drug_col_name"]]].nunique())
            # Save the subset of y data
            fname = f"{st}_{params['y_data_suffix']}.csv"
            df_y_cs[st].to_csv(elem[2] / fname, index=False)


    #load_drug_data(stage)
    #preprocess()
    #preprocess_MLmodel_specific()

    #load_cell_data()
    #preprocess()
    #preprocess_MLmodel_specific()

    #combine_data() # Filter, extract features and build combination ?
    #store_testbed() # PyTorch dataset




def main():
    params = csa.frm.initialize_parameters(filepath, default_model="csa_graphdrp_default_model.txt")
    params = csa.frm.initialize_parameters(filepath,
                                       default_model="csa_graphdrp_default_model.txt",
                                       additional_definitions = csa.csa_conf + csa_gdrp_conf + gdrp_data_conf,
                                       required = csa.req_csa_args,
                                      )

    run(params)
    print("\nFinished CSA GraphDRP pre-processing (transformed raw DRP data to model input ML data).")


if __name__ == "__main__":
    main()
