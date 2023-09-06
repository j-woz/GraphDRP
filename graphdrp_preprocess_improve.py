"""Functionality for Preprocessing Data for Training a GraphDRP Model."""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from improve import framework as frm
from improve import dataloader as dtl
from improve.torch_utils import TestbedDataset
from improve.rdkit_utils import build_graph_dict_from_smiles_collection

filepath = Path(__file__).resolve().parent

gdrp_data_conf = [
    {"name": "x_data",
     "nargs": "+",
     "type": str,
     "help": "List of feature files.",
    },
    {"name": "y_data",
     "nargs": "+",
     "type": str,
     "help": "List of output files.",
    },
    {"name": "data_set",
     "type": str,
     "help": "Data set to preprocess.",
    },
    {"name": "split_id",
     "type": int,
     "default": 0,
     "help": "ID of split to read. This is used to find training/validation/testing partitions and read lists of data samples to use for preprocessing.",
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
    {"name": "canc_col_name",
     "default": "improve_sample_id",
     "type": str,
     "help": "Column name that contains the cancer sample ids.",
    },
    {"name": "drug_col_name",
     "default": "improve_chem_id",
     "type": str,
     "help": "Column name that contains the drug ids.",
    },
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

]

req_aux = [elem["name"] for elem in gdrp_data_conf]

req_preprocess_args = req_aux.extend(["y_col_name", "model_outdir"])


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


def raw_data_available(params: Dict) -> frm.DataPathDict:
    """
    Sweep the expected raw data folder and check that files needed for cross-study analysis (CSA) are available.

    :param Dict params: Dictionary of parameters read

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
    inpath = mainpath / "raw_data"
    if inpath.exists() == False:
        raise Exception(f"ERROR ! {inpath} not found.\n")
    # Make sure that the data subdirectories exist
    xpath = frm.check_path_and_files("x_data", params["x_data"], inpath)
    ypath = frm.check_path_and_files("y_data", params["y_data"], inpath)
    # Make sure that the splits exist ?
    spath = frm.check_path_and_files("splits", [], inpath)

    return {"x_data": xpath, "y_data": ypath, "splits": spath}


def check_data_available(params: Dict) -> frm.DataPathDict:
    """
    Sweep the expected input paths and check that raw data files needed for preprocessing are available.

    :param Dict params: Dictionary of parameters read

    :return: Path to directories requested stored in dictionary with str key str and Path value.
    :rtype: DataPathDict
    """
    # Check that raw data is available
    # Expected
    # raw_data -> {splits, x_data, y_data}
    ipathd = raw_data_available(params)

    # Create output directory. Do not complain if it exists.
    opath = Path(params["model_outdir"])
    os.makedirs(opath, exist_ok=True)

    # Return in DataPathDict structure
    inputdtd = {"x_data": ipathd["x_data"], "y_data": ipathd["y_data"], "splits": ipathd["splits"]}
    outputdtd = {"preprocess": opath}

    return inputdtd, outputdtd


def load_response_data(inpath_dict: frm.DataPathDict,
        y_file_name: str,
        source: str,
        split_id: int,
        stage: str,
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


def compose_data_arrays(df_response, df_drug, df_cell, drug_col_name, canc_col_name):
        """ Returns drug and cancer feature data, and response values.
        Args:
            df_response: drug response dataframe. This already has been filtered to three columns: drug_id, cell_id and drug_response.
            df_drug: drug features dataframe.
            df_cell: cell features dataframe.
            drug_col_name: Column name that contains the drug ids.
            canc_col_name: Column name that contains the cancer sample ids.

        Returns: Numpy arrays with drug features, cell features and responses
            xd, xc, y
        """
        xd = [] # To collect drug features
        xc = [] # To collect cell features
        y = [] # To collect responses
        # To collect missing or corrupted data
        # nan_rsp_list = []
        # miss_cell = []
        # miss_drug = []
        count_nan_rsp = 0
        count_miss_cell = 0
        count_miss_drug = 0
        # Convert to indices for rapid lookup
        df_drug = df_drug.set_index([drug_col_name])
        df_cell = df_drug.set_index([canc_col_name])
        for i in range(df.shape[0]):  # tuples of (drug name, cell id, response)
            if i > 0 and (i%15000 == 0):
                print(i)
            drug, cell, rsp = df_response.iloc[i, :].values.tolist()
            if np.isnan(rsp):
                # nan_rsp_list.append(rsp)
                count_nan_rsp += 1
            # If drug and cell features are available
            try: # Look for drug
                drug_features = df_drug.loc[drug]
            except KeyError: # drug not found
                # miss_drug.append(drug)
                count_miss_drug += 1
            else: # Look for cell
                try:
                    cell_features = df_cell.loc[cell]
                except KeyError: # cell not found
                    # miss_cell.append(cell)
                    count_miss_cell += 1
                else: # Both drug and cell were found
                    xd.append(drug_features[1:].values) # xd contains list of drug feature vectors
                    xc.append(cell_features[1:].values) # xc contains list of cell feature vectors
                    y.append(rsp)

        print("Number of NaN responses: ", count_nan_rsp)
        print("Number of drugs not found: ", count_miss_drug)
        print("Number of cells not found: ", count_miss_cell)

        xd, xc, y = np.asarray(xd), np.asarray(xc), np.asarray(y)

        return xd, xc, y


def run(params):
    """Execute data pre-processing for graphDRP model.

    Parameters
    ----------
    params: python dictionary
        A dictionary of Candle keywords and parsed values.
    """
    # --------------------------------------------
    # Check consistency of parameter specification
    # --------------------------------------------
    check_parameter_consistency(params)

    # ------------------------------------------------------
    # Check data availability and create output directory
    # ------------------------------------------------------
    # [Req]
    indtd, outdtd = check_data_available(params)
    # indtd is dictionary with input_description: path components
    # outdtd is dictionary with output_description: path components

    # -------------------
    # Load drug data
    # -------------------
    # Soft coded for smiles for now
    fname = indtd["x_data"] / params["drug_file"]
    if fname.exists() == False:
        raise Exception(f"ERROR ! Drug data from {fname} file not found.\n")
    df_drug = dtl.load_drug_data(fname,
                           columns=["improve_chem_id", "smiles"],
                          )

    smile_graphs = build_graphs_from_smiles_collection(df_drug["smiles"].values)

    # -------------------
    # Load cancer data
    # -------------------
    fname = indtd["x_data"] / params["cell_file"]
    if fname.exists() == False:
        raise Exception(f"ERROR ! Cancer data from {fname} file not found.\n")
    df_cell = dtl.load_cell_data(fname,
                          params["canc_col_name"],
                          params["gene_system_identifier"],
                         )

    # Use landmark genes (gene selection)
    if params["use_lincs"]:
        with open(filepath/"landmark_genes") as f: # AWFUL that this is not in data site but in code repo
            genes = [str(line.rstrip()) for line in f]
        # genes = ["ge_" + str(g) for g in genes]  # This is for legacy data
        genes = dtl.common_elements(genes, df_cell.columns[1:])
        cols = [params["canc_col_name"]] + genes
        df_cell = df_cell[cols]


    # -------------------------------------------
    # Construct ML data for every stage
    # -------------------------------------------
    stages = ["train", "val", "test"]
    df_cell_s = {}
    df_y_s = {}

    for st in stages:
        print(f"Building stage: {st}")
        source = params["data_set"]
        split_id = params["split_id"]

        # -----------------------------
        # Load y data according to stage
        # ------------------------------
        df_y = load_response_data(indtd,
                                    params["response_file"],
                                    source,
                                    split_id,
                                    st,
                                    params["canc_col_name"],
                                    params["drug_col_name"],
                          )
        # Retain (canc, drug) response samples for which omic data is available
        df_y_s[st], df_cell_s[st] = dtl.get_common_samples(df1=df_y,
                                                          df2=df_cell,
                                                          ref_col=params["canc_col_name"])
        print(df_y_s[st][[params["canc_col_name"], params["drug_col_name"]]].nunique())

    # Normalize features using training set (or training and validation sets?)


    for st in stages:
        # Sub-select desired response column (y_col_name)
        # And reduce response dataframe to 3 columns: drug_id, cell_id and selected_drug_response
        df_y = df_y_s[st][[params["drug_col_name"], params["canc_col_name"], params["y_col_name"]]]
        xd, xc, y = compose_data_arrays(df_y, df_drug, df_cell_s[st], params["drug_col_name"], params["canc_col_name"])
        print("stage ", st, "--> xd ", xd.shape, "xc ", xc.shape, "y ", y.shape)
        # Save the processed (all) data as PyTorch dataset
        TestbedDataset(root=outdtd["preprocess"],
                       dataset=st + "_" + params["x_data_suffix"],
                       xd=xd,
                       xt=xc,
                       y=y,
                       smile_graph=smile_graphs)
        # Save the subset of y data
        fname = f"{st}_{params['y_data_suffix']}.csv"
        df_y_s[st].to_csv(outdtd["preprocess"] / fname, index=False)

    #load_drug_data(stage)
    #preprocess()
    #preprocess_MLmodel_specific()

    #load_cell_data()
    #preprocess()
    #preprocess_MLmodel_specific()

    #combine_data() # Filter, extract features and build combination ?
    #store_testbed() # PyTorch dataset



def main():
    params = frm.initialize_parameters(filepath,
                                       default_model="graphdrp_default_model.txt",
                                       additional_definitions = gdrp_data_conf,
                                       required = req_preprocess_args,
                                      )
    run(params)
    print("\nFinished GraphDRP pre-processing (transformed raw DRP data to model input ML data).")


if __name__ == "__main__":
    main()

