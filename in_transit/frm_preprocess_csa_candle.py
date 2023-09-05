from pathlib import Path
import os

from typing import Deque, Dict, List, Tuple, Union

import pandas as pd

import frm

fdir = Path(__file__).resolve().parent

#req_args = ["main_data_dir", "source_data", "target_data"]
req_args = ["source_data", "target_data"]
frm.required.extend(req_args)

# level_map encodes the relationship btw the column and gene identifier system
level_map_cell_data = {"Ensembl": 0, "Entrez": 1, "Gene_Symbol": 2}

csa_conf = [
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
     "type": frm.str2bool,
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

frm.additional_definitions.extend(csa_conf)

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



def load_drug_data(inpath_dict: frm.RawDataPathDict,
                   x_file_name: str,
                   index = None,
                   columns = None,
                   sep: str = "\t",
                   verbose: bool = True) -> pd.DataFrame:
    """
    Returns dataframe with drug features read from specified file.
    Index or columns may be redefined if requested.

    Args:
        inpath_dict: Dictionary of paths and info about raw data input directories.
        x_file_name: Name of file for reading feature x_data.
        sep: Separator used in data file.
        verbose: Flag for verbosity. If True, info about computations is displayed. Default: True.

    Returns:
        pd.Dataframe: dataframe that contains drug response values.
    """
    x_data_file = inpath_dict["x_data"] / x_file_name
    if x_data_file.exists() == False:
        raise Exception(f"ERROR ! {x_file_name} file not available.\n")
    df = pd.read_csv(x_data_file, sep=sep)

    if columns is not None: # Set columns
        df.columns = columns

    if index is not None: # Set index
        df = df.set_index(index)

    if verbose:
        print(f"Data read: {x_file_name}")
        print(f"Shape of constructed drug data framework: {df.shape}")
    return df



def load_response_data(inpath_dict: frm.RawDataPathDict,
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


def set_col_names_in_multilevel_dataframe(
    df: pd.DataFrame,
    level_map: dict,
    gene_system_identifier: Union[str, List[str]] = "Gene_Symbol") -> pd.DataFrame:
    """ Util function that supports loading of the omic data files.
    Returns the input dataframe with the multi-level column names renamed as
    specified by the gene_system_identifier arg.

    Args:
        df (pd.DataFrame): omics dataframe
        level_map (dict): encodes the column level and the corresponding identifier systems
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]

    Returns:
        pd.DataFrame: the input dataframe with the specified multi-level column names
    """
    df = df.copy()

    level_names = list(level_map.keys())
    level_values = list(level_map.values())
    n_levels = len(level_names)

    if isinstance(gene_system_identifier, list) and len(gene_system_identifier) == 1:
        gene_system_identifier = gene_system_identifier[0]

    if isinstance(gene_system_identifier, str):
        if gene_system_identifier == "all":
            df.columns = df.columns.rename(level_names, level=level_values)  # assign multi-level col names
        else:
            df.columns = df.columns.get_level_values(level_map[gene_system_identifier])  # retain specific column level
    else:
        if len(gene_system_identifier) > n_levels:
            raise Exception(f"ERROR ! 'gene_system_identifier' can't contain more than {n_levels} items.\n")
        set_diff = list(set(gene_system_identifier).difference(set(level_names)))
        if len(set_diff) > 0:
            raise Exception(f"ERROR ! Passed unknown gene identifiers: {set_diff}.\n")
        kk = {i: level_map[i] for i in level_map if i in gene_system_identifier}
        df.columns = df.columns.rename(list(kk.keys()), level=kk.values())  # assign multi-level col names
        drop_levels = list(set(level_map.values()).difference(set(kk.values())))
        df = df.droplevel(level=drop_levels, axis=1)
    return df


def load_cell_data(inpath_dict: frm.RawDataPathDict,
        x_file_name: str,
        canc_col_name = "improve_sample_id",
        gene_system_identifier: Union[str, List[str]] = "Gene_Symbol",
        sep: str = "\t",
        verbose: bool = True) -> pd.DataFrame:
    """
    Returns data frame with specified cell line data.

    Args:
        inpath_dict: Dictionary of paths and info about raw data input directories.
        x_file_name: Name of file for reading the x_data.
        canc_col_name: Column name that contains the cancer sample ids. Default: "improve_sample_id".
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]
        sep: Separator used in data file.
        verbose: Flag for verbosity. If True, info about computations is displayed. Default: True.

    Returns:
        pd.DataFrame: dataframe with the cell line data.
    """
    x_data_file = inpath_dict["x_data"] / x_file_name
    if x_data_file.exists() == False:
        raise Exception(f"ERROR ! {x_file_name} file not available.\n")

    header = [i for i in range(len(level_map_cell_data))]

    df = pd.read_csv(x_data_file, sep=sep, index_col=0, header=header)
    df.index.name = canc_col_name  # assign index name
    df = set_col_names_in_multilevel_dataframe(df, level_map_cell_data, gene_system_identifier)
    df = df.reset_index()
    if verbose:
        print(f"Data read: {x_file_name}")
        print(f"Shape of constructed cell data framework: {df.shape}")
        print(f"Unique cells:  {df[canc_col_name].nunique()}")
    return df


def get_common_samples(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        ref_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Search for common data in reference column and retain only .

    Args:
        df1, df2 (pd.DataFrame): dataframes
        ref_col (str): the ref column to find the common values

    Returns:
        df1, df2 after filtering for common data.

    Example:
        Before:

        df1:
        col1	ref_col	    col3	    col4
        CCLE	ACH-000956	Drug_749	0.7153
        CCLE	ACH-000325	Drug_1326	0.9579
        CCLE	ACH-000475	Drug_490	0.213

        df2:
        ref_col     col2                col3                col4
        ACH-000956  3.5619370596224327	0.0976107966264223	4.888499735514123
        ACH-000179  5.202025844609336	3.5046203924035524	3.5058909297299574
        ACH-000325  6.016139702655253	0.6040713236688608	0.0285691521967709

        After:

        df1:
        col1	ref_col	    col3	    col4
        CCLE	ACH-000956	Drug_749	0.7153
        CCLE	ACH-000325	Drug_1326	0.9579

        df2:
        ref_col     col2                col3                col4
        ACH-000956  3.5619370596224327	0.0976107966264223	4.888499735514123
        ACH-000325  6.016139702655253	0.6040713236688608	0.0285691521967709
    """
    # Retain df1 and df2 samples with common ref_col
    common_ids = list(set(df1[ref_col]).intersection(df2[ref_col]))
    df1 = df1[df1[ref_col].isin(common_ids)].reset_index(drop=True)
    df2 = df2[df2[ref_col].isin(common_ids)].reset_index(drop=True)
    return df1, df2


def common_elements(list1: List, list2: List, verbose: bool = True) -> List:
    """
    Return list of elements that the provided lists have in common.

    Args:
        list1: One list.
        list2: Another list.
        verbose: Flag for verbosity. If True, info about computations is displayed. Default: True.

    Returns:
        List of common elements.
    """

    in_common = list(set(list1).intersection(set(list2)))
    if verbose:
        print("Elements in common count: ", len(in_common))

    return in_common
    # with open(Path(src_raw_data_dir)/"../landmark_genes") as f:



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
    tobuildq, inpathd = frm.directory_tree_from_parameters_csa(params, step = "pre-process")
    # inpathd is dictionary with folder_name: path components
    # Each element of the queue contains a tuple ((source, target, split_id), ipath, opath)
    print(tobuildq)

    # -------------------
    # Load cancer data
    # -------------------
    df_x = load_cell_data(inpathd,
                          params["cell_file"],
                          params["canc_col_name"],
                          params["gene_system_identifier"],
                         )

    # Use landmark genes (gene selection)
    if params["use_lincs"]:
        with open(fdir/"landmark_genes") as f: # AWFUL that this is not in data site but in code repo
            genes = [str(line.rstrip()) for line in f]
        # genes = ["ge_" + str(g) for g in genes]  # This is for legacy data
        genes = common_elements(genes, df_x.columns[1:])
        cols = [params["canc_col_name"]] + genes
        df_x = df_x[cols]

    # -------------------
    # Load drug data
    # -------------------
    # Soft coded for smiles for now
    df_x2 = load_drug_data(inpathd,
                           params["drug_file"],
                           columns=["improve_chem_id", "smiles"],
                          )

    # -----------------
    # Construct ML data
    # -----------------
    stages = ["train", "val", "test"]
    df_x_cs = {}
    df_y_cs = {}

    while tobuildq:
        elem = tobuildq.popleft() # This is (DataSplit, SplitPath)
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
            df_y_cs[st], df_x_cs[st] = get_common_samples(df1=df_y,
                                                          df2=df_x,
                                                          ref_col=params["canc_col_name"])
            print(df_y_cs[st][[params["canc_col_name"], params["drug_col_name"]]].nunique())
            # Save the subset of y data
            fname = f"{st}_{params['y_data_suffix']}.csv"
            df_y_cs[st].to_csv(elem[1] / fname, index=False)


    #load_drug_data(stage)
    #preprocess()
    #preprocess_MLmodel_specific()

    #load_cell_data()
    #preprocess()
    #preprocess_MLmodel_specific()

    #combine_data() # Filter, extract features and build combination ?
    #store_testbed() # PyTorch dataset




def main():
    params = frm.initialize_parameters(default_model="frm_csa_model.txt")
    run(params)
    #
    print("\nFinished pre-processing (transformed raw DRP data to model input ML data).")


if __name__ == "__main__":
    main()
