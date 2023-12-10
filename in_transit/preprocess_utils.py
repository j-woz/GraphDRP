#

def load_single_drug_response_data_v2(
        # source: Union[str, List[str]],
        source: str,
        # split: Union[int, None] = None,
        # split_type: Union[str, List[str], None] = None,
        split_file_name: Union[str, List[str], None] = None,
        y_col_name: str = "auc",
        sep: str = "\t",
        verbose: bool = True) -> pd.DataFrame:
    """
    Returns datarame with cancer ids, drug ids, and drug response values. Samples
    from the original drug response file are filtered based on the specified
    sources.

    Args:
        source (str or list of str): DRP source name (str) or multiple sources (list of strings)
        split(int or None): split id (int), None (load all samples)
        split_type (str or None): one of the following: 'train', 'val', 'test'
        y_col_name (str): name of drug response measure/score (e.g., AUC, IC50)

    Returns:
        pd.Dataframe: dataframe that contains drug response values
    """
    # TODO: currently, this func implements loading a single data source (CCLE or CTRPv2 or ...)
    df = pd.read_csv(improve_globals.y_file_path, sep=sep)

    # Get a subset of samples
    if isinstance(split_file_name, list) and len(split_file_name) == 0:
        raise ValueError("Empty list is passed via split_file_name.")
    if isinstance(split_file_name, str):
        split_file_name = [split_file_name]
    ids = load_split_ids(split_file_name)
    df = df.loc[ids]
    # else:
    #     # Get the full dataset for a given source
    #     df = df[df[improve_globals.source_col_name].isin([source])]

    # # Get a subset of cols
    # cols = [improve_globals.source_col_name,
    #         improve_globals.drug_col_name,
    #         improve_globals.canc_col_name,
    #         y_col_name]
    # df = df[cols]  # [source, drug id, cancer id, response]

    df = df.reset_index(drop=True)
    if verbose:
        print(f"Response data: {df.shape}")
        print(f"Unique cells:  {df[improve_globals.canc_col_name].nunique()}")
        print(f"Unique drugs:  {df[improve_globals.drug_col_name].nunique()}")
    return df


def load_gene_expression_data(
        gene_system_identifier: Union[str, List[str]] = "Gene_Symbol",
        sep: str = "\t",
        verbose: bool = True) -> pd.DataFrame:
    """
    Returns gene expression data.

    Args:
        gene_system_identifier (str or list of str): gene identifier system to use
            options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                     combination of ["Entrez", "Gene_Symbol", "Ensembl"]

    Returns:
        pd.DataFrame: dataframe with the omic data
    """
    # level_map encodes the relationship btw the column and gene identifier system
    level_map = {"Ensembl": 0, "Entrez": 1, "Gene_Symbol": 2}
    header = [i for i in range(len(level_map))]

    df = pd.read_csv(improve_globals.gene_expression_file_path, sep=sep, index_col=0, header=header)

    df.index.name = improve_globals.canc_col_name  # assign index name
    df = set_col_names_in_multilevel_dataframe(df, level_map, gene_system_identifier)
    if verbose:
        print(f"Gene expression data: {df.shape}")
    return df


def get_common_samples(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        ref_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Args:
        df1, df2 (pd.DataFrame): dataframes
        ref_col (str): the ref column to find the common values

    Returns:
        df1, df2

    Example:
        TODO
    """
    # Retain (canc, drug) response samples for which we have omic data
    common_ids = list(set(df1[ref_col]).intersection(df2[ref_col]))
    # print(df1.shape)
    df1 = df1[df1[improve_globals.canc_col_name].isin(common_ids)].reset_index(drop=True)
    # print(df1.shape)
    # print(df2.shape)
    df2 = df2[df2[improve_globals.canc_col_name].isin(common_ids)].reset_index(drop=True)
    # print(df2.shape)
    return df1, df2


def load_smiles_data(
        sep: str = "\t",
        verbose: bool = True) -> pd.DataFrame:
    """
    IMPROVE-specific func.
    Read smiles data.
    src_raw_data_dir : data dir where the raw DRP data is stored
    """
    df = pd.read_csv(improve_globals.smiles_file_path, sep=sep)

    # TODO: updated this after we update the data
    df.columns = ["improve_chem_id", "smiles"]

    if verbose:
        print(f"SMILES data: {df.shape}")
        # print(df.dtypes)
        # print(df.dtypes.value_counts())
    return df
