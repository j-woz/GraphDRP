"""Functionality for Pre-processing in Cross-Study Analysis (CSA) for GraphDRP Model."""

from pathlib import Path
import os

from typing import Deque, Dict, List, Tuple, Union

import pandas as pd

from improve import csa
from improve import dataloader as dtl
from graphdrp_train_improve import gdrp_data_conf

filepath = Path(__file__).resolve().parent


def run(params: Dict):
    """Execute specified data preprocessing.

    Parameters
    ----------
    params: python dictionary
        A dictionary of Candle keywords and parsed values.
    """

########## Use graphdrp_preprocess_improve......

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
