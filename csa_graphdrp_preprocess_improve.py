"""Functionality for Pre-processing in Cross-Study Analysis (CSA) for GraphDRP Model."""

from pathlib import Path
import os

from typing import Deque, Dict, List, Tuple, Union

import pandas as pd

from improve import csa
from improve import dataloader as dtl
from improve.torch_utils import TestbedDataset
from graphdrp_preprocess_improve import (
    gdrp_data_conf,
    req_preprocess_args,
    check_parameter_consistency,
    raw_data_available,
    common_data,
    load_response_data,
    compose_data_arrays,
)

filepath = Path(__file__).resolve().parent

not_used_from_model = ["data_set", "split_id"]

required_csa = list(set(csa.req_csa_args).union(set(req_preprocess_args)).difference(set(not_used_from_model)))


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
    tobuildq, inpathd = csa.directory_tree_from_parameters(params, raw_data_available, step = "preprocess")
    # inpathd is dictionary with folder_name: path components
    # Each element of the queue contains a tuple ((source, target, split_id), ipath, opath)
    print(tobuildq)

    # ------------------------------------------------------
    # Construct data frames for drug and cell features
    # ------------------------------------------------------
    # [Req]
    df_drug, df_cell_all, smile_graphs = common_data(params, inpathd)

    # -------------------------------------------
    # Construct ML data for every split and stage
    # -------------------------------------------
    stages = ["train", "val", "test"]

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
            df_response = load_response_data(inpathd,
                                       params["response_file"],
                                       source,
                                       split_id,
                                       st,
                                       params["canc_col_name"],
                                       params["drug_col_name"],
                          )
            # Retain (canc, drug) response samples for which omic data is available
            df_y, df_cell = dtl.get_common_samples(df1=df_response,
                                               df2=df_cell_all,
                                               ref_col=params["canc_col_name"])
            print(df_y[[params["canc_col_name"], params["drug_col_name"]]].nunique())

            # Normalize features using training set -> ToDo: implement this
            #if st == "train":
                # Normalize
                # Store normalization object
            #else:
                # Use previous normalization object

            # Sub-select desired response column (y_col_name)
            # And reduce response dataframe to 3 columns: drug_id, cell_id and selected drug_response
            df_y = df_y[[params["drug_col_name"], params["canc_col_name"], params["y_col_name"]]]
            # Combine data
            xd, xc, y = compose_data_arrays(df_y, df_drug, df_cell, params["drug_col_name"], params["canc_col_name"])
            print("stage ", st, "--> xd ", xd.shape, "xc ", xc.shape, "y ", y.shape)
            # Save the processed (all) data as PyTorch dataset
            TestbedDataset(root=elem[2],#outdtd["preprocess"],
                       dataset=st + "_" + params["data_suffix"],
                       xd=xd,
                       xt=xc,
                       y=y,
                       smile_graph=smile_graphs)

            # Save the subset of y data
            fname = f"{st}_{params['y_data_suffix']}.csv"
            df_y.to_csv(elem[2] / fname, index=False)


def main():
    params = csa.initialize_parameters(filepath,
                                       default_model="csa_graphdrp_default_model.txt",
                                       additional_definitions = gdrp_data_conf,
                                       required = required_csa,
                                       topop = not_used_from_model,
                                      )

    run(params)
    print("\nFinished CSA GraphDRP pre-processing (transformed raw DRP data to model input ML data).")


if __name__ == "__main__":
    main()
