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
    build_common_data,
    build_stage_dependent_data,
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
    df_drug, df_cell_all, smile_graphs = build_common_data(params, inpathd)

    # -------------------------------------------
    # Construct ML data for every split and stage
    # -------------------------------------------
    stages = ["train", "val", "test"]
    scaler = None
    while tobuildq:
        elem = tobuildq.popleft() # This is (DataSplit, ISplitPath, OSplitPath)
        for st in stages:
            print(f"Building stage: {st}")
            source = elem[0].data_source
            split_id = elem[0].split_source_index
            if st == "test": # Test data comes from target data
                source = elem[0].data_target
                split_id = elem[0].split_target_index

            outdtd = {"preprocess": elem[2]}

            scaler = build_stage_dependent_data(params,
                                   inpathd,
                                   outdtd,
                                   st,
                                   source,
                                   split_id,
                                   df_drug,
                                   df_cell_all,
                                   smile_graphs,
            )


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
