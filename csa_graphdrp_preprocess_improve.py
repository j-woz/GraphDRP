"""Functionality for Pre-processing in Cross-Study Analysis (CSA) for GraphDRP Model."""

from pathlib import Path
import os

from typing import Deque, Dict, List, Tuple, Union

import pandas as pd

from improve import csa
from improve import dataloader as dtl
from graphdrp_preprocess_improve import gdrp_data_conf, req_preprocess_args

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

########## Use graphdrp_preprocess_improve......


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
