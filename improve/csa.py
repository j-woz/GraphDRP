"""Functionality for Cross-Study Analysis (CSA) in IMPROVE."""

import os
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, Tuple, Union

import improve.framework as frm

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
    {"name": "source_data",
     "nargs": "+",
     "type": str,
     "help": "List of data sources to use for training/validation.",
    },
    {"name": "target_data",
     "nargs": "+",
     "type": str,
     "help": "List of data sources to use for testing.",
    },
    {"name": "split_ids",
     "nargs": "+",
     "type": int,
     "help": "List of data samples to use for training/validation/testing.",
    },
]

req_csa_args = [elem["name"] for elem in csa_conf]


class DataSplit:
    """Define structure of information for split."""
    def __init__(self, dsource: str, dtarget: str, sindex: int, tindex: int):
        self.data_source = dsource
        self.data_target = dtarget
        self.split_source_index = sindex
        self.split_target_index = tindex


def directory_tree_from_parameters(
    params: Dict,
    raw_data_check: Callable,
    step: str = "preprocess",
) -> Tuple[Deque, Union[frm.DataPathDict, None]]:
    """
    Check input data and construct output directory trees from parameters for cross-study analysis (CSA).

    Input and output structure depends on step.
    For preprocess step, input structure is represented by DataPathDict.
    In other steps, raw input is None and the output queue contains input and output paths.

    :param Dict params: Dictionary of parameters read
    :param Callable raw_data_check: Function that checks raw data input and returns paths to x-data/y-data/splits.
    :param string step: String to specify if this is applied during preprocess, train or test.

    :return: Paths and info about processed data output directories and raw data input.
    :rtype: (Deque, (Path, Path, Path))
    """
    inpath_dict = None
    if step == "preprocess":
        # Check that raw data is available
        inpath_dict = raw_data_check(params)
    # Create subdirectory if it does not exist
    # Structure:
    # ml_data -> {source_data-target_data} -> {split_id}
    mainpath = Path(os.environ["IMPROVE_DATA_DIR"]) # Already checked
    outpath = mainpath / "ml_data"
    os.makedirs(outpath, exist_ok=True)
    # If used during train or test structure is slightly different
    # ml_data -> models -> {source_data-target_data} -> {split_id}
    inpath = outpath
    if step == "train": # Create structured output path
        outpath = outpath / "models"
        os.makedirs(outpath, exist_ok=True)
    elif step == "test": # Check that expected input path exists
        inpath = inpath / "models"
        if inpath.exists() == False:
            raise Exception(f"ERROR ! '{inpath}' not found.\n")
        outpath = inpath
    print("Preparing to store output under: ", outpath)

    # Create queue of cross study combinations to process and check inputs
    split_queue = deque()
    for sdata in params["source_data"]:
        for tdata in params["target_data"]:
            tag = sdata + "-" + tdata
            tagpath = outpath / tag
            inpath = inpath / tag
            if step != "preprocess" and inpath.exists() == False:
                raise Exception(f"ERROR ! '{inpath}' not found.\n")
            elif step != "test":
                os.makedirs(tagpath, exist_ok=True)

            # From this point on the depth of the path does not increase
            itagpath = inpath
            otagpath = tagpath

            if len(params["split_ids"]) == 0:
                # Need defined split ids
                raise Exception(f"ERROR ! No split ids have been defined.\n")
            else:
                for id in params["split_ids"]:
                    index = "split_" + str(id)
                    outpath = otagpath / index
                    inpath = itagpath / index
                    if step != "preprocess" and inpath.exists() == False:
                        raise Exception(f"ERROR ! '{inpath}' not found.\n")
                    elif step != "test":
                        os.makedirs(outpath, exist_ok=True)

                    tid = -1 # Used to indicate all splits
                    if sdata == tdata:
                        tid = id # Need to limit to the defined split id
                    if step == "train": # Check existence of x_data and y_data
                        for stg in ["train", "val", "test"]:
                            fname = f"{stg}_{params['y_data_suffix']}.csv"
                            ydata = inpath / fname
                            if ydata.exists() == False:
                                raise Exception(f"ERROR ! Ground truth data '{ydata}' not found.\n")
                            fname = f"{stg}_{params['x_data_suffix']}.pt"
                            xdata = inpath / "processed" / fname
                            if xdata.exists() == False:
                                raise Exception(f"ERROR ! Feature data '{xdata}' not found.\n")
                    elif step == "test": # Check existence of trained model
                        trmodel = inpath / params["model_params"]
                        if trmodel.exists() == False:
                            raise Exception(f"ERROR ! Trained model '{trmodel}' not found.\n")
                    split_queue.append((DataSplit(sdata, tdata, id, tid), inpath, outpath))
    return split_queue, inpath_dict
