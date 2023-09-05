"""Functionality for Cross-Study Analysis (CSA) in IMPROVE."""

import os
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Tuple, Union

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
    {"name": "split_id",
     "nargs": "+",
     "type": int,
     "help": "List of data samples to use for training/validation/testing.",
    },
]

frm.additional_definitions.extend(csa_conf)
req_args = [l["name"] for l in csa_conf]
frm.required.extend(req_args)


class DataSplit:
    """Define structure of information for split."""
    def __init__(self, dsource: str, dtarget: str, sindex: int, tindex: int):
        self.data_source = dsource
        self.data_target = dtarget
        self.split_source_index = sindex
        self.split_target_index = tindex


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


def directory_tree_from_parameters(
    params: Dict,
    step: str = "pre-process",
) -> Tuple[Deque, Union[frm.DataPathDict, None]]:
    """
    Check input data and construct output directory trees from parameters for cross-study analysis (CSA).

    Input and output structure depends on step.
    For pre-process step, input structure is represented by DataPathDict.
    In other steps, raw input is None and the output queue contains input and output paths.

    :param Dict params: Dictionary of parameters read
    :param string step: String to specify if this is applied during pre-process, train or test.

    :return: Paths and info about processed data output directories and raw data input.
    :rtype: (Deque, (Path, Path, Path))
    """
    inpath_dict = None
    if step == "pre-process":
        # Check that raw data is available
        inpath_dict = raw_data_available(params)
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
            if step != "pre-process" and inpath.exists() == False:
                raise Exception(f"ERROR ! '{inpath}' not found.\n")
            elif step != "test":
                os.makedirs(tagpath, exist_ok=True)

            # From this point on the depth of the path does not increase
            itagpath = inpath
            otagpath = tagpath

            if len(params["split_id"]) == 0:
                # Need a defined split id
                raise Exception(f"ERROR ! No split id has been defined.\n")
            else:
                for id in params["split_id"]:
                    index = "split_" + str(id)
                    outpath = otagpath / index
                    inpath = itagpath / index
                    if step != "pre-process" and inpath.exists() == False:
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
