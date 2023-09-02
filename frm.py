from pathlib import Path
import os

# Check that environmental variable "IMPROVE_DATA_DIR" has been specified
if os.getenv("IMPROVE_DATA_DIR") is None:
    raise Exception(
                "ERROR ! Required system variable not specified.  You must define IMPROVE_DATA_DIR ... Exiting.\n"
            )
os.environ["CANDLE_DATA_DIR"] = os.environ["IMPROVE_DATA_DIR"]

from collections import deque
from typing import Deque, Dict, List, Tuple, Union
from typing import TypeAlias

import torch

import candle
str2bool = candle.str2bool
#import candle_improve_utils as improve_utils

file_path = Path(__file__).resolve().parent

RawDataPathDict: TypeAlias = dict[str, Path]

# IMPROVE params that are relevant to all IMPROVE models
improve_basic_conf = [
]

improve_preprocess_conf = [
    {"name": "download",
     "type": candle.str2bool,
     "default": False,
     "help": "Flag to indicate if downloading from FTP site."
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
    {"name": "y_col_name",
     "type": str,
     "default": "auc",
     "help": "Column name in the y data file (e.g., response.tsv), that represents \
              the target variable that the model predicts. In drug response prediction \
              problem it can be IC50, AUC, and others."
    },
]

# Params that are specific to this model
improve_model_conf = [
    {"name": "model_arch",
     "default": "GINConvNet",
     "choices": ["GINConvNet", "GATNet", "GAT_GCN", "GCNNet"],
     "type": str,
     "help": "Model architecture to run."},
    {"name": "log_interval",
     "action": "store",
     "type": int,
     "help": "Interval for saving o/p"},
    {"name": "cuda_name",  # TODO: how should we control this?
     "action": "store",
     "type": str,
     "help": "Cuda device (e.g.: cuda:0, cuda:1."},
]


improve_train_conf = [
    {"name": "y_data_suffix",
      "type": str,
      "default": "y_data",
      "help": "Suffix to compose file name for storing true y values."},
    {"name": "x_data_suffix",
      "type": str,
      "default": "x_data",
      "help": "Suffix to compose file name for storing features (x values)."},
    {"name": "model_params",
     "type": str,
     "default": "model.pt",
     "help": "Filename to store trained model parameters."},
    ####
    {"name": "train_ml_data_dir",
     "action": "store",
     "type": str,
     "help": "Datadir where train data is stored."},
    {"name": "val_ml_data_dir",
     "action": "store",
     "type": str,
     "help": "Datadir where val data is stored."},
    {"name": "model_eval",
     "type": str,
     "default": "test_response.csv",
     "help": "Name of file to store inference results."},
    {"name": "json_scores",
     "type": str,
     "default": "test_scores.json",
     "help": "Name of file to store scores."},
]

improve_infer_conf = [
    {"name": "test_ml_data_dir",
     "action": "store",
     "type": str,
     "help": "Datadir where test data is stored."},
]


# Combine improve configuration and model specific configuration into additional_definitions
additional_definitions = improve_basic_conf + \
    improve_preprocess_conf + \
    improve_model_conf + \
    improve_train_conf + \
    improve_infer_conf

required = [
]

class DataSplit:
    """Define structure of information for split."""
    def __init__(self, dsource: str, dtarget: str, sindex: int, tindex: int):
        self.data_source = dsource
        self.data_target = dtarget
        self.split_source_index = sindex
        self.split_target_index = tindex


def check_path_and_files(folder_name: str, file_list: List, inpath: Path) -> Path:
    """Checks if a folder and its files are available in path.

    Returns a path to the folder if it exists or raises an exception if it does
    not exist, or if not all the listed files are present.

    :param string folder_name: Name of folder to look for in path.
    :param list file_list: List of files to look for in folder
    :param inpath: Path to look into for folder and files

    :return: Path to folder requested
    :rtype: Path
    """
    outpath = inpath / folder_name
    # Check if folder is in path
    if outpath.exists():
        # Make sure that the specified files exist
        for fn in file_list:
            auxdir = outpath / fn
            if auxdir.exists() == False:
                raise Exception(f"ERROR ! {fn} file not available.\n")
    else:
        raise Exception(f"ERROR ! {folder_name} folder not available.\n")

    return outpath


def raw_data_available_csa(params: Dict) -> RawDataPathDict:
    """
    Sweep the expected raw data folder and check that files needed for cross-study analysis (CSA) are available.

    :param Dict params: Dictionary of parameters read

    :return: Path to directories requested stored in dictionary with str key str and Path value.
    :rtype: RawDataPathDict
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
    xpath = check_path_and_files("x_data", params["x_data"], inpath)
    ypath = check_path_and_files("y_data", params["y_data"], inpath)
    # Make sure that the splits exist ?
    spath = check_path_and_files("splits", [], inpath)

    return {"x_data": xpath, "y_data": ypath, "splits": spath}


def directory_tree_from_parameters_csa(
    params: Dict,
    step: str = "pre-process",
) -> Tuple[Deque, Union[RawDataPathDict, None]]:
    """
    Check input data and construct output directory trees from parameters for cross-study analysis (CSA).

    Input and output structure depends on step.
    For pre-process step, input structure is represented by RawDataPathDict.
    In other steps, raw input is None and the output queue contains input and output paths.

    :param Dict params: Dictionary of parameters read
    :param string step: String to specify if this is applied during pre-process, train or test.

    :return: Paths and info about processed data output directories and raw data input.
    :rtype: (Deque, (Path, Path, Path))
    """
    inpath_dict = None
    if step == "pre-process":
        # Check that raw data is available
        inpath_dict = raw_data_available_csa(params)
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
            raise Exception(f"ERROR ! {inpath} not found.\n")
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
                raise Exception(f"ERROR ! {inpath} not found.\n")
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
                        raise Exception(f"ERROR ! {inpath} not found.\n")
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


# -----------------------------
# CANDLE class and initialize_parameters
# Note: this is used here to load the IMPROVE hard settings from candle_imporve.json
# TODO: some of the IMPROVE hard settings are specific to the DRP problem. We may consider
#       renaming it. E.g. candle_improve_drp.json.

class BenchmarkFRM(candle.Benchmark):
    """ Benchmark for FRM. """

    def set_locals(self):
        """ Set parameters for the benchmark.

        Parameters
        ----------
        required: set of required parameters for the benchmark.
        additional_definitions: list of dictionaries describing the additional parameters for the
            benchmark.
        """
        # improve_hard_settings_file_name is a json file that contains settings
        # for IMPROVE that should not be modified by model curators/users.
        #improve_hard_settings_file_name = "candle_improve.json"  # TODO: this may be defined somewhere else
        improve_definitions = []#improve_utils.parser_from_json(improve_hard_settings_file_name)
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions + improve_definitions


def initialize_parameters(default_model="frm_default_model.txt"):
    """ Parse execution parameters from file or command line.

    Parameters
    ----------
    default_model : string
        File containing the default parameter definition.

    Returns
    -------
    gParameters: python dictionary
        A dictionary of Candle keywords and parsed values.
    """

    # Build benchmark object
    frm = BenchmarkFRM(
        filepath=file_path,
        defmodel=default_model,
        framework="python",  # TODO (Q-ap): should this be pytorch?
        prog="frm",
        desc="frm functionality",
    )

    # Initialize parameters
    # TODO (Q-ap): where are all the places that gParameters devided from?
    # This is important to specify in the docs for model curators.
    # Is it:  candle_improve.json, frm.py, frm_default_model.txt
    gParameters = candle.finalize_parameters(frm)
    #gParameters = improve_utils.build_improve_paths(gParameters)  # TODO (C-ap): not sure we need this.

    return gParameters


# TODO: While the implementation of this func is model-specific, we may want
# to require that all models have this func defined for their models. Also,
# we need to decide where this func should be located.
def predicting(model, device, loader):
    """ Method to run predictions/inference.
    This is used in *train.py and *infer.py

    Parameters
    ----------
    model : pytorch model
        Model to evaluate.
    device : string
        Identifier for hardware that will be used to evaluate model.
    loader : pytorch data loader.
        Object to load data to evaluate.

    Returns
    -------
    total_labels: numpy array
        Array with ground truth.
    total_preds: numpy array
        Array with inferred outputs.
    """
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print("Make prediction for {} samples...".format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _ = model(data)
            # Is this computationally efficient?
            total_preds = torch.cat((total_preds, output.cpu()), 0)  # preds to tensor
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)  # labels to tensor
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()
