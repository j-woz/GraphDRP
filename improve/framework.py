""" Basic Definitions of IMPROVE Framework. """
# TODO. Should we rename this script to something else?
# TODO. Questions about graphdrp_default_model.txt
# Global_Params: it contains a mix of model-specific and other args(?)
# What's the diff between [Global_Params] and [Preprocess]?

import os
import argparse

# Check that environmental variable "IMPROVE_DATA_DIR" has been specified
if os.getenv("IMPROVE_DATA_DIR") is None:
    raise Exception(
                "ERROR ! Required system variable not specified.  You must \
                define IMPROVE_DATA_DIR ... Exiting.\n"
            )
os.environ["CANDLE_DATA_DIR"] = os.environ["IMPROVE_DATA_DIR"]

from pathlib import Path
from typing import List, Set, NewType, Dict, Optional # use NewType becuase TypeAlias is available from python 3.10

# TODO! torch only used in predicting(). predicting() is moved to model_utils.py
# import torch  

SUPPRESS = argparse.SUPPRESS

import candle
str2bool = candle.str2bool
finalize_parameters = candle.finalize_parameters

# DataPathDict: TypeAlias = dict[str, Path]
DataPathDict = NewType("DataPathDict", Dict[str, Path])

# ---------------------------------------------------------------------
# Define global variables that are accessible to all models
import types
imp_glob = types.SimpleNamespace()

# imp_glob.main_data_dir = Path.joinpath(fdir, "csa_data")
imp_glob.MAIN_DATA_DIR = os.environ["IMPROVE_DATA_DIR"]

# Dir names corresponding to the primary input/output blocks in the pipeline
# {}: input/output
# []: process
# train path:      {raw_data} --> [preprocess] --> {ml_data} --> [train] --> {models}
# inference path:  {ml_data, models} --> [inference] --> {infer}
imp_glob.RAW_DATA_DIR_NAME = "raw_data"  # benchmark data
# imp_glob.ml_data_dir_name = "ml_data"    # preprocessed data for a specific ML model
# imp_glob.models_dir_name = "models"      # output from model training
# imp_glob.infer_dir_name = "infer"        # output from model inference (testing)

# Secondary dirs in raw_data
imp_glob.X_DATA_DIR_NAME = "x_data"      # feature data
imp_glob.Y_DATA_DIR_NAME = "y_data"      # target data
imp_glob.SPLITS_DIR_NAME = "splits"      # splits files

# Globals derived from the ones defined above
imp_glob.RAW_DATA_DIR = Path(imp_glob.MAIN_DATA_DIR, imp_glob.RAW_DATA_DIR_NAME)  # raw_data
# imp_glob.ml_data_dir = Path(imp_glob.main_data_dir / imp_glob.ml_data_dir_name)  # ml_data
# imp_glob.models_dir = Path(imp_glob.main_data_dir / imp_glob.models_dir_name)   # models
# imp_glob.infer_dir = Path(imp_glob.main_data_dir / imp_glob.infer_dir_name)    # infer
# -----
imp_glob.X_DATA_DIR = Path(imp_glob.RAW_DATA_DIR, imp_glob.X_DATA_DIR_NAME)    # x_data
imp_glob.Y_DATA_DIR = Path(imp_glob.RAW_DATA_DIR, imp_glob.Y_DATA_DIR_NAME)    # y_data
imp_glob.SPLITS_DIR = Path(imp_glob.RAW_DATA_DIR, imp_glob.SPLITS_DIR_NAME)    # splits
# ---------------------------------------------------------------------

# Parameters that are relevant to all IMPROVE models
improve_basic_conf = [
    {"name": "y_col_name",
     "type": str,
     "default": "auc",
     "help": "Column name in the y data file (e.g., response.tsv), that \
             represents the target variable that the model learns to predict. \
             In drug response prediction problem it can be IC50, AUC, and others."
    },
    {"name": "model_outdir",
     "type": str,
     "default": "./out/", 
     "help": "Path to save model results.",  # TODO. check "help".
    },
    #
    # ... TODO. GLOBAL VARS (start)
    {"name": "raw_data_dir",
     "type": str,
     "default": "raw_data",
     "help": "Data dir name that stores the raw data, including x data, y data, and splits.",
    },
    # ...
    {"name": "x_data_dir",
     "type": str,
     "default": "x_data",
     "help": "Data dir name that stores feature data files (i.e., x data).",
    },
    {"name": "y_data_dir",
     "type": str,
     "default": "y_data",
     "help": "Data dir name that stores target data files (i.e., y data).",
    },
    {"name": "splits_dir",
     "type": str,
     "default": "splits",
     "help": "Dir name that stores files that contain split integer ids.",
    },
    {"name": "pred_col_name_suffix",
     "type": str,
     "default": "_pred",
     "help": "Suffix to add to a column name in the y data file to identify \
             predictions made by the model (e.g., if y_col_name is 'auc', then \
             a new column that stores model predictions will be added to the y \
             data file and will be called 'auc_pred')."
    },
    # ... TODO. GLOBAL VARS (end)
    #
    # {"name": "split_id",
    #  "type": int,
    #  "default": 0,
    #  "help": "ID of split to read. This is used to find training/validation/testing \
    #          partitions and read a list of data samples to use for preprocessing.",
    # },
    # {"name": "cell_file", # TODO. Should this be list?
    #  "type": str,
    #  "default": "cancer_gene_expression.tsv",
    #  "help": "File with cancer feature data",
    # },
    # {"name": "drug_file", # TODO. Should this be list?
    #  "type": str,
    #  "default": "drug_SMILES.tsv",
    #  "help": "File with drug feature data",
    # },
    # 
    # ... TODO. DRP-specific (start)
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
    # ... TODO. DRP-specific (end)
    # 
    {"name": "x_data_files", # TODO. Should this be list?
     "type": str,
     "default": "cancer_gene_expression.tsv",
     "help": "File with cancer feature data",
    },
]

# Parameters that are relevant to all IMPROVE preprocessing scripts
improve_preprocess_conf = [
    {"name": "download",
     "type": candle.str2bool,
     "default": False,
     "help": "Flag to indicate if downloading from FTP site."
    },
    # ... TODO. Added
    {"name": "train_split_file_name",
     # "default": ,
     "type": str,
     "nargs": "+",
     "required": True,
     "help": "The path to the file that contains the split ids (e.g., 'split_0_tr_id',  'split_0_vl_id', 'split_0_tr_size_1024').",
    },
    {"name": "val_split_file_name",
     # "default": ,
     "type": str,
     "nargs": "+",
     "required": True,
     "help": "The path to the file that contains the split ids (e.g., 'split_0_tr_id',  'split_0_vl_id').",
    },
    {"name": "test_split_file_name",
     # "default": ,
     "type": str,
     "nargs": "+",
     "required": True,
     "help": "The path to the file that contains the split ids (e.g., 'split_0_tr_id',  'split_0_vl_id', 'split_0_tr_size_1024').",
    },
]

# Parameters that are relevant to all IMPROVE training scripts
# TODO. Defaults for some of these args are specified in graphdrp_default_model.txt
# Which file takes precedence (graphdrp_default_model.txt and defs below)?
improve_train_conf = [
    {"name": "y_data_suffix",
      "type": str,
      "default": "y_data",
      "help": "Suffix to compose file name for storing true y values."},
    {"name": "data_suffix",
      "type": str,
      "default": "data",
      "help": "Suffix to compose file name for storing features (x values)."},
    {"name": "model_params", # TODO. Consider renaming (e.g., model_file)
     "type": str,
     "default": "model.pt",
     "help": "Filename to store trained model parameters."},
    {"name": "train_ml_data_dir",
     "action": "store",
     "type": str,
     "help": "Datadir where train data is stored."},
    {"name": "val_ml_data_dir",
     "action": "store",
     "type": str,
     "help": "Datadir where val data is stored."},
    {"name": "train_data_processed", # TODO. Is this like ml_data?
     "action": "store",
     "type": str,
     "help": "Name of pytorch processed train data file."},
    {"name": "val_data_processed",
     "action": "store",
     "type": str,
     "help": "Name of pytorch processed val data file."},
    {"name": "model_eval_suffix",
     "type": str,
     "default": "predicted", # TODO. rename?
     "help": "Suffix to use for name of file to store inference results."},
    {"name": "json_scores_suffix",
     "type": str,
     "default": "scores",
     "help": "Suffix to use for name of file to store scores."},
    {"name": "val_batch",
     "type": int,
     "default": argparse.SUPPRESS,
     "help": "Validation batch size.",
    },
    {"name": "patience",
     "type": int,
     "default": argparse.SUPPRESS,
     "help": "Number of epochs with no improvement after which training will \
             be stopped.",
    },
]

# Parameters that are relevant to all IMPROVE testing scripts
improve_test_conf = [
    {"name": "test_ml_data_dir",
     "action": "store",
     "type": str,
     "help": "Datadir where test data is stored."
    },
    {"name": "test_data_processed",
     "action": "store",
     "type": str,
     "help": "Name of pytorch processed test data file."
    },
    {"name": "test_batch",
     "type": int,
     "default": argparse.SUPPRESS,
     "help": "Test batch size.",
    },

]


# Combine improve configuration into additional_definitions
# TODO. Consider renaming (e.g., improve_additional_definition)
frm_additional_definitions = improve_basic_conf + \
    improve_preprocess_conf + \
    improve_train_conf + \
    improve_test_conf

# Required
frm_required = []


def parser_from_json(json_file):
    """ Custom parser to read a json file and return the list of included keywords.
        Special case for True/False since these are not handled correctly by the default
        python command line parser.
        All keywords defined in json files are subsequently available to be overwritten
        from the command line, using the CANDLE command line parser.
    Parameters
    ----------
    json_file: File to be parsed
    ...
    Return
    ----------
    new_defs: Dictionary of parameters
    """
    file = open(json_file,)
    params = json.load(file)
    new_defs = []
    for key in params:
        if params[key][0] == "True" or params[key][0] == "False":
            new_def = {'name': key,
                       'type': (type(candle.str2bool(params[key][0]))),
                       'default': candle.str2bool(params[key][0]),
                       'help': params[key][1]
                       }
        else:
            new_def = {'name': key,
                       'type': (type(params[key][0])),
                       'default': params[key][0],
                       'help': params[key][1]
                       }
        new_defs.append(new_def)
    return new_defs


def build_paths(params):
    """ Build paths for raw_data, x_data, y_data. """
    mainpath = Path(os.environ["IMPROVE_DATA_DIR"])
    if mainpath.exists() == False:
        raise Exception(f"ERROR ! {mainpath} not found.\n")

    raw_data_path = mainpath / params["raw_data_dir"]
    params["raw_data_path"] =  raw_data_path
    if raw_data_path.exists() == False:
        raise Exception(f"ERROR ! {raw_data_path} not found.\n")

    x_data_path = raw_data_path / params["x_data_dir"]
    params["x_data_path"] =  x_data_path
    if x_data_path.exists() == False:
        raise Exception(f"ERROR ! {x_data_path} not found.\n")

    y_data_path = raw_data_path / params["y_data_dir"]
    params["y_data_path"] =  y_data_path
    if y_data_path.exists() == False:
        raise Exception(f"ERROR ! {y_data_path} not found.\n")

    splits_path = raw_data_path / params["splits_dir"]
    params["splits_path"] =  splits_path
    if splits_path.exists() == False:
        raise Exception(f"ERROR ! {splits_path} not found.\n")

    _path = raw_data_path / params["splits_dir"]
    params["splits_path"] =  splits_path
    if splits_path.exists() == False:
        raise Exception(f"ERROR ! {splits_path} not found.\n")
    return params


class ImproveBenchmark(candle.Benchmark):
    """ Benchmark for Improve Models. """

    def set_locals(self):
        """ Set parameters for the benchmark.

        Parameters
        ----------
        required: set of required parameters for the benchmark.
        additional_definitions: list of dictionaries describing the additional
            parameters for the benchmark.
        """
        if frm_required is not None:
            self.required.update(set(frm_required)) # This considers global framework required arguments
        if frm_additional_definitions is not None:
            self.additional_definitions.extend(frm_additional_definitions) # This considers global framework definitions


# # TODO. previous version.
# class ImproveBenchmark(candle.Benchmark):
#     def set_locals(self):
#         """ Functionality to set variables specific for the benchmark
#         - required: set of required parameters for the benchmark.
#         - additional_definitions: list of dictionaries describing the additional parameters for the
#         benchmark.
#         """
#         # improve_hard_settings_file_name is a json file that contains settings
#         # for IMPROVE that should not be modified by model curators/users.
#         print('Additional definitions built from json files')
#         improve_hard_settings_file_name = "candle_improve.json"  # TODO: this may be defined somewhere else
#         additional_definitions = parser_from_json(improve_hard_settings_file_name)
#         print(additional_definitions, flush=True)
#         if required is not None:
#             self.required = set(required)
#         if additional_definitions is not None:
#             self.additional_definitions = additional_definitions


def initialize_parameters(filepath,
                          default_model: str="frm_default_model.txt",
                          additional_definitions: Optional[List]=None,
                          required: Optional[List]=None):
    """ Parse execution parameters from file or command line.

    Parameters
    ----------
    default_model : string
        File containing the default parameter definition.
    additional_definitions : List
        List of additional definitions from calling script.
    required: Set
        Required arguments from calling script.

    Returns
    -------
    gParameters: python dictionary
        A dictionary of Candle keywords and parsed values.
    """

    # Build benchmark object
    # import pdb; pdb.set_trace()
    frm = ImproveBenchmark(
        filepath=filepath,
        defmodel=default_model,
        framework="pytorch",
        prog="frm",
        desc="Framework functionality in improve",
        additional_definitions=additional_definitions,
        required=required,
    )

    gParameters = candle.finalize_parameters(frm)

    return gParameters


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


# TODO! While the implementation of this func is model-specific, we want to
# require that all models have this func defined for their models.
# This func is moved to model_utils.py.
# def predicting(model, device, loader):
#     """ Method to run predictions/inference.
#     This is used in *train.py and *infer.py

#     Parameters
#     ----------
#     model : pytorch model
#         Model to evaluate.
#     device : string
#         Identifier for hardware that will be used to evaluate model.
#     loader : pytorch data loader.
#         Object to load data to evaluate.

#     Returns
#     -------
#     total_labels: numpy array
#         Array with ground truth.
#     total_preds: numpy array
#         Array with inferred outputs.
#     """
#     model.eval()
#     total_preds = torch.Tensor()
#     total_labels = torch.Tensor()
#     print("Make prediction for {} samples...".format(len(loader.dataset)))
#     with torch.no_grad():
#         for data in loader:
#             data = data.to(device)
#             output, _ = model(data)
#             # Is this computationally efficient?
#             total_preds = torch.cat((total_preds, output.cpu()), 0)  # preds to tensor
#             total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)  # labels to tensor
#     return total_labels.numpy().flatten(), total_preds.numpy().flatten()
