from pathlib import Path

import torch

import candle
import candle_improve_utils as improve_utils

file_path = Path(__file__).resolve().parent


# IMPROVE params that are relevant to all IMPORVE models
improve_general_params = [
    {"name": "main_data_dir",  # TODO: need to determine where this will be defined!
     "type": str,
     "default": "csa_data",
     "help": "Main data directory that contains the dataset (e.g., csa_data, lca_data, etc.)."},
    # ---------
    {"name": "download",
     "type": candle.str2bool,
     "default": False,
     "help": "Flag to indicate if downloading from FTP site."},
    # ---------
    {"name": "y_col_name",
     "type": str,
     "default": "auc",
     "help": "Column name in the y data file (e.g., response.tsv), that represents \
              the target variable that the model predicts. In drug response prediction \
              problem is can be IC50, AUC, and others."},
]

improve_preprocess_params = [
    {"name": "train_data_name",
     "type": str,
     "help": "Name of the training dataset (e.g., CCLE, GDSCv2)."},
    {"name": "val_data_name",
     "type": str,
     "help": "Name of the val dataset (e.g., CCLE, GDSCv2)."},
    {"name": "test_data_name",
     "type": str,
     "help": "Name of the test dataset (e.g., CCLE, GDSCv2)."},
    # ---------
    {"name": "train_split_file_name",
     "type": str,
     "help": "File containing integers, representing row numbers in the y data file \
              (e.g., response.tsv). These rows/samples will be used for training \
              (e.g., CCLE_split_0_train.txt, CCLE_split_0_train_size_5.txt)."},
    {"name": "val_split_file_name",
     "type": str,
     "help": "File containing row numbers in the y data file that will be used \
              val set (e.g., CCLE_split_0_val.txt, CCLE_split_0_val_size_5.txt)."},
    {"name": "test_split_file_name",
     "type": str,
     "help": "File containing row numbers in the y data file that will be used \
              test set (e.g., CCLE_split_0_test.txt, CCLE_split_0_test_size_5.txt)."},
    # ---------
    {"name": "ml_data_outdir",
     "type": str,
     "help": "Path to store the generated ML data during the preprocessing step."},
]

improve_train_params = [
    {"name": "train_ml_data_dir",
     "action": "store",
     "type": str,
     "help": "Datadir where train data is stored."},
    {"name": "val_ml_data_dir",
     "action": "store",
     "type": str,
     "help": "Datadir where val data is stored."},
    # {"name": "model_outdir",  # Note! we use candle known param ckpt_directory
    #  "action": "store",
    #  "type": str,
    #  "help": "Datadir to store trained model."},
    # ---------
    # {"name": "val_pred_file_name",
    #  "type": str,
    #  "default": "val_preds.csv",
    #  "help": "Name of file to store inference results on val data."},
    # {"name": "test_pred_file_name",
    #  "type": str,
    #  "default": "test_preds.csv",
    #  "help": "Name of file to store inference results on test data."},
    # ---------
    # {"name": "val_y_data_file_name",
    #  "type": str,
    #  "default": "val_response.csv",
    #  "help": "Name of file that conatins true y values of val set."},
    # {"name": "test_y_data_file_name",
    #  "type": str,
    #  "default": "test_response.csv",
    #  "help": "Name of file that conatins true y values of test set."},
    # ---------
    # {"name": "json_val_scores",
    #  "type": str,
    #  "default": "val_scores.json",
    #  "help": "Name of file to store val scores."},
    # {"name": "json_test_scores",
    #  "type": str,
    #  "default": "test_scores.json",
    #  "help": "Name of file to store test scores."},
    {"name": "model_params",  ## TODO (rename?): consider making this more descriptive, e.g., model_file_name
     "type": str,
     "default": "model.pt",
     "help": "Filename to store trained model."},
]

improve_infer_params = [
    {"name": "test_ml_data_dir",
     "action": "store",
     "type": str,
     "help": "Datadir where test data is stored."},
]

# Params that are specific to this model
model_specific_params = [
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

# Combine improve_params and model_specific_params into additional_definitions
additional_definitions = improve_general_params + \
    improve_preprocess_params + \
    improve_train_params + \
    improve_infer_params + \
    model_specific_params

# additional_definitions = [
#     # {"name": "pred_col_name_suffix",  # TODO (removed): this is defined in candle_improve.json
#     #  "type": str,
#     #  "default": "_pred",
#     #  "help": "Tag to add to predictions when storing the data frame."},
#     # {"name": "y_col_name",  ## improve prm
#     #  "type": str,
#     #  "default": "auc",
#     #  "help": "Drug sensitivity score to use as the target variable (e.g., IC50, AUC)."},
#     # {"name": "model_arch",  ## model prm
#     #  "default": "GINConvNet",
#     #  "choices": ["GINConvNet", "GATNet", "GAT_GCN", "GCNNet"],
#     #  "type": str,
#     #  "help": "Model architecture to run."},
#     # Preprocessing
#     # {"name": "download",  ## improve prm
#     #  "type": candle.str2bool,
#     #  "default": False,
#     #  "help": "Flag to indicate if downloading from FTP site.",},
#     # {"name": "set",  # TODO (removed): not used with our workflows (used with original GraphDRP)
#     #  "default": "mixed",
#     #  "choices": ["mixed", "cell", "drug"],
#     #  "type": str,
#     #  "help": "Validation scheme (data splitting strategy).", },
#     # {"name": "train_split",  # TODO: (renamed): rename to train_split_file_name
#     #     "nargs": "+",
#     #     "type": str,
#     #     "help": "path to the file that contains the split ids (e.g., 'split_0_tr_id',  'split_0_vl_id').", },
#     # Training / Inference
#     # {"name": "log_interval",  ## model prm
#     #  "action": "store",
#     #  "type": int,
#     #  "help": "Interval for saving o/p", },
#     # {"name": "cuda_name",  # model prm. TODO: how should we control this?
#     #  "action": "store",
#     #  "type": str,
#     #  "help": "Cuda device (e.g.: cuda:0, cuda:1."},
#     # {"name": "train_ml_data_dir",  ## improve prm
#     #  "action": "store",
#     #  "type": str,
#     #  "help": "Datadir where train data is stored."},
#     # {"name": "val_ml_data_dir",  ## improve prm
#     #  "action": "store",
#     #  "type": str,
#     #  "help": "Datadir where val data is stored."},
#     # {"name": "test_ml_data_dir",  ## improve prm
#     #  "action": "store",
#     #  "type": str,
#     #  "help": "Datadir where test data is stored."},
#     # {"name": "model_outdir",  ## improve prm
#     #  "action": "store",
#     #  "type": str,
#     #  "help": "Datadir to store trained model."},
#     # {"name": "model_params",  # TODO (rename?): why is it called model_params??
#     #  "type": str,
#     #  "default": "model.pt",
#     #  "help": "Filename to store trained model."},
#     # {"name": "pred_fname",  ## improve prm
#     #  "type": str,
#     #  "default": "test_preds.csv",
#     #  "help": "Name of file to store inference results."},
#     # {"name": "response_data",  # TODO (renamed): created one for val and one for test.
#     #  "type": str,
#     #  "default": "test_response.csv",
#     #  "help": "Name of file to store inference results."},
#     # {"name": "json_test_scores",  # TODO (renamed): created one for val and one for test.
#     #  "type": str,
#     #  "default": "test_scores.json",
#     #  "help": "Name of file to store scores."},
# ]

# TODO (C-ap): not sure these are "required". Check this!
required = [
    # "train_data",
    # "val_data",
    # "test_data",
    # "train_split",
]


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
        improve_hard_settings_file_name = "candle_improve.json"  # TODO: this may be defined somewhere else
        improve_definitions = improve_utils.parser_from_json(improve_hard_settings_file_name)
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
    gParameters = improve_utils.build_improve_paths(gParameters)  # TODO (C-ap): not sure we need this.

    return gParameters


# TODO: While the implmenetation of this func is model-specific, we may want
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
