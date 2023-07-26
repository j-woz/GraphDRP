import json
import os
import sys
import pandas as pd
from pprint import pprint
from pathlib import Path
# from pathlib import PurePath
from scipy.stats.mstats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Union

import candle


file_path = os.path.dirname(os.path.realpath(__file__))


required = []


def remove_suffix(text, suffix):
    if suffix and text.endswith(suffix):
        return text[:-len(suffix)]
    return text


def parser_from_json(json_file):
    """ Custom parser to read a json file and return the list of included keywords.
        Special case for True/False since these are not handled correctly by the default
        python command line parser.
        All keywords defined in json files are subsequently available to be overwritten
        from the command line, using the CANDLE command line parser.
    Parameters
    ----------
    json_file: File to be parsed

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


def construct_improve_dir_path(dir_name, dir_path, params):
    """ Custom function to construct directory paths in IMPROVE
    """
    new_key = dir_name
    old_key = dir_name + '_name'
    new_val = dir_path + '/' + params[old_key]
    print("Appending key:", new_key, new_val)

    params[new_key] = Path(new_val)

    return params


def construct_improve_file_path(file_name, dir_path, suffix, new_suffix, value, params):
    """ Custom function to construct file paths in IMPROVE
        Given a dictionary and a key name, remove the suffix
        and generate a new key with a new suffix appended.
    """
    file = remove_suffix(file_name, suffix)
    file = file + new_suffix
    params[file] = Path(dir_path / value)

    return params


def add_improve_key(key_name, old_suffix, new_suffix, params):
    """ Custom function to construct file paths in IMPROVE
        Given a dictionary and a key name, remove the suffix
        and generate a new key with a new suffix appended.
    """
    new_key = remove_suffix(key_name, old_suffix)
    new_key = new_key + new_suffix
    params[new_key] = params[key_name]

    return params


def build_improve_paths(params):

    # special cases -- no point automating
    params = construct_improve_dir_path("raw_data_dir", "main_data_dir", params)
    params = construct_improve_dir_path("ml_data_dir", "main_data_dir", params)
    params = construct_improve_dir_path("models_dir", "main_data_dir", params)
    params = construct_improve_dir_path("infer_dir", "main_data_dir", params)

    params = construct_improve_dir_path("x_data_dir", "raw_data_dir", params)
    params = construct_improve_dir_path("y_data_dir", "raw_data_dir", params)
    params = construct_improve_dir_path("splits_dir", "raw_data_dir", params)

    dir_path = params["x_data_dir"]
    # loop over cancer features
    new_dict = {}
    for k in params:
        if k.endswith('_fname'):
            # <k>_file_path = dir_path + '/' + <k>-'_fname'
            new_dict = construct_improve_file_path(k, dir_path, '_fname', '_file_path',
                                                   params[k], new_dict)
    # loop over drug features
    for k in params:
        if k.endswith('_file_name'):
            # <k>_file_path = dir_path + '/' + <k>-'_flle_name'
            new_dict = construct_improve_file_path(k, dir_path, '_file_name', '_file_path',
                                                   params[k], new_dict)

    params.update(new_dict)

    return params

# -----------------------------
# General utilities


def str2Class(str):
    return getattr(sys.modules[__name__], str)

# -----------------------------
# Metrics


def compute_metrics(y_true, y_pred, metrics):
    """Compute the specified set of metrics.

    Parameters
    ----------
    y_true : numpy array
        True values to predict.
    y_pred : numpy array
        Prediction made by the model.
    metrics: python list
        List of metrics to compute.

    Returns
    -------
    eval: python dictionary
        A dictionary of evaluated metrics.
    """
    eval = {}
    for mtstr in metrics:
        mapstr = mtstr
        if mapstr == "pcc":
            mapstr = "pearson"
        elif mapstr == "scc":
            mapstr = "spearman"
        elif mapstr == "r2":
            mapstr = "r_square"
        eval[mtstr] = str2Class(mapstr)(y_true, y_pred)

    return eval


def mse(y_true, y_pred):
    """Compute Mean Squared Error (MSE).

    Parameters
    ----------
    y_true : numpy array
        True values to predict.
    y_pred : numpy array
        Prediction made by the model.

    Returns
    -------
        float value corresponding to MSE. If several outputs, errors of all outputs are averaged with uniform weight.
    """
    mse = mean_squared_error(y_true, y_pred)
    return mse


def rmse(y_true, y_pred):
    """Compute Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : numpy array
        True values to predict.
    y_pred : numpy array
        Prediction made by the model.

    Returns
    -------
        float value corresponding to RMSE. If several outputs, errors of all outputs are averaged with uniform weight.
    """
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return rmse


def pearson(y_true, y_pred):
    """Compute Pearson Correlation Coefficient (PCC).

    Parameters
    ----------
    y_true : numpy array
        True values to predict.
    y_pred : numpy array
        Prediction made by the model.

    Returns
    -------
        float value corresponding to PCC.
    """
    pcc = pearsonr(y_true, y_pred)[0]
    return pcc


def spearman(y_true, y_pred):
    """Compute Spearman Correlation Coefficient (SCC).

    Parameters
    ----------
    y_true : numpy array
        True values to predict.
    y_pred : numpy array
        Prediction made by the model.

    Returns
    -------
        float value corresponding to SCC.
    """
    scc = spearmanr(y_true, y_pred)[0]
    return scc


def r_square(y_true, y_pred):
    """Compute R2 Coefficient.

    Parameters
    ----------
    y_true : numpy array
        True values to predict.
    y_pred : numpy array
        Prediction made by the model.

    Returns
    -------
        float value corresponding to R2. If several outputs, scores of all outputs are averaged with uniform weight.
    """

    return r2_score(y_true, y_pred)

# -----------------------------
# Save Functions


def save_preds(df: pd.DataFrame, params: Dict, outpath: Union[str, Path], round_decimals: int = 4) -> None:
    """ Save model predictions.
    This function throws errors if the dataframe does not include the expected
    columns: canc_col_name, drug_col_name, y_col_name, y_col_name + "_pred"

    Args:
        df (pd.DataFrame): df with model predictions
        params: dictionary with run configuration
        outpath (str or PosixPath): outdir to save the model predictions df
        round (int): round response values

    Returns:
        None
    """
    # Check that the 4 columns exist
    y_col_name = params["y_col_name"]
    assert params["canc_col_name"] in df.columns, f"{params['canc_col_name']} was not found in columns."
    assert params["drug_col_name"] in df.columns, f"{params['drug_col_name']} was not found in columns."
    assert y_col_name in df.columns, f"{y_col_name} was not found in columns."
    pred_col_name = y_col_name + f"{params['pred_col_name_suffix']}"
    assert pred_col_name in df.columns, f"{pred_col_name} was not found in columns."

    # Round
    df = df.round({y_col_name: round_decimals, pred_col_name: round_decimals})

    # Save preds df
    # df.to_csv(outpath, index=False)
    df.to_csv(Path(outpath), index=False)
    return None


class ImproveBenchmark(candle.Benchmark):

    def set_locals(self):
        """ Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        print('Additional definitions built from json files')
        additional_definitions = parser_from_json("candle_improve.json")
        print(additional_definitions, flush=True)
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


def initialize_parameters():
    # Build agent object

    driver = ImproveBenchmark(file_path, 'dummy.txt', 'keras',
                              prog='CANDLE_example', desc='CANDLE example driver script')

    # Initialize parameters
    gParameters = candle.finalize_parameters(driver)
    # benchmark.logger.info('Params: {}'.format(gParameters))
    run_params = gParameters

    return run_params


def main():
    print("Running main")
    params = initialize_parameters()
    params = build_improve_paths(params)
    print("After building paths")
    pprint(params)


if __name__ == "__main__":
    main()
