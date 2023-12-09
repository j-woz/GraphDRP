""" Functionality for inferencing with trained GraphDRP Model. """

import os
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from typing import Dict

import torch
from torch_geometric.data import DataLoader

from improve import framework as frm

# from improve.torch_utils import TestbedDataset
from improve.metrics import compute_metrics
# from models.gat import GATNet
# from models.gat_gcn import GAT_GCN
# from models.gcn import GCNNet
# from models.ginconv import GINConvNet

# Model-specific imports
from model_utils.torch_utils import (
    TestbedDataset,
    build_GraphDRP_dataloader,
    # set_GraphDRP,
    # train_epoch,
    predicting,
    load_GraphDRP,
    determine_device
)

# [Req] Imports from preprocess and train scripts
from graphdrp_preprocess_improve import preprocess_params
from graphdrp_train_improve import metrics_list, train_params

# from graphdrp_train_improve import (
#     gdrp_data_conf,
#     gdrp_model_conf,
#     determine_device,
#     build_PT_data_loader,
#     evaluate_model,
#     store_predictions_df,
#     compute_performace_scores,
# )

# from graphdrp_train_improve import (
#     metrics_list,
#     model_preproc_params,
#     model_train_params,
#     # determine_device,
#     # build_PT_data_loader,
#     # evaluate_model,
#     # store_predictions_df,
#     # compute_performace_scores,
# )

filepath = Path(__file__).resolve().parent

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_infer_params
# 2. model_infer_params
# 
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific params in this script.
app_infer_params = []

# 2. Model-specific params (Model: GraphDRP)
# All params in model_infer_params are optional.
# If no params are required by the model, then it should be an empty list.
model_infer_params = []

# [Req] Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
infer_params = app_infer_params + model_infer_params
# req_infer_params = []
# ---------------------


# def check_data_available(params: Dict) -> frm.DataPathDict:
#     """
#     Sweep the expected input paths and check that files needed in testing are available.

#     :param Dict params: Dictionary of parameters read

#     :return: Path to directories requested stored in dictionary with str key str and Path value.
#     :rtype: DataPathDict
#     """
#     # Expected
#     # test_ml_data_dir / processed / test_data_processed
#     # test_data_processed --> it has pt extension and is located inside a 'processed' folder
#     # Make sure that the test data exists
#     itestpath = Path(params["test_ml_data_dir"]) / "processed"
#     if itestpath.exists() == False:
#         raise Exception(f"ERROR ! Processed testing data folder {itestpath} not found.\n")
#     itest = itestpath / params["test_data_processed"]
#     if itest.exists() == False:
#         raise Exception(f"ERROR ! Processed testing data {itest} not found.\n")

#     # Check if testing data frame exists
#     itestdfpath = None
#     if "test_data_df" in params:
#         itestdfpath = Path(params["test_ml_data_dir"]) / params["test_data_df"]
#         if itestdfpath.exists() == False:
#             itestdfpath = None
#             message = (
#                        f"Data frame with original testing response data: {params['test_data_df']} not found." \
#                        + " Will continue but will only store partial (available) data frame.\n"
#             )
#             warnings.warn(message, RuntimeWarning)
#     else:
#         message = (
#                    f"Data frame with original testing response data not specified (no 'test_data_df' keyword)." \
#                        + " Will continue but will only store partial (available) data frame.\n"
#             )
#         warnings.warn(message, RuntimeWarning)

#     # Create output directory. Do not complain if it exists.
#     opath = Path(params["model_outdir"])
#     os.makedirs(opath, exist_ok=True)
#     modelpath = opath / params["model_params"]
#     if modelpath.exists() == False:
#         raise Exception(f"ERROR ! Trained model {modelpath} not found.\n")
#     fname = f"test_{params['model_eval_suffix']}.csv"
#     if itestdfpath is None:
#         fname = f"test_{params['model_eval_suffix']}_partial.csv"
#     predpath = opath / fname
#     fname = f"test_{params['json_scores_suffix']}.json"
#     scorespath = opath / fname

#     # Return in DataPathDict structure
#     inputdtd = {"test": itestpath, "model": modelpath, "df": itestdfpath}
#     outputdtd = {"pred": predpath, "scores": scorespath}
#     return inputdtd, outputdtd


def run(params):
    """ Run model inference.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on test data according
            to the metrics_list.
    """
    # import ipdb; ipdb.set_trace()

    # ------------------------------------------------------
    # [Req] Create output dir for the model. 
    # ------------------------------------------------------
    frm.create_outdir(outdir=params["infer_outdir"])

    # ------------------------------------------------------
    # Check data availability and create output directory
    # ------------------------------------------------------
    # [Req]
    # indtd, outdtd = check_data_available(params)
    # indtd is dictionary with input_description: path components
    # outdtd is dictionary with output_description: path components

    # ------------------------------------------------------
    # [Req] Create data names for test
    # ------------------------------------------------------
    # test_data_fname = frm.build_ml_data_name(params, stage="test",
    #                                          file_format=params["data_format"])
    test_data_fname = frm.build_ml_data_name(params, stage="test")
    # GraphDRP -- remove data_format
    test_data_fname = test_data_fname.split(params["data_format"])[0]

    # ------------------------------------------------------
    # [GraphDRP] Prepare dataloaders
    # ------------------------------------------------------
    print("\nTest data:")
    print(f"test_ml_data_dir: {params['test_ml_data_dir']}")
    print(f"test_batch: {params['test_batch']}")
    test_loader = build_GraphDRP_dataloader(params["test_ml_data_dir"],
                                            test_data_fname,
                                            params["test_batch"],
                                            shuffle=False)

    # ------------------------------------------------------
    # [Req]
    # ------------------------------------------------------
    # Determine CUDA/CPU device and configure CUDA device if available
    device = determine_device(params["cuda_name"])

    # -----------------------------
    # Load best model and compute preditions
    # -----------------------------
    # Load the (best) saved model (as determined based on val data)
    modelpath = frm.build_model_path(params, model_dir=params["model_dir"]) # [Req]
    model = load_GraphDRP(params, modelpath, device)
    model.eval()

    # Compute predictions
    test_true, test_pred = predicting(model, device, data_loader=test_loader)

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        params,
        y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    test_scores = frm.compute_performace_scores(
        params,
        y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"], metrics=metrics_list
    )

    return test_scores


# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_params + train_params + infer_params
    params = frm.initialize_parameters(
        filepath,
        # default_model="graphdrp_default_model.txt",
        # default_model="graphdrp_params.txt",
        # default_model="params_ws.txt",
        default_model="params_cs.txt",
        additional_definitions=additional_definitions,
        # required=req_infer_args,
        required=None,
    )
    test_scores = run(params)
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
