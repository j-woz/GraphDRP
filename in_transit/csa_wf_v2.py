cuda_name = "cuda:5"

import os
import warnings
from pathlib import Path
import pandas as pd

import improve_utils
from improve_utils import improve_globals as ig

# import frm_preprocess
import frm_preprocess_tr_vl_te
import frm_train
import frm_infer

from ap_utils.classlogger import Logger
from ap_utils.utils import get_print_func, Timer


fdir = Path(__file__).resolve().parent


# AP
lg = Logger(ig.main_data_dir/"csa_full.log")
# print_fn = print
print_fn = get_print_func(lg.logger)
print_fn(f"File path: {fdir}")

### Source and target data sources
## Set 1 - full analysis
# source_datasets = ["CCLE", "CTRPv2", "gCSI", "GDSCv1", "GDSCv2"]
# target_datasets = ["CCLE", "CTRPv2", "gCSI", "GDSCv1", "GDSCv2"]
## Set 2 - full analysis for CCLE as source
source_datasets = ["CCLE"]
# target_datasets = ["CCLE", "CTRPv2", "gCSI", "GDSCv1", "GDSCv2"]
target_datasets = ["CCLE", "gCSI", "GDSCv1", "GDSCv2"]
# target_datasets = ["gCSI", "GDSCv1", "GDSCv2"]
## Set 3 - only CCLE and source and target
# source_datasets = ["CCLE"]
# target_datasets = ["CCLE"]

## Specifies which splits to use
# split_nums = []
split_nums = [4, 7]
# split_nums = [0]

## Parameters of the experiment/run/workflow
# TODO: this should be stored as the experiment metadata that we can go back check
# epochs = 1
# epochs = 2
# epochs = 10
epochs = 50
# epochs = 100
# epochs = 200
y_col_name = "auc"  # TODO: we put this a file that specifies improve_drp param
# y_col_name = "auc1"
config_file_name = "csa_params.txt"
config_file_path = fdir/config_file_name
model_arch = 0    # TODO: we put this a file that specifies the model specific param


# ===============================================================
###  Generate CSA results (within- and cross-study)
# ===============================================================

timer = Timer()
import pdb; pdb.set_trace()
# Iterate over source datasets
# Note! The "source_data_name" iterations are independent of each other
for source_data_name in source_datasets:

    # Get the split file paths
    # This parsing assumes splits file names are: SOURCE_split_NUM_[train/val/test].txt
    if len(split_nums) == 0:
        # Get all splits
        split_files = list((ig.splits_dir).glob(f"{source_data_name}_split_*.txt"))
        split_nums = [str(s).split("split_")[1].split("_")[0] for s in split_files]
        split_nums = sorted(set(split_nums))
        # num_splits = 1
    else:
        # Use the specified splits
        split_files = []
        for s in split_nums:
            split_files.extend(list((ig.splits_dir).glob(f"{source_data_name}_split_{s}_*.txt")))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    files_joined = [str(s) for s in split_files]

    # --------------------
    # Preprocess and Train
    # --------------------
    for split in split_nums:
        print_fn(f"Split {int(split) + 1} (id {split}) out of {len(split_nums)} splits.")
        # Check that train, val, and test are available. Otherwise, continue to the next split.
        # split = 11
        # files_joined = [str(s) for s in split_files]
        phase = "train"
        fname = f"{source_data_name}_split_{split}_{phase}.txt"
        if fname not in "\t".join(files_joined):
            warnings.warn(f"\nThe {phase} split file {fname} is missing (continue to next split)")
            continue
        phase = "val"
        fname = f"{source_data_name}_split_{split}_{phase}.txt"
        if fname not in "\t".join(files_joined):
            warnings.warn(f"\nThe {phase} split file {fname} is missing (continue to next split)")
            continue
        phase = "test"
        fname = f"{source_data_name}_split_{split}_{phase}.txt"
        if fname not in "\t".join(files_joined):
            warnings.warn(f"\nThe {phase} split file {fname} is missing (continue to next split)")
            continue

        # Iterate over target datasets
        for target_data_name in target_datasets:
            print_fn(f"\nSource data: {source_data_name}")
            print_fn(f"Target data: {target_data_name}\n")

            EXP_ML_DATA_DIR = ig.ml_data_dir/f"{source_data_name}-{target_data_name}"/f"split_{split}"

            if source_data_name == target_data_name:
                # If source and target are the same, then infer on the test split
                test_split_file_name = f"{source_data_name}_split_{split}_test.txt"
            else:
                # If source and target are different, then infer on the entire target dataset
                test_split_file_name = f"{target_data_name}_all.txt"

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # p1 (none): Preprocess train data
            # train_split_files = list((ig.splits_dir).glob(f"{source_data_name}_split_0_train*.txt"))  # TODO: placeholder for lc analysis
            # import pdb; pdb.set_trace()
            timer_preprocess = Timer()
            # frm_preprocess.main([
            ml_data_path = frm_preprocess_tr_vl_te.main([
                "--train_data_name", str(source_data_name),
                "--val_data_name", str(source_data_name),
                "--test_data_name", str(target_data_name),
                "--train_split_file_name", f"{source_data_name}_split_{split}_train.txt",
                "--val_split_file_name", f"{source_data_name}_split_{split}_val.txt",
                "--test_split_file_name", str(test_split_file_name),
                "--y_col_name", y_col_name,
                "--outdir", str(EXP_ML_DATA_DIR)
            ])
            timer_preprocess.display_timer(print_fn)

            # Train
            # Train a single model for a given [source, split] pair
            MODEL_OUTDIR = ig.models_dir/f"{source_data_name}"/f"split_{split}"
            if MODEL_OUTDIR.exists() is False:
                # p2 (p1): Train model
                # Train using train samples and early stop using val samples
                TRAIN_ML_DATA_DIR = EXP_ML_DATA_DIR
                VAL_ML_DATA_DIR = EXP_ML_DATA_DIR
                # MODEL_OUTDIR = ig.models_dir/f"{source_data_name}-{target_data_name}"/f"split_{split}"
                # MODEL_OUTDIR = ig.models_dir/f"{source_data_name}"/f"split_{split}"
                # import pdb; pdb.set_trace()
                timer_train = Timer()
                frm_train.main([
                    # "--config_file", str(config_file_path),  # TODO: we should be able to pass the config_file
                    "--epochs", str(epochs),  # available in config_file
                    "--train_ml_data_dir", str(TRAIN_ML_DATA_DIR),
                    "--val_ml_data_dir", str(VAL_ML_DATA_DIR),
                    "--model_outdir", str(MODEL_OUTDIR),
                    # "--ckpt_directory", str(MODEL_OUTDIR),  # TODO: we'll use candle known param ckpt_directory instead of model_outdir
                    "--model_arch", str(model_arch),  # specific to GraphDRP
                    "--y_col_name", y_col_name,
                    "--cuda_name", cuda_name
                ])
                timer_train.display_timer(print_fn)

            # Infer
            # p3 (p1, p2): Inference
            TEST_ML_DATA_DIR = EXP_ML_DATA_DIR
            infer_outdir = ig.infer_dir/f"{source_data_name}-{target_data_name}"/f"split_{split}"
            # import pdb; pdb.set_trace()
            timer_infer = Timer()
            frm_infer.main([
                # "--config_file", config_file_path,
                "--test_ml_data_dir", str(TEST_ML_DATA_DIR),
                "--model_dir", str(MODEL_OUTDIR),
                "--infer_outdir", str(infer_outdir),
                "--model_arch", str(model_arch),  # specific to GraphDRP
                "--y_col_name", y_col_name,
                "--cuda_name", cuda_name
            ])
            timer_infer.display_timer(print_fn)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # # Note! The "split" iterations are independent of each other
    # for split in split_nums:
    #     print_fn(f"Split {int(split) + 1} (id {split}) out of {len(split_nums)} splits.")
    #     # Check that train, val, and test are available. Otherwise, continue to the next split.
    #     # split = 11
    #     files_joined = [str(s) for s in split_files]
    #     phase = "train"
    #     fname = f"{source_data_name}_split_{split}_{phase}.txt"
    #     if fname not in "\t".join(files_joined):
    #         warnings.warn(f"\nThe {phase} split file {fname} is missing (continue to next split)")
    #         continue
    #     phase = "val"
    #     fname = f"{source_data_name}_split_{split}_{phase}.txt"
    #     if fname not in "\t".join(files_joined):
    #         warnings.warn(f"\nThe {phase} split file {fname} is missing (continue to next split)")
    #         continue
    #     phase = "test"
    #     fname = f"{source_data_name}_split_{split}_{phase}.txt"
    #     if fname not in "\t".join(files_joined):
    #         warnings.warn(f"\nThe {phase} split file {fname} is missing (continue to next split)")
    #         continue

    #     # Iterate over target datasets
    #     for target_data_name in target_datasets:
    #         print_fn(f"\nSource data: {source_data_name}")
    #         print_fn(f"Target data: {target_data_name}\n")

    #         EXP_ML_DATA_DIR = ig.ml_data_dir/f"{source_data_name}-{target_data_name}"/f"split_{split}"

    #         if source_data_name == target_data_name:
    #             # If source and target are the same, then infer on the test split
    #             test_split_file_name = f"{source_data_name}_split_{split}_test.txt"
    #         else:
    #             # If source and target are different, then infer on the entire target dataset
    #             test_split_file_name = f"{target_data_name}_all.txt"

    #         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #         # p1 (none): Preprocess train data
    #         # train_split_files = list((ig.splits_dir).glob(f"{source_data_name}_split_0_train*.txt"))  # TODO: placeholder for lc analysis
    #         # import pdb; pdb.set_trace()
    #         timer_preprocess = Timer()
    #         # frm_preprocess.main([
    #         ml_data_path = frm_preprocess_tr_vl_te.main([
    #             "--train_data_name", str(source_data_name),
    #             "--val_data_name", str(source_data_name),
    #             "--test_data_name", str(target_data_name),
    #             "--train_split_file_name", f"{source_data_name}_split_{split}_train.txt",
    #             "--val_split_file_name", f"{source_data_name}_split_{split}_val.txt",
    #             "--test_split_file_name", str(test_split_file_name),
    #             "--y_col_name", y_col_name,
    #             "--outdir", str(EXP_ML_DATA_DIR)
    #         ])
    #         timer_preprocess.display_timer(print_fn)

    #         # p2 (p1): Train model
    #         # Train using train samples and early stop using val samples
    #         TRAIN_ML_DATA_DIR = EXP_ML_DATA_DIR
    #         VAL_ML_DATA_DIR = EXP_ML_DATA_DIR
    #         MODEL_OUTDIR = ig.models_dir/f"{source_data_name}-{target_data_name}"/f"split_{split}"
    #         # import pdb; pdb.set_trace()
    #         timer_train = Timer()
    #         frm_train.main([
    #             # "--config_file", str(config_file_path),  # TODO: we should be able to pass the config_file
    #             "--epochs", str(epochs),  # available in config_file
    #             "--train_ml_data_dir", str(TRAIN_ML_DATA_DIR),
    #             "--val_ml_data_dir", str(VAL_ML_DATA_DIR),
    #             "--model_outdir", str(MODEL_OUTDIR),
    #             # "--ckpt_directory", str(MODEL_OUTDIR),  # TODO: we'll use candle known param ckpt_directory instead of model_outdir
    #             "--model_arch", str(model_arch),  # specific to GraphDRP
    #             "--y_col_name", y_col_name,
    #             "--cuda_name", "cuda:5"
    #         ])
    #         timer_train.display_timer(print_fn)

    #         # p3 (p1, p2): Inference
    #         TEST_ML_DATA_DIR = EXP_ML_DATA_DIR
    #         infer_outdir = ig.infer_dir/f"{source_data_name}-{target_data_name}"/f"split_{split}"
    #         # import pdb; pdb.set_trace()
    #         timer_infer = Timer()
    #         frm_infer.main([
    #             # "--config_file", config_file_path,
    #             "--test_ml_data_dir", str(TEST_ML_DATA_DIR),
    #             "--model_dir", str(MODEL_OUTDIR),
    #             "--infer_outdir", str(infer_outdir),
    #             "--model_arch", str(model_arch),  # specific to GraphDRP
    #             "--y_col_name", y_col_name,
    #             "--cuda_name", "cuda:5"
    #         ])
    #         timer_infer.display_timer(print_fn)


# timer.display_timer(print)
timer.display_timer(print_fn)
print_fn("Finished a full cross-study run.")
