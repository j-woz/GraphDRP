import os
import warnings
from pathlib import Path
import pandas as pd

import improve_utils
from improve_utils import improve_globals as ig

import frm_preprocess
import frm_train
import frm_infer

from ap_utils.classlogger import Logger
from ap_utils.utils import get_print_func, Timer


fdir = Path(__file__).resolve().parent


# AP
lg = Logger(ig.main_data_dir/"csa_cross.log")
# print_fn = print
print_fn = get_print_func(lg.logger)
print_fn(f"File path: {fdir}")


# data_sources = ["ccle", "ctrp", "gcsi", "gdsc1", "gdsc2"]
data_sources = ["CCLE", "CTRPv2", "gCSI", "GDSCv1", "GDSCv2"]
fea_list = ["ge", "mordred"]
fea_sep = "."
seed = 0


# Parameters of the experiment/run/workflow
# TODO: this should be stored as the experiment metadata that we can go back check
# split = 0
epochs = 1
# epochs = 2
# epochs = 10
# epochs = 50
y_col_name = "auc"
config_file_name = "csa_params.txt"
config_file_path = fdir/config_file_name
model_arch = 0  # GraphDRP-specific param (0-3: different model architectures)


# ===============================================================
###  Cross-study
# ===============================================================

# import pdb; pdb.set_trace()
timer = Timer()
for source_data_name in data_sources:

    # Iter over target datasets (for cross-study analysis)
    # target_data_name = source_data_name
    for target_data_name in data_sources:
        if source_data_name == target_data_name:
            continue
        print_fn(f"\nSource data: {source_data_name}")
        print_fn(f"Target data: {target_data_name}\n")

        # Get the split file paths
        split_files = list((ig.splits_dir).glob(f"{source_data_name}_split_*.txt"))
        # This parsing assumes splits file names are: SOURCE_split_NUM_[train/val/test].txt
        split_nums = [str(s).split("split_")[1].split("_")[0] for s in split_files]
        split_nums = sorted(set(split_nums))

        for split in split_nums:
            print_fn(f"Split {int(split) + 1} (id {split}) out of {len(split_nums)} splits.")
            # Check that train, val, and test are available
            # split = 11
            files_joined = [str(s) for s in split_files]
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

            # p1 (none): Preprocess train data
            # train_split_files = list((ig.splits_dir).glob(f"{source_data_name}_split_0_train*.txt"))  # TODO: placeholder for lc analysis
            TRAIN_ML_DATA_DIR = ig.ml_data_dir/source_data_name/f"split_{split}_train_val"
            # import pdb; pdb.set_trace()
            frm_preprocess.main([
                "--source_data_name", str(source_data_name),
                # "--split_file_name", [f"{source_data_name}_split_{split}_train.txt" f"{source_data_name}_split_{split}_val.txt"],  # str or list of strings
                "--split_file_name", f"{source_data_name}_split_{split}_train.txt",  f"{source_data_name}_split_{split}_val.txt",  # str or list of strings
                "--y_col_name", y_col_name,
                "--outdir", str(TRAIN_ML_DATA_DIR)
            ])

            # p2 (none): Preprocess val data
            VAL_ML_DATA_DIR = ig.ml_data_dir/source_data_name/f"split_{split}_test"
            # import pdb; pdb.set_trace()
            frm_preprocess.main([
                "--source_data_name", str(source_data_name),
                "--split_file_name", f"{source_data_name}_split_{split}_test.txt",  # str or list of strings
                "--outdir", str(VAL_ML_DATA_DIR),
                "--y_col_name", y_col_name
            ])

            # p3 (p1, p2): Train model
            # --------
            ## Train
            # --------
            # Train using train and val samples
            # Early stop using train samples
            # Save model to dir that encodes the train and val info in the dir name
            MODEL_OUTDIR = ig.models_dir/source_data_name/f"split_{split}"/"train_val-test"
            # import pdb; pdb.set_trace()
            frm_train.main([
                # "--config_file", str(config_file_path),  # TODO: we should be able to pass the config_file
                "--epochs", str(epochs),  # available in config_file
                "--train_ml_data_dir", str(TRAIN_ML_DATA_DIR),
                "--val_ml_data_dir", str(VAL_ML_DATA_DIR),
                "--model_outdir", str(MODEL_OUTDIR),
                "--model_arch", str(model_arch),  # GraphDRP specific
                "--y_col_name", y_col_name,
                "--cuda_name", "cuda:5"
            ])

            # p4 (none): Preprocess test data
            TEST_ML_DATA_DIR = ig.ml_data_dir/target_data_name/f"all"
            # import pdb; pdb.set_trace()
            frm_preprocess.main([
                "--source_data_name", str(target_data_name),
                "--split_file_name", f"{target_data_name}_all.txt",  # str or list of strings
                "--outdir", str(TEST_ML_DATA_DIR),
                "--y_col_name", y_col_name
            ])

            # p5 (p3, p4): Inference
            # --------
            ## Infer
            # --------
            infer_outdir = ig.infer_dir/f"{source_data_name}-{target_data_name}"/f"split_{split}"
            # import pdb; pdb.set_trace()
            frm_infer.main([
                # "--config_file", config_file_path,
                "--test_ml_data_dir", str(TEST_ML_DATA_DIR),
                "--model_dir", str(MODEL_OUTDIR),
                "--infer_outdir", str(infer_outdir),
                "--model_arch", str(model_arch),  # GraphDRP specific
                "--y_col_name", y_col_name,
                "--cuda_name", "cuda:5"
            ])


# timer.display_timer(print)
timer.display_timer(print_fn)
print_fn("Finished a full cross-study run.")
