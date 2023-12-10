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
lg = Logger(ig.main_data_dir/"csa_within.log")
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


# def continue_if_split_not_found(source_data_name, split, phase):
#     """
#     source_data_name (str):
#     split (str or int):
#     phase (str): train, val, or test
#     Retuns:
#         bool: True if split not found, and False otherwise
#     Example:
#         continue_if_split_not_found("CCLE", 0, "train")
#     """
#     # train_fname = f"{source_data_name}_split_{split}_train.txt"
#     fname = f"{source_data_name}_split_{split}_{phase}.txt"
#     if fname not in "\t".join([str(s) for s in split_files]):
#         warnings.warn(f"\nThe {phase} split file {fname} " \
#                       "is missing (continue to the next split)")
#         continue


# ===============================================================
###  Within-study
# ===============================================================

# import ipdb; ipdb.set_trace()
timer = Timer()
for source_data_name in data_sources:

    # Source and target are the same in within-study analysis
    target_data_name = source_data_name
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
        TRAIN_ML_DATA_DIR = ig.ml_data_dir/source_data_name/f"split_{split}_train"
        # import pdb; pdb.set_trace()
        frm_preprocess.main([
            "--source_data_name", str(source_data_name),
            # "--target_data_name", "",  # This might be required in certain models (e.g., IGTD, MCA)
            "--split_file_name", f"{source_data_name}_split_{split}_train.txt",  # str or list of strings
            "--outdir", str(TRAIN_ML_DATA_DIR),
            "--y_col_name", y_col_name
        ])

        # p2 (none): Preprocess val data
        VAL_ML_DATA_DIR = ig.ml_data_dir/source_data_name/f"split_{split}_val"
        # import pdb; pdb.set_trace()
        frm_preprocess.main([
            "--source_data_name", str(source_data_name),
            "--split_file_name", f"{source_data_name}_split_{split}_val.txt",  # str or list of strings
            "--outdir", str(VAL_ML_DATA_DIR),
            "--y_col_name", y_col_name
        ])

        # p3 (p1, p2): Train model
        # --------
        ## Train
        # --------
        # Train using train samples
        # Early stop using val samples
        # Save model to dir that encodes the train and val info in the dir name
        MODEL_OUTDIR = ig.models_dir/source_data_name/f"split_{split}"/"train-val"
        # import pdb; pdb.set_trace()
        frm_train.main([
            # "--config_file", str(config_file_path),  # TODO: we should be able to pass the config_file
            "--epochs", str(epochs),  # available in config_file
            "--train_ml_data_dir", str(TRAIN_ML_DATA_DIR),
            "--val_ml_data_dir", str(VAL_ML_DATA_DIR),
            "--model_outdir", str(MODEL_OUTDIR),
            "--model_arch", str(model_arch),  # GraphDRP specific
            "--y_col_name", y_col_name,
            "--cuda_name", "cuda:4"
        ])

        # p4 (none): Preprocess test data
        TEST_ML_DATA_DIR = ig.ml_data_dir/source_data_name/f"split_{split}_test"
        # import pdb; pdb.set_trace()
        frm_preprocess.main([
            "--source_data_name", str(source_data_name),
            "--split_file_name", f"{source_data_name}_split_{split}_test.txt",  # str or list of strings
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
            # "--config_file", str(config_file_path),  # TODO: we should be able to pass the config_file
            "--test_ml_data_dir", str(TEST_ML_DATA_DIR),
            "--model_dir", str(MODEL_OUTDIR),
            "--infer_outdir", str(infer_outdir),
            "--model_arch", str(model_arch),  # GraphDRP specific
            "--y_col_name", y_col_name,
            "--cuda_name", "cuda:4"
        ])


# timer.display_timer(print)
timer.display_timer(print_fn)
print_fn("Finished a full within-study run.")





# ----------------------------------------------------
# Temp
# ----------------------------------------------------

def groupby_src_and_print(df, print_fn=print):
    # print_fn(df.groupby('SOURCE').agg({'CancID': 'nunique', 'DrugID': 'nunique'}).reset_index())
    print_fn(df.groupby(improve_globals.source_col_name).agg(
        {improve_globals.canc_col_name: 'nunique',
         improve_globals.drug_col_name: 'nunique'}).reset_index())


def load_and_merge(source_data_name, use_lincs=True, verbose=False):
    """ ... """
    src_dir = datadir/f"data.{source_data_name}"
    splitdir = src_dir/"splits"

    # Load data
    rsp = pd.read_csv(src_dir/f"rsp_{source_data_name}.csv")      # Drug response
    ge = pd.read_csv(src_dir/f"ge_{source_data_name}.csv")        # Gene expressions
    mrd = pd.read_csv(src_dir/f"mordred_{source_data_name}.csv")  # Mordred descriptors
    fps = pd.read_csv(src_dir/f"ecfp2_{source_data_name}.csv")    # Morgan fingerprints
    smi = pd.read_csv(src_dir/f"smiles_{source_data_name}.csv")   # SMILES

    # Use landmark genes
    if use_lincs:
        with open(src_dir/"../landmark_genes") as f:
            genes = [str(line.rstrip()) for line in f]
        genes = ["ge_" + str(g) for g in genes]
        print(len(set(genes).intersection(set(ge.columns[1:]))))
        genes = list(set(genes).intersection(set(ge.columns[1:])))
        cols = ["CancID"] + genes
        ge = ge[cols]

    if verbose:
        groupby_src_and_print(rsp)
        print("Unique cell lines with gene expressions", ge["CancID"].nunique())
        print("Unique drugs with Mordred", mrd["DrugID"].nunique())
        print("Unique drugs with ECFP2", fps["DrugID"].nunique())

    # Merge (tidy df)
    data = pd.merge(rsp, ge, on='CancID', how='inner')
    data = pd.merge(data, mrd, on='DrugID', how='inner')
    groupby_src_and_print(data)
    return data


# Get features (x), target (y), and meta
def extract_subset_fea(df, fea_list, fea_sep='_'):
    """ Extract features based feature prefix name. """
    fea = [c for c in df.columns if (c.split(fea_sep)[0]) in fea_list]
    return df[fea]

    
def split_data(data, source_data_name, split):
    """ ... """
    src_dir = datadir/f"data.{source_data_name}"
    splitdir = src_dir/"splits"
    # print("\nGet the splits.")

    with open(splitdir/f"split_{split}_tr_id") as f:
        tr_id = [int(line.rstrip()) for line in f]

    with open(splitdir/f"split_{split}_te_id") as f:
        te_id = [int(line.rstrip()) for line in f]

    # Train and test data
    tr_data = data.loc[tr_id]
    te_data = data.loc[te_id]

    # Val data from tr_data
    from sklearn.model_selection import train_test_split
    tr_data, vl_data = train_test_split(tr_data, test_size=0.12, random_state=seed)

    tr_data = tr_data.reset_index(drop=True)
    vl_data = vl_data.reset_index(drop=True)
    te_data = te_data.reset_index(drop=True)
    print("Train", tr_data.shape)
    print("Val  ", vl_data.shape)
    print("Test ", te_data.shape)

    xtr, ytr = extract_subset_fea(tr_data, fea_list=fea_list, fea_sep=fea_sep), tr_data[[trg_name]]
    xvl, yvl = extract_subset_fea(vl_data, fea_list=fea_list, fea_sep=fea_sep), vl_data[[trg_name]]
    xte, yte = extract_subset_fea(te_data, fea_list=fea_list, fea_sep=fea_sep), te_data[[trg_name]]
    assert xtr.shape[0] == ytr.shape[0], "Size missmatch."
    assert xvl.shape[0] == yvl.shape[0], "Size missmatch."
    assert xte.shape[0] == yte.shape[0], "Size missmatch."

    return tr_data, vl_data, te_data, xtr, ytr, xvl, yvl, xte, yte


# Scores
def calc_scores(y_true, y_pred):
    """ ... """
    import sklearn
    from scipy.stats import pearsonr, spearmanr
    scores = {}
    scores['r2'] = sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred)
    scores['mean_absolute_error'] = sklearn.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)
    scores['spearmanr'] = spearmanr(y_true, y_pred)[0]
    scores['pearsonr'] = pearsonr(y_true, y_pred)[0]
    return scores
