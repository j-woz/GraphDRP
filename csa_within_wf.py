import os
from pathlib import Path
import pandas as pd

import frm_preprocess
# from improve_utils import imp_globals
import improve_utils
from improve_utils import improve_globals as ig


fdir = Path(__file__).resolve().parent

# data_sources = ["ccle", "ctrp", "gcsi", "gdsc1", "gdsc2"]
data_sources = ["CCLE", "CTRPv2", "gCSI", "GDSCv1", "GDSCv2"]
fea_list = ["ge", "mordred"]
# fea_sep = "_"
fea_sep = "."
# trg_name = "AUC"
# trg_name = "auc"
seed = 0

def groupby_src_and_print(df, print_fn=print):
    # print_fn(df.groupby('SOURCE').agg({'CancID': 'nunique', 'DrugID': 'nunique'}).reset_index())
    print_fn(df.groupby(improve_globals.source_col_name).agg(
        {improve_globals.canc_col_name: 'nunique',
         improve_globals.drug_col_name: 'nunique'}).reset_index())

# source_data_name = "ccle"
# source_data_name = "ctrp"
# source_data_name = "gcsi"
# source_data_name = "gdsc1"
# source_data_name = "gdsc2"
# datadir = fdir/f"ml.dfs/July2020/data.{source_data_name}"

# datadir = fdir/"ml.dfs/July2020"


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


# Parameters of the experiment/run/workflow
# TODO: this should be stored as the experiment metadata that we can go back check
# source_data_name = "CCLE"
# target_data_name = "GDSCv2"
# target_data_name = None
# split = 0
epochs = 2
# epochs = 10
y_col_name = "auc"
config_file_name = "csa_params.txt"
config_file_path = fdir/config_file_name

# Metadata
# raw_data_dir_name = "raw_data"
# x_datadir_name = "x_data"
# y_datadir_name = "y_data"
# canc_col_name = "CancID"
# drug_col_name = "DrugID"
# ge_fname = "ge.parquet"  # cancer feature
# smiles_fname = "smiles.csv"  # drug feature
# y_file_substr = "rsp"
# ...
# X_DATA_DIR_NAME = ig.x_data_dir_name
# Y_DATA_DIR_NAME = ig.y_data_dir_name
# CANC_COL_NAME = ig.canc_col_name
# DRUG_COL_NAME = ig.drug_col_name
# GE_FNAME = ig.ge_fname
# SMILES_FNAME = ig.smiles_fname
# # Y_FILE_SUBSTR = ig.y_file_substr
# Y_FILE_NAME = ig.y_file_name
# ----
# Split dir name (CSG: 'splits', LC: 'lc_splits')
# splits_datadir_name = "splits"
# SPLITDIR_NAME=splits_datadir_name

# MAIN_DATA_DIR is the dir that stores all the data (IMPROVE_DATADIR, CANDLE_DATA_DIR, else)
# TODO: The MAIN_DATA_DIR and the sub-directories below should standardized. How?
# MAIN_DATA_DIR_NAME = "improve_data_dir"
# MAIN_DATA_DIR_NAME = "csa_data"
# MAIN_DATA_DIR = fdir/MAIN_DATA_DIR_NAME

# Sub-directories
# RAW_DATA_DIR = MAIN_DATA_DIR/raw_data_dir_namC
# ML_DATA_DIR = MAIN_DATA_DIR/"ml_data"
# MODEL_DIR = MAIN_DATA_DIR/"models"
# INFER_DIR = MAIN_DATA_DIR/"infer"
# RAW_DATA_DIR = MAIN_DATA_DIR/ig.raw_data_dir_name
# ML_DATA_DIR = MAIN_DATA_DIR/ig.ml_data_dir_name
# MODEL_DIR = MAIN_DATA_DIR/ig.models_dir_name
# INFER_DIR = MAIN_DATA_DIR/ig.infer_dir_name

# PP_OUTDIR=ML_DATA_DIR  # preprocess
# TR_OUTDIR=ML_DATA_DIR  # preprocess
# IF_OUTDIR=ML_DATA_DIR  # preprocess


### Within-study
# import ipdb; ipdb.set_trace(context=5)
for source_data_name in data_sources:

    # Get the number of available splits
    ## sf = RAW_DATA_DIR/f"data.{source_data_name}"/splits_datadir_name  # split files
    ## split_files = list((sf).glob("split_*tr_id"))
    split_files = list((ig.splits_dir).glob(f"{source_data_name}_split_*_train.txt"))

    # TODO: check that train, val, and test are available

    for split in range(len(split_files)):
    # for split in range(10):

        # ----------
        # Preprocess
        # ----------
        # import ipdb; ipdb.set_trace(context=5)
        # TRAIN_ML_DATA_DIR=ML_DATA_DIR/f"data.{source_data_name}"/f"split_{split}_tr"
        # frm_preprocess.main([
        #     "--source_data_name", str(source_data_name),
        #     # "--target_data_name", "",  # This might be required in certain models (e.g., IGTD, MCA)
        #     "--splitdir_name", str(SPLITDIR_NAME),
        #     "--split_file_name", f"split_{split}_tr_id",
        #     "--y_col_name", y_col_name,
        #     "--outdir", str(TRAIN_ML_DATA_DIR),
        #     "--main_data_dir", str(MAIN_DATA_DIR)
        # ])

        # p1 (none): Preprocess train data
        # train_split_files = list((ig.splits_dir).glob(f"{source_data_name}_split_0_train*.txt"))  # TODO: placeholder for lc analysis
        TRAIN_ML_DATA_DIR = ig.ml_data_dir/source_data_name/f"split_{split}_train"
        import pdb; pdb.set_trace()
        frm_preprocess.main([
            "--source_data_name", str(source_data_name),
            "--split_file_name", f"{source_data_name}_split_{split}_train.txt",  # str or list of strings
            "--y_col_name", y_col_name,
            "--outdir", str(TRAIN_ML_DATA_DIR),
            "--split", str(split)
        ])

        # p2 (none): Preprocess val data
        VAL_ML_DATADIR = ig.ml_data_dir/source_data_name/f"split_{split}_val"
        import pdb; pdb.set_trace()
        frm_preprocess.main([
            "--source_data_name", str(source_data_name),
            "--split_file_name", f"{source_data_name}_split_{split}_val.txt",  # str or list of strings
            "--y_col_name", y_col_name,
            "--outdir", str(VAL_ML_DATADIR),
            "--split", str(split)
        ])

        # p4 (none): Preprocess test data
        TEST_ML_DATADIR = ig.ml_data_dir/source_data_name/f"split_{split}_test"
        import pdb; pdb.set_trace()
        frm_preprocess.main([
            "--source_data_name", str(source_data_name),
            "--split_file_name", f"{source_data_name}_split_{split}_test.txt",  # str or list of strings
            "--y_col_name", y_col_name,
            "--outdir", str(TEST_ML_DATADIR),
            "--split", str(split)
        ])

        # p3 (p1, p2): Train model
        # --------
        ## Train
        # --------
        # Train using tr samples
        # Early stop using vl samples
        # Save model to dir that encodes the tr and vl info in the dir name
        # TODO: consider separate cross-study and within-study results
        import pdb; pdb.set_trace()
        MODEL_OUTDIR = ig.models_dir/source_data_name/f"split_{split}"/"tr_vl"
        frm_train.main([
            "--config_file", config_file_path,
            "--epochs", epochs,
            "--y_col_name", y_col_name,
            "--train_ml_datadir", TRAIN_ML_DATADIR,
            "--val_ml_datadir", VAL_ML_DATADIR,
            "--model_outdir", MODEL_OUTDIR
        ])

        # p5 (p3, p5): Inference
        pass


        # SPLITDIR_NAME=splits
        # TRAIN_ML_DATA_DIR=$ML_DATA_DIR/data."$source_data_name"/split_"$split"_tr
        # VAL_ML_DATADIR=$ML_DATA_DIR/data."$source_data_name"/split_"$split"_vl
        # TEST_ML_DATADIR=$ML_DATA_DIR/data."$source_data_name"/split_"$split"_te
        # python frm_preprocess.py \
        #     --source_data_name $source_data_name \
        #     --splitdir_name $SPLITDIR_NAME \
        #     --split_file_name split_"$split"_tr_id \
        #     --y_col_name $y_col_name \
        #     --outdir $TRAIN_ML_DATA_DIR
        # python frm_preprocess.py \
        #     --source_data_name $source_data_name \
        #     --splitdir_name $SPLITDIR_NAME \
        #     --split_file_name split_"$split"_vl_id \
        #     --y_col_name $y_col_name \
        #     --outdir $VAL_ML_DATADIR
        # python frm_preprocess.py \
        #     --source_data_name $target_data_name \
        #     --splitdir_name $SPLITDIR_NAME \
        #     --split_file_name split_"$split"_te_id \
        #     --y_col_name $y_col_name \
        #     --outdir $TEST_ML_DATADIR




        data = load_and_merge(source_data_name, use_lincs=True)
        tr_data, vl_data, te_data, xtr, ytr, xvl, yvl, xte, yte = split_data(data, source_data_name, split)

        # Scale
        # from ml.scale import scale_fea
        # xdata = scale_fea(xdata=xdata, scaler_name=args['scaler'])

        # Train model
        import lightgbm as lgb
        ml_init_args = {'n_estimators': 1000, 'max_depth': -1, 'learning_rate': 0.1,
                        'num_leaves': 31, 'n_jobs': 8, 'random_state': None}
        ml_fit_args = {'verbose': False, 'early_stopping_rounds': 50}
        ml_fit_args['eval_set'] = (xvl, yvl)
        model = lgb.LGBMRegressor(objective='regression', **ml_init_args)
        model.fit(xtr, ytr, **ml_fit_args)

        # outdir = fdir/f"results.csa.{source_data_name}"
        outdir = fdir/f"results.csa"
        os.makedirs(outdir, exist_ok=True)

        # Predict
        y_pred = model.predict(xte)
        y_true = yte.values.squeeze()
        pred = pd.DataFrame({"True": y_true, "Pred": y_pred})
        te_df = te_data[["DrugID", "CancID", trg_name]]
        pred = pd.concat([te_df, pred], axis=1)
        assert sum(pred["True"] == pred[trg_name]) == pred.shape[0], "Columns 'AUC' and 'True' are the ground truth."
       
        pred_fname = f"{source_data_name}_{source_data_name}_split_{split}.csv"
        pred.to_csv(outdir/pred_fname, index=False)

        # Drop preds on target set
        trg_studies = [s for s in data_sources if s not in source_data_name]
        # import ipdb; ipdb.set_trace(context=5)
        for trg_study in trg_studies:

            data = load_and_merge(trg_study, use_lincs=True)
            xdata = extract_subset_fea(data, fea_list=fea_list, fea_sep=fea_sep)
            ydata = data[[trg_name]]
            meta = data[["DrugID", "CancID", trg_name]]

            y_pred = model.predict(xdata)
            y_true = ydata.values.squeeze()
            pred = pd.DataFrame({"True": y_true, "Pred": y_pred})

            pred = pd.concat([meta, pred], axis=1)
            assert sum(pred["True"] == pred[trg_name]) == pred.shape[0], "Columns trg_name and 'True' are the ground truth."
            pred_fname = f"{source_data_name}_{trg_study}_split_{split}.csv"
            pred.to_csv(outdir/pred_fname, index=False)

            scores = calc_scores(y_true=y_true, y_pred=y_pred)
            print(scores)


# =====================================================================
# =====================================================================
# =====================================================================






# import ipdb; ipdb.set_trace(context=5)
# for source_data_name in data_sources:

#     # Get the number of available splits



#     for split in range(10):

#         data = load_and_merge(source_data_name, use_lincs=True)
#         tr_data, vl_data, te_data, xtr, ytr, xvl, yvl, xte, yte = split_data(data, source_data_name, split)

#         # Scale
#         # from ml.scale import scale_fea
#         # xdata = scale_fea(xdata=xdata, scaler_name=args['scaler'])

#         # Train model
#         import lightgbm as lgb
#         ml_init_args = {'n_estimators': 1000, 'max_depth': -1, 'learning_rate': 0.1,
#                         'num_leaves': 31, 'n_jobs': 8, 'random_state': None}
#         ml_fit_args = {'verbose': False, 'early_stopping_rounds': 50}
#         ml_fit_args['eval_set'] = (xvl, yvl)
#         model = lgb.LGBMRegressor(objective='regression', **ml_init_args)
#         model.fit(xtr, ytr, **ml_fit_args)

#         # outdir = fdir/f"results.csa.{source_data_name}"
#         outdir = fdir/f"results.csa"
#         os.makedirs(outdir, exist_ok=True)

#         # Predict
#         y_pred = model.predict(xte)
#         y_true = yte.values.squeeze()
#         pred = pd.DataFrame({"True": y_true, "Pred": y_pred})
#         te_df = te_data[["DrugID", "CancID", trg_name]]
#         pred = pd.concat([te_df, pred], axis=1)
#         assert sum(pred["True"] == pred[trg_name]) == pred.shape[0], "Columns 'AUC' and 'True' are the ground truth."
       
#         pred_fname = f"{source_data_name}_{source_data_name}_split_{split}.csv"
#         pred.to_csv(outdir/pred_fname, index=False)

#         # Drop preds on target set
#         trg_studies = [s for s in data_sources if s not in source_data_name]
#         # import ipdb; ipdb.set_trace(context=5)
#         for trg_study in trg_studies:

#             data = load_and_merge(trg_study, use_lincs=True)
#             xdata = extract_subset_fea(data, fea_list=fea_list, fea_sep=fea_sep)
#             ydata = data[[trg_name]]
#             meta = data[["DrugID", "CancID", trg_name]]

#             y_pred = model.predict(xdata)
#             y_true = ydata.values.squeeze()
#             pred = pd.DataFrame({"True": y_true, "Pred": y_pred})

#             pred = pd.concat([meta, pred], axis=1)
#             assert sum(pred["True"] == pred[trg_name]) == pred.shape[0], "Columns trg_name and 'True' are the ground truth."
#             pred_fname = f"{source_data_name}_{trg_study}_split_{split}.csv"
#             pred.to_csv(outdir/pred_fname, index=False)

#             scores = calc_scores(y_true=y_true, y_pred=y_pred)
#             print(scores)
