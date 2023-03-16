#!/bin/bash --login

# --------------------------------------------------
## Within-study workflow
# --------------------------------------------------

# Parameters of the experiment/run/workflow
# TODO: this should be stored as the experiment metadata that we can go back check
source_data_name="ccle"
target_data_name=$source_data_name
split=5
# epochs=2
epochs=10
y_col_name=AUC

# MAIN_DATA_DIR is the dir that stores all the data (IMPROVE_DATA_DIR, CANDLE_DATA_DIR, else)
# TODO: The MAIN_DATA_DIR and the sub-directories below should standardized. How?
MAIN_DATA_DIR=improve_data_dir

# Sub-directories
ML_DATA_DIR=$MAIN_DATA_DIR/ml_data
MODEL_DIR=$MAIN_DATA_DIR/models
INFER_DIR=$MAIN_DATA_DIR/infer

OUTDIR=$ML_DATA_DIR

## Preprocess CSG
# TODO: If a model needs info about the target dataset (primarily for CSG), this can be provided as target_data_name.
SPLITDIR_NAME=splits
TRAIN_ML_DATADIR=$ML_DATA_DIR/data."$source_data_name"/split_"$split"_tr
VAL_ML_DATADIR=$ML_DATA_DIR/data."$source_data_name"/split_"$split"_vl
TEST_ML_DATADIR=$ML_DATA_DIR/data."$source_data_name"/split_"$split"_te
python frm_preprocess.py \
    --source_data_name $source_data_name \
    --splitdir_name $SPLITDIR_NAME \
    --split_file_name split_"$split"_tr_id \
    --y_col_name $y_col_name \
    --outdir $TRAIN_ML_DATADIR
python frm_preprocess.py \
    --source_data_name $source_data_name \
    --splitdir_name $SPLITDIR_NAME \
    --split_file_name split_"$split"_vl_id \
    --y_col_name $y_col_name \
    --outdir $VAL_ML_DATADIR
python frm_preprocess.py \
    --source_data_name $target_data_name \
    --splitdir_name $SPLITDIR_NAME \
    --split_file_name split_"$split"_te_id \
    --y_col_name $y_col_name \
    --outdir $TEST_ML_DATADIR

# ## Preprocess LC
# # SPLITDIR=$ML_DATA_DIR/lc_splits
# SPLITDIR_NAME=lc_splits
# tr_sz=4
# python frm_preprocess.py --source_data_name $source_data_name \
#     --splitdir_name $SPLITDIR_NAME \
#     --split_file_name split_"$split"_tr_sz_"$tr_sz"_id \
#     --y_col_name AUC
# python frm_preprocess.py --source_data_name $source_data_name \
#     --splitdir_name $SPLITDIR_NAME \
#     --split_file_name split_"$split"_vl_id \
#     --y_col_name AUC
# python frm_preprocess.py --source_data_name $source_data_name \
#     --splitdir_name $SPLITDIR_NAME \
#     --split_file_name split_"$split"_te_id \
#     --y_col_name AUC


## HPO
# TODO: Here should be HPO to determine the best HPs


## Train (and early-stop using val data)
# Train using tr samples
# Early stop using vl samples
# Save model to dir that encodes the tr and vl info in the dir name
MODEL_OUTDIR=$MODEL_DIR/"$source_data_name"/split_"$split"/"tr_vl"
python frm_train.py \
    --config_file frm_params.txt \
    --epochs $epochs \
    --y_col_name $y_col_name \
    --train_ml_datadir $TRAIN_ML_DATADIR \
    --val_ml_datadir $VAL_ML_DATADIR \
    --model_outdir $MODEL_OUTDIR


## Infer
model_dir=$MODEL_OUTDIR
infer_outdir=$INFER_DIR/"$source_data_name-$target_data_name"/split_"$split"
python frm_infer.py \
    --config_file frm_params.txt \
    --test_ml_datadir $TEST_ML_DATADIR \
    --model_dir $model_dir \
    --infer_outdir $infer_outdir
