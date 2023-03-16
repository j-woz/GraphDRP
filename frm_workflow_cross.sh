#!/bin/bash --login

# --------------------------------------------------
## Cross-study workflow
# --------------------------------------------------

# Parameters of the experiment/run/workflow
# TODO: this should be stored as the experiment metadata that we can go back check
source_data_name="ccle"
target_data_name=gdsc2
split=5
epochs=10

# MAIN_DATA_DIR is the dir that stores all the data (IMPROVE_DATA_DIR, CANDLE_DATA_DIR, else)
MAIN_DATA_DIR=improve_data_dir

# Sub-directories
ML_DATA_DIR=$MAIN_DATA_DIR/ml_data
MODEL_DIR=$MAIN_DATA_DIR/models
INFER_DIR=$MAIN_DATA_DIR/infer

OUTDIR=$ML_DATA_DIR


## Preprocess
# Combine tr and vl samples to use for model dev
SPLITDIR_NAME=splits
TRAIN_ML_DATADIR=$ML_DATA_DIR/data."$source_data_name"/split_"$split"_tr_vl
VAL_ML_DATADIR=$ML_DATA_DIR/data."$source_data_name"/split_"$split"_te
TEST_ML_DATADIR=$ML_DATA_DIR/data."$target_data_name"/full
python frm_preprocess.py \
    --source_data_name $source_data_name \
    --splitdir_name $SPLITDIR_NAME \
    --split_file_name split_"$split"_tr_id split_"$split"_vl_id \
    --y_col_name AUC \
    --outdir $TRAIN_ML_DATADIR
python frm_preprocess.py \
    --source_data_name $source_data_name \
    --splitdir_name $SPLITDIR_NAME \
    --split_file_name split_"$split"_te_id \
    --y_col_name AUC \
    --outdir $VAL_ML_DATADIR
# Use all samples (i.e., full dataset)
python frm_preprocess.py \
    --source_data_name $target_data_name \
    --splitdir_name $SPLITDIR_NAME \
    --split_file_name full \
    --y_col_name AUC \
    --outdir $TEST_ML_DATADIR


## HPO
# TODO: Here should be HPO to determine the best HPs


## Train (and early-stop using val data)
# Train using tr AND vl samples
# Early stop using te samples
# Save model to dir that encodes the tr, vl, and te info in the dir name
# -----
# train_ml_datadir=$ML_DATA_DIR/data."$source_data_name"/split_"$split"_tr_vl
# val_ml_datadir=$ML_DATA_DIR/data."$source_data_name"/split_"$split"_te
# -----
MODEL_OUTDIR=$MODEL_DIR/"$source_data_name"/split_"$split"/"tr_vl_te"
python frm_train.py \
    --config_file frm_params.txt \
    --epochs $epochs \
    --train_ml_datadir $TRAIN_ML_DATADIR \
    --val_ml_datadir $VAL_ML_DATADIR \
    --model_outdir $MODEL_OUTDIR
    # --source_data_name $source_data_name \
    # --split $split \


## Infer
# test_ml_datadir=$ML_DATA_DIR/data."$target_data_name"/full
model_dir=$MODEL_OUTDIR
infer_outdir=$INFER_DIR/"$source_data_name-$target_data_name"
python frm_infer.py \
    --config_file frm_params.txt \
    --test_ml_datadir $TEST_ML_DATADIR \
    --model_dir $model_dir \
    --infer_outdir $infer_outdir
