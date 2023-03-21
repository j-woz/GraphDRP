#!/bin/bash --login

# ================================
## Learning Curves (LC) workflow
# ================================

# --------------------
## Workflow settings
# --------------------
cuda=$1

# Parameters of the experiment/run/workflow
# TODO: this should be stored as the experiment metadata that we can revisit
# source_data_name="ccle"
# source_data_name="ctrp"
# source_data_name="gdsc1"
source_data_name="gdsc2"
target_data_name=$source_data_name
split=0
# split=1
# split=2
# split=3
# split=4
# split=5
# split=6
# split=7
# split=8
# split=9
# split=10
# epochs=2
# epochs=10
epochs=30
# epochs=50
y_col_name=AUC
# LC settings
tr_sz_start=0
# tr_sz_end=0
tr_sz_end=10
tr_sz_arr=($(seq $tr_sz_start 1 $tr_sz_end))

# MAIN_DATA_DIR is the dir that stores all the data (IMPROVE_DATA_DIR, CANDLE_DATA_DIR, else)
# TODO: The MAIN_DATA_DIR and the sub-directories below should standardized. How?
MAIN_DATA_DIR=improve_data_dir
# MAIN_DATA_DIR=improve_data_dir_lc

# Sub-directories
ML_DATA_DIR=$MAIN_DATA_DIR/ml_data
MODEL_DIR=$MAIN_DATA_DIR/models
INFER_DIR=$MAIN_DATA_DIR/infer

OUTDIR=$ML_DATA_DIR

## ML datadirs
# TRAIN_ML_DATADIR, however, can change based on the run (e.g., learning curves)
VAL_ML_DATADIR=$ML_DATA_DIR/data."$source_data_name"/split_"$split"_vl
TEST_ML_DATADIR=$ML_DATA_DIR/data."$source_data_name"/split_"$split"_te


## (Arrays in bash)
# tr_sz_start=0
# tr_sz_end=11
# tr_sz_arr=($(seq $tr_sz_start 1 $tr_sz_end))
# # for ii in {0..${n_splits}}; do
# # for ii in {0..99}; do
# for tr_sz in ${tr_sz_arr[@]}; do
#     echo -e "Train size $tr_sz"
# done


# -------------
## Preprocess
# -------------
SPLITDIR_NAME=lc_splits
# tr_sz=0
# TRAIN_ML_DATADIR=$ML_DATA_DIR/data."$source_data_name"/split_"$split"_tr_sz_"$tr_sz"_id
for tr_sz in ${tr_sz_arr[@]}; do
    tr_split_file_name=split_"$split"_tr_sz_"$tr_sz"_id
    TRAIN_ML_DATADIR=$ML_DATA_DIR/data."$source_data_name"/$tr_split_file_name
    # echo -e "Split $split; train size $tr_sz"
    python frm_preprocess.py \
        --source_data_name $source_data_name \
        --splitdir_name $SPLITDIR_NAME \
        --split_file_name $tr_split_file_name \
        --y_col_name $y_col_name \
        --outdir $TRAIN_ML_DATADIR
done
vl_split_file_name=split_"$split"_vl_id
python frm_preprocess.py \
    --source_data_name $source_data_name \
    --splitdir_name $SPLITDIR_NAME \
    --split_file_name $vl_split_file_name \
    --y_col_name $y_col_name \
    --outdir $VAL_ML_DATADIR
te_split_file_name=split_"$split"_te_id
python frm_preprocess.py \
    --source_data_name $source_data_name \
    --splitdir_name $SPLITDIR_NAME \
    --split_file_name $te_split_file_name \
    --y_col_name $y_col_name \
    --outdir $TEST_ML_DATADIR


# ------
## HPO
# ------
# TODO: Here should be HPO to determine the best HPs


# --------
## Train
# --------
# Train using tr samples
# Early stop using vl samples
# Save model to dir that encodes the tr and vl info in the dir name
for tr_sz in ${tr_sz_arr[@]}; do
    tr_split_file_name=split_"$split"_tr_sz_"$tr_sz"_id
    TRAIN_ML_DATADIR=$ML_DATA_DIR/data."$source_data_name"/$tr_split_file_name
    # MODEL_OUTDIR=$MODEL_DIR/"$source_data_name"/split_"$split"/"tr_vl"
    MODEL_OUTDIR=$MODEL_DIR/"$source_data_name"/split_"$split"/tr_sz_"$tr_sz"_vl
    echo -e "Split $split; train size $tr_sz"
    echo -e "Model outdir $MODEL_OUTDIR"
    python frm_train.py \
        --config_file frm_params.txt \
        --epochs $epochs \
        --y_col_name $y_col_name \
        --train_ml_datadir $TRAIN_ML_DATADIR \
        --val_ml_datadir $VAL_ML_DATADIR \
        --model_outdir $MODEL_OUTDIR \
        --cuda_name $cuda
done


# --------
## Infer
# --------
for tr_sz in ${tr_sz_arr[@]}; do
    model_dir=$MODEL_DIR/"$source_data_name"/split_"$split"/tr_sz_"$tr_sz"_vl
    infer_outdir=$INFER_DIR/"$source_data_name-$target_data_name"/split_"$split"_tr_sz_"$tr_sz"
    python frm_infer.py \
        --config_file frm_params.txt \
        --test_ml_datadir $TEST_ML_DATADIR \
        --model_dir $model_dir \
        --infer_outdir $infer_outdir
done


# ------------------
## Post-processing
# ------------------
# python lc_code/tmp_plot.py
