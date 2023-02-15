#!/bin/bash --login

# --------------------------------------------------
## Cross-study workflow
# --------------------------------------------------

# Parameters
source_data_name="ccle"
target_data_name=gdsc2
split=5
epochs=2

ML_DATA_DIR=improve_data_dir/ml_data
MODEL_DIR=improve_data_dir/models
INFER_DIR=improve_data_dir/infer


## Preprocess
# Combine tr and vl samples to use for model dev
python frm_preprocess.py --source_data_name $source_data_name --split_file_name split_"$split"_tr_id split_"$split"_vl_id --sens_score AUC
python frm_preprocess.py --source_data_name $source_data_name --split_file_name split_"$split"_te_id --sens_score AUC
# Use all samples (i.e., full dataset)
python frm_preprocess.py --source_data_name $target_data_name --split_file_name full --sens_score AUC


## HPO
# TODO: Here should be HPO to determine the best HPs


## Train (and early-stop using val data)
# Train using tr AND vl samples
# Early stop using te samples
# Save model to dir that encodes the tr, vl, and te info in the dir name
train_ml_datadir=$ML_DATA_DIR/data."$source_data_name"/split_"$split"_tr_vl
val_ml_datadir=$ML_DATA_DIR/data."$source_data_name"/split_"$split"_te
model_outdir=$MODEL_DIR/"$source_data_name"/split_"$split"/"tr_vl_te"
python frm_train.py \
    --config_file frm_params.txt \
    --epochs $epochs \
    --source_data_name $source_data_name \
    --split $split \
    --train_ml_datadir $train_ml_datadir \
    --val_ml_datadir $val_ml_datadir \
    --model_outdir $model_outdir


## Infer
test_ml_datadir=$ML_DATA_DIR/data."$target_data_name"/full
model_dir=$model_outdir
infer_outdir=$INFER_DIR/"$source_data_name-$target_data_name"
python frm_infer.py \
    --config_file frm_params.txt \
    --test_ml_datadir $test_ml_datadir \
    --model_dir $model_dir \
    --infer_outdir $infer_outdir
