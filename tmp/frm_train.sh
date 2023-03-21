#!/bin/bash --login

# Parameters
src="ccle"
target_data_name=$src
split=5
epochs=2

ML_DATA_DIR=improve_data_dir/ml_data
MODEL_DIR=improve_data_dir/models
INFER_DIR=improve_data_dir/infer

## Train (and early-stop using val data)
# Train using tr samples
# Early stop using vl samples
# Save model to dir that encodes the tr and vl info in the dir name
train_ml_datadir=$ML_DATA_DIR/data."$src"/split_"$split"_tr
val_ml_datadir=$ML_DATA_DIR/data."$src"/split_"$split"_vl
model_outdir=$MODEL_DIR/"$src"/split_"$split"/"tr_vl"
python frm_train.py \
    --config_file frm_params.txt \
    --epochs $epochs \
    --src $src \
    --split $split \
    --train_ml_datadir $train_ml_datadir \
    --val_ml_datadir $val_ml_datadir \
    --model_outdir $model_outdir

## Train (and early-stop using val data)
# Train using tr AND vl samples
# Early stop using te samples
# Save model to dir that encodes the tr, vl, and te info in the dir name
train_ml_datadir=$ML_DATADIR/data."$src"/split_"$split"_tr_vl
val_ml_datadir=$ML_DATADIR/data."$src"/split_"$split"_te
model_outdir=$MODEL_DIR/"$src"/split_"$split"/"tr_vl_te"
python frm_train.py \
    --config_file frm_params.txt \
    --epochs $epochs \
    --src $src \
    --split $split \
    --train_ml_datadir $train_ml_datadir \
    --val_ml_datadir $val_ml_datadir \
    --model_outdir $model_outdir
