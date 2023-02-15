#!/bin/bash --login

# Parameters
src="ccle"
target_data_name=$src
split=5
epochs=2

ML_DATA_DIR=improve_data_dir/ml_data
MODEL_DIR=improve_data_dir/models
INFER_DIR=improve_data_dir/infer

## Infer
test_ml_datadir=$ML_DATA_DIR/data."$src"/split_"$split"_te
model_dir=$model_outdir
# infer_outdir=$INFER_DIR/"$src"/split_"$split"
# infer_outdir=$INFER_DIR/"$src"/split_"$split"/"tr_vl-te"
infer_outdir=$INFER_DIR/"$src-$target_data_name"/split_"$split"
python frm_infer.py \
    --config_file frm_params.txt \
    --src $src \
    --test_ml_datadir $test_ml_datadir \
    --model_dir $model_dir \
    --infer_outdir $infer_outdir

## Infer
test_ml_datadir=$ML_DATADIR/data."$src"/split_"$split"_te
model_dir=$model_outdir
infer_outdir=$INFER_DIR/"$src-$target_data_name"
python frm_infer.py \
    --config_file frm_params.txt \
    --src $src \
    --test_ml_datadir $test_ml_datadir \
    --model_dir $model_dir \
    --infer_outdir $infer_outdir
