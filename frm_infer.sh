#!/bin/bash --login

# Note! Here model_dir was trained using the splits 5 but that doesn't have
# be the case.
model_dir="improve_data_dir/models/ccle/split_5_tr_vl-split_5_te"
src="gdsc2" # target dataset to run inference on
test_ml_datadir="full" # how which samples to use ("full" to all samples)
python frm_infer.py \
    --config_file prm_within_src.txt \
    --src $src \
    --model_dir $model_dir \
    --test_ml_datadir $test_ml_datadir
