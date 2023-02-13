#!/bin/bash --login

# Use data splits (within study)
src="ccle"
split=5
epochs=10
python frm_train.py \
    --config_file prm_within_src.txt \
    --epochs $epochs \
    --src $src \
    --split $split \
    --train_ml_datadir split_"$split"_tr \
    --val_ml_datadir split_"$split"_vl

# Use data splits (cross study)
src="ccle"
split=5
epochs=10
python frm_train.py \
    --config_file prm_within_src.txt \
    --epochs $epochs \
    --src $src \
    --split $split \
    --train_ml_datadir split_"$split"_tr_vl \
    --val_ml_datadir split_"$split"_te

# # Use full dataset (cross study)
# python frm_train.py \
#     --config_file prm_cross_src.txt \
#     --src ccle \
#     --split full \
#     # --output_dir improve_data_dir \
#     # --experiment_id ccle \
#     # --run_id split_5 \
