#!/bin/bash --login

## Data for within-study analysis
src="ccle"
target_data_name=$src
split=5

## Within-study
python frm_preprocess.py --src $src --split_file_name split_"$split"_tr_id
python frm_preprocess.py --src $src --split_file_name split_"$split"_vl_id
python frm_preprocess.py --src $src --split_file_name split_"$split"_te_id

## Cross-study analysis
# Combine train and val samples to use for model dev
python frm_preprocess.py --src $src --split_file_name split_"$split"_tr_id split_"$split"_vl_id
python frm_preprocess.py --src $src --split_file_name split_"$split"_te_id
# Use all samples (i.e., full dataset)
python frm_preprocess.py --src $target_data_name --split_file_name full

