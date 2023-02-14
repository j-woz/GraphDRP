#!/bin/bash --login

## Data for within-study analysis
python frm_preprocess.py --src ccle --split_file_name split_5_tr_id
python frm_preprocess.py --src ccle --split_file_name split_5_vl_id
python frm_preprocess.py --src ccle --split_file_name split_5_te_id

## Data for cross-study analysis
# Combine train and val samples to use for model dev
python frm_preprocess.py --src ccle --split_file_name split_5_tr_id split_5_vl_id
# Use all samples (i.e., full dataset)
python frm_preprocess.py --src gdsc2 --split_file_name full
