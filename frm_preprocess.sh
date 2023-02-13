#!/bin/bash --login

# Generate required datasets:
# choice:
#     0: create mixed test dataset
#     1: create saliency map dataset
#     2: create blind drug dataset
#     3: create blind cell dataset

# Data for within-study analysis
python frm_preprocess.py --src ccle --split_file_name split_5_tr_id
python frm_preprocess.py --src ccle --split_file_name split_5_vl_id
python frm_preprocess.py --src ccle --split_file_name split_5_te_id

# Data for cross-study analysis
python frm_preprocess.py --src ccle --split_file_name split_5_tr_id split_5_vl_id
python frm_preprocess.py --src gdsc2 --split_file_name full
