#!/bin/bash

epochs=2
# epochs=10
# epochs=20

# # Within-study
# python graphdrp_train_improve.py \
#     --epochs $epochs \
#     --train_ml_data_dir ml_data/GDSCv1-CCLE/split_0 \
#     --val_ml_data_dir ml_data/GDSCv1-CCLE/split_0 \
#     --model_outdir out_model/models/GDSCv1/split_0 \
#     --cuda_name cuda:2

# Cross-study
python -m ipdb graphdrp_train_improve.py \
    --train_ml_data_dir ml_data/GDSCv1-CCLE/split_0 \
    --val_ml_data_dir ml_data/GDSCv1-CCLE/split_0 \
    --model_outdir out_model/GDSCv1/split_0 \
    --epochs $epochs \
    --cuda_name cuda:7

# Within-study
# python frm_train.py \
#     --epochs $epochs \
#     --train_ml_data_dir csa_data/ml_data/CCLE-CCLE/split_4 \
#     --val_ml_data_dir csa_data/ml_data/CCLE-CCLE/split_4 \
#     --model_outdir csa_data/models/CCLE/split_4 \
#     --model_arch 0 \
#     --y_col_name auc \
#     --cuda_name cuda:2

# # Cross-study
# python frm_train.py \
#     --epochs $epochs \
#     --train_ml_data_dir csa_data/ml_data/CCLE-GDSCv1/split_4 \
#     --val_ml_data_dir csa_data/ml_data/CCLE-GDSCv1/split_4 \
#     --model_outdir csa_data/models/CCLE/split_4 \
#     --model_arch 0 \
#     --y_col_name auc \
#     --cuda_name cuda:7
