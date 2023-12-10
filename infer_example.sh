#!/bin/bash

# # Cross-study
# python frm_infer.py \
#     --test_ml_data_dir csa_data/ml_data/CCLE-GDSCv1/split_4 \
#     --model_dir csa_data/models/CCLE/split_4 \
#     --infer_outdir csa_data/infer/CCLE-GDSCv1/split_4 \
#     --cuda_name cuda:2

# Within-study
python -m ipdb graphdrp_infer_improve.py \
    --test_ml_data_dir ml_data/GDSCv1-CCLE/split_0 \
    --model_dir out_model/GDSCv1/split_0 \
    --infer_outdir out_infer/GDSCv1-CCLE/split_0
    --cuda_name cuda:7

# # Within-study
# python frm_infer.py \
#     --test_ml_data_dir csa_data/ml_data/CCLE-CCLE/split_4 \
#     --model_dir csa_data/models/CCLE/split_4 \
#     --infer_outdir csa_data/infer/CCLE-CCLE/split_4 \
#     --model_arch 0 \
#     --y_col_name auc \
#     --cuda_name cuda:2

# # Cross-study
# python frm_infer.py \
#     --test_ml_data_dir csa_data/ml_data/CCLE-GDSCv1/split_4 \
#     --model_dir csa_data/models/CCLE/split_4 \
#     --infer_outdir csa_data/infer/CCLE-GDSCv1/split_4 \
#     --model_arch 0 \
#     --y_col_name auc \
#     --cuda_name cuda:2

