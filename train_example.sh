#!/bin/bash

epochs=2
# epochs=10
# epochs=20
# epochs=100

# Within-study
# All train outputs are saved in params["model_outdir"]
SOURCE=CCLE
TARGET=CCLE
python graphdrp_train_improve.py \
    --train_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_0 \
    --val_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_0 \
    --model_outdir out_model/${SOURCE}/split_0 \
    --cuda_name cuda:7
    # --epochs $epochs \


# Cross-study
# All train outputs are saved in params["model_outdir"]
SOURCE=GDSCv1
TARGET=CCLE
python graphdrp_train_improve.py \
    --train_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_0 \
    --val_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_0 \
    --model_outdir out_model/${SOURCE}/split_0 \
    --cuda_name cuda:7
    # --epochs $epochs \
