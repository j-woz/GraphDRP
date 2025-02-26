#!/bin/bash

# Below are two examples of how to run the end-to-end scripts:
# 1. Within-study analysis
# 2. Cross-study analysis
# Uncomment and run the one you are you interested in.


# # Download the benchmark CSA data
# wget --cut-dirs=8 -P ./ -nH -np -m https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/

# ======================================================================
# Set env variables:
# 1. IMPROVE_DATA_DIR
# 2. IMPROVE lib
# TODO finish this
# -------------------
current_dir=/lambda_stor/data/apartin/projects/IMPROVE/pan-models/GraphDRP
echo "PWD: $current_dir"

# Set env variable for IMPROVE lib
# export PYTHONPATH=$PYTHONPATH:

# Set env variable for IMPROVE_DATA_DIR (benchmark dataset)
# export IMPROVE_DATA_DIR="./csa_data/"
# echo "IMPROVE_DATA_DIR: $IMPROVE_DATA_DIR"
# ======================================================================

# ----------------------------------------
# 1. Within-study
# ---------------

SOURCE=CCLE
TARGET=CCLE

# Preprocess
# All preprocess outputs are saved in params["ml_data_outdir"]
python graphdrp_preprocess_improve.py \
    --train_split_file ${SOURCE}_split_0_train.txt \
    --val_split_file ${SOURCE}_split_0_val.txt \
    --test_split_file ${TARGET}_split_0_test.txt \
    --ml_data_outdir ml_data/${SOURCE}-${TARGET}/split_0

# Train
# All train outputs are saved in params["model_outdir"]
python graphdrp_train_improve.py \
    --train_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_0 \
    --val_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_0 \
    --model_outdir out_model/${SOURCE}/split_0

# Infer
# All infer outputs are saved in params["infer_outdir"]
python graphdrp_infer_improve.py \
    --test_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_0 \
    --model_dir out_model/${SOURCE}/split_0 \
    --infer_outdir out_infer/${SOURCE}-${TARGET}/split_0


# ----------------------------------------
# 2. Cross-study
# --------------

# SOURCE=GDSCv1
# TARGET=CCLE

# # Preprocess
# # All preprocess outputs are saved in params["ml_data_outdir"]
# python graphdrp_preprocess_improve.py \
#     --train_split_file ${SOURCE}_split_0_train.txt \
#     --val_split_file ${SOURCE}_split_0_val.txt \
#     --test_split_file ${TARGET}_all.txt \
#     --ml_data_outdir ml_data/${SOURCE}-${TARGET}/split_0

# # Train
# # All train outputs are saved in params["model_outdir"]
# python graphdrp_train_improve.py \
#     --train_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_0 \
#     --val_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_0 \
#     --model_outdir out_model/${SOURCE}/split_0

# # Infer
# # All infer outputs are saved in params["infer_outdir"]
# python graphdrp_infer_improve.py \
#     --test_ml_data_dir ml_data/${SOURCE}-${TARGET}/split_0 \
#     --model_dir out_model/${SOURCE}/split_0 \
#     --infer_outdir out_infer/${SOURCE}-${TARGET}/split_0
