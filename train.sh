#############################################################################################################
### THIS IS A TEMPLATE FILE. SUBSTITUTE '???' ENTRIES WITH THE APPROPRiATE INFORMATION FOR YOUR CONTAINER ###
#############################################################################################################

#!/bin/bash

# arg 1 CUDA_VISIBLE_DEVICES
# arg 2 CANDLE_DATA_DIR
# arg 3 CANDLE_CONFIG

### Path to your CANDLEized model's main Python script ###
## Usage in container
CANDLE_MODEL=/usr/local/GraphDRP/graphdrp_candle.py
## Debug on lambda
# CANDLE_MODEL=/lambda_stor/data/apartin/projects/IMPROVE/pan-models/GraphDRP/graphdrp_candle.py

# Check total input parameters
# test: bash train_tmp.sh 0
# test: bash train_tmp.sh 0 candle_data_dir
if [[ "$#" < 2  ]] ; then
	    echo "Illegal number of parameters"
	    echo "CUDA_VISIBLE_DEVICES CANDLE_DATA_DIR are required"
	    exit -1
fi

# REQUIRED_PARAMS=("CUDA_VISIBLE_DEVICES" "CANDLE_DATA_DIR")
# N_PARAMS_REQUIRED=${#REQUIRED_PARAMS[@]}
# N_PARAMS_PASSED="$#"

# echo ${REQUIRED_PARAMS[0]}
# echo ${REQUIRED_PARAMS[@]}

# Check total input parameters
# test: bash train.sh 5
# test: bash train.sh 5 candle_data_dir
# if [[ $N_PARAMS_PASSED < $N_PARAMS_REQUIRED ]] ; then
# 	    echo "Illegal number of parameters"
#         echo "Total parameters passed: $N_PARAMS_PASSED"
# 	    echo "Required parameters: ${REQUIRED_PARAMS[@]}"
# 	    exit -1
# fi

# Reading command line arguments
CUDA_VISIBLE_DEVICES=$1
CANDLE_DATA_DIR=$2
CANDLE_CONFIG=$3

# # Reading command line arguments
# # https://github.com/JDACS4C-IMPROVE/DeepTTC/blob/develop/train.sh
# CUDA_VISIBLE_DEVICES=$1; shift
# CANDLE_DATA_DIR=$1; shift

# Check if CANDLE_CONFIG was passed as input parameter and define CMD.
# Check if "${CANDLE_CONFIG}" is zero
# test: bash train.sh 0 candle_data_dir
# test: bash train.sh 0 candle_data_dir graphdrp_default_model.txt
# test: bash train.sh 0 candle_data_dir graphdrp_model_candle.txt
# https://linuxhint.com/bash_operator_examples/#o48
if [ ! -z "${CANDLE_CONFIG}" ]; then
        # https://linuxhint.com/bash_operator_examples/#o53
        if [ ! -f "$CANDLE_CONFIG" ]; then
            echo "This file was not found: $CANDLE_CONFIG"
            echo "Correct usage: train.sh CUDA_VISIBLE_DEVICES CANDLE_DATA_DIR CANDLE_CONFIG"
            exit -1
        else
            CMD="python3 ${CANDLE_MODEL} --config_file ${CANDLE_CONFIG}"
        fi
else
	CMD="python3 ${CANDLE_MODEL}"
fi

# Tom:       CMD="python3 ${CANDLE_MODEL}"
# Oleksandr: CMD="python3 ${CANDLE_MODEL} $@"
# CMD="python3 ${CANDLE_MODEL} $@"

# Display runtime arguments
echo "Using container ... "
echo "Using CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Using CANDLE_DATA_DIR: ${CANDLE_DATA_DIR}"
echo "Using CANDLE_CONFIG: ${CANDLE_CONFIG}"
echo "Running command: ${CMD}"

# Set up environmental variables and execute model
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} CANDLE_DATA_DIR=${CANDLE_DATA_DIR} $CMD
