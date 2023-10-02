#!/bin/bash --login

# Runs training for 2 epochs and dums predictions and scores into ap_res
# python training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch 2 --log_interval 20 --cuda_name "cuda:0" --set drug

# python training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch 2 --log_interval 20 --cuda_name "cuda:0" --set drug

# ep=2
# root=data_processed/mix_drug_cell
# tr_file=train_data
# python -m pdb training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch $ep --log_interval 20 --cuda_name "cuda:0" --tr_file $tr_file --vl_file val_data --te_file test_data --gout out --set mix --root $root

# python -m ipdb preprocess_sc.py --datadir ./data/data.gdsc1
# python preprocess_cs.py --datadir ./data/data.gdsc1

# # Create data
# # python preprocess_cs.py --choice 4  --datadir ./data/data.gdsc1 --which_data org
# python preprocess_cs.py --choice 4  --datadir ./data/data.gdsc1 --which_data cs --split 0

# Train model
# ep=2
# split=0
# src=$1
# device=$2
# datadir=data
# python training_cs.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch $ep --log_interval 20 --cuda_name "cuda:$device" --train_data train_data --val_data val_data --test_data test_data --datadir $datadir --split $split --src $src

ep=50
src=$1
device=$2
datadir=data
# train_batch=1024
train_batch=256
# lr=0.0001
lr=0.0005
n_splits=9
for sp in $(seq 0 $n_splits); do
    echo "Split $sp out of $n_splits"
    sleep 1
    python training_cs.py --model 0 --train_batch $train_batch --val_batch 1024 --test_batch 1024 \
        --lr $lr --num_epoch $ep --log_interval 10 --cuda_name "cuda:$device" \
        --train_data train_data --val_data val_data --test_data test_data --datadir $datadir \
        --split $sp --src $src
    sleep 1
done


# n_splits=9
# sources=(ccle ctrp gcsi gdsc1 gdsc2)
# for src in ${sources[@]}; do
#     echo "Processing $src"
#     sleep 1
#     python training_cs.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001  \
#         --num_epoch $ep --log_interval 20 --cuda_name "cuda:0" --tr_file $tr_file --vl_file val_data \
#         --te_file test_data --gout out.cs --datadir $datadir --root $root --split $split
#     sleep 1
# done

# SETS=(1 2 3)
# for SET in ${SETS[@]}; do
#     out_dir=$GLOBAL_SUFX/set${SET}
#     echo "Outdir $out_dir"
#     for split in $(seq $START_SPLIT 1 $N_SPLITS); do
#         device=$(($split % 6))
#         echo "Set $SET; Split $split; Device $device"
#         jsrun -n 1 -a 1 -c 4 -g 1 ./jsrun_brk.sh $device $split $out_dir $SET \
#             exec >logs/run"$split".log 2>&1 &
#     done
# done


# n_splits=9
# for sp in $(seq 0 $n_splits); do
#     echo "Split $sp out of $n_splits"
#     sleep 1
#     python preprocess_cs.py --choice 4  --datadir $datadir --which_data cs --split $sp
#     sleep 1
# done

