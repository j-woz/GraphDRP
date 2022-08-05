#!/bin/bash --login

src=$1
datadir=./data/data.$src

# datadir=./data/data.ccle
# datadir=./data/data.ctrp
# datadir=./data/data.gcsi
# datadir=./data/data.gdsc1
# datadir=./data/data.gdsc2

# Create data
# python preprocess_cs.py --choice 4  --datadir $datadir --which_data cs --split 0

n_splits=9
for sp in $(seq 0 $n_splits); do
    echo "Split $sp out of $n_splits"
    sleep 1
    python preprocess_cs.py --choice 4  --datadir $datadir --which_data cs --split $sp
    sleep 1
done

# # Train model
# ep=2
# root=data/data.gdsc1/data_split_0
# split=0
# tr_file=train_data
# python training_cs.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch $ep --log_interval 20 --cuda_name "cuda:0" --tr_file $tr_file --vl_file val_data --te_file test_data --gout out.cs --set mix --root $root --split $split
