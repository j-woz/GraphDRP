#!/bin/bash --login

# Runs training for 2 epochs and dums predictions and scores into ap_res
# python training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch 2 --log_interval 20 --cuda_name "cuda:0" --set drug

# python training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch 2 --log_interval 20 --cuda_name "cuda:0" --set drug

# ep=2
# root=data_processed/mixed_set
# tr_file=train_data
# python -m pdb training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch $ep --log_interval 20 --cuda_name "cuda:0" --tr_file $tr_file --vl_file val_data --te_file test_data --gout out --set mix --root $root

data_dir=csa_data/ml_data/CCLE-CCLE/split_4
outdir=csa_data/models/CCLE-CCLE/split_4
epochs=2
# epoch=200
python -m pdb frm_train.py \
    --train_ml_data_dir $data_dir \
    --val_ml_data_dir $data_dir \
    --model_outdir $outdir \
    --epochs $epochs \
    --cuda_name "cuda:0"


