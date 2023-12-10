#!/bin/bash --login

# Runs training for 2 epochs and dums predictions and scores into ap_res
# python training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch 2 --log_interval 20 --cuda_name "cuda:0" --set drug

# python training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch 2 --log_interval 20 --cuda_name "cuda:0" --set drug

# ep=2
# root=data_processed/mixed_set
# tr_file=train_data
# python -m pdb training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch $ep --log_interval 20 --cuda_name "cuda:0" --tr_file $tr_file --vl_file val_data --te_file test_data --gout out --set mix --root $root

# Preprocess
src=CCLE
trg=CCLE
split=0
data_dir=csa_data/ml_data/$src-$trg/split_$split
# outdir=csa_data/models/CCLE-CCLE/split_4
train_data_name=$src
val_data_name=$src
test_data_name=$trg
# train_split_file_name=csa_data/raw_data/splits/${src}_split_${split}_train.txt
# val_split_file_name=csa_data/raw_data/splits/${src}_split_${split}_val.txt
# test_split_file_name=csa_data/raw_data/splits/${trg}_split_${split}_test.txt
train_split_file_name=${src}_split_${split}_train.txt
val_split_file_name=${src}_split_${split}_val.txt
test_split_file_name=${trg}_split_${split}_test.txt
# test_data_name=GDSCv1
python -m ipdb frm_preprocess_tr_vl_te.py \
    --train_data_name $train_data_name \
    --val_data_name $val_data_name \
    --test_data_name $test_data_name \
    --train_split_file_name $train_split_file_name \
    --val_split_file_name $val_split_file_name \
    --test_split_file_name $test_split_file_name \
    --outdir $data_dir


# # Train
# data_dir=csa_data/ml_data/CCLE-CCLE/split_4
# outdir=csa_data/models/CCLE-CCLE/split_4
# epochs=2
# # epoch=200
# python -m pdb frm_train.py \
#     --train_ml_data_dir $data_dir \
#     --val_ml_data_dir $data_dir \
#     --model_outdir $outdir \
#     --epochs $epochs \
#     --cuda_name "cuda:0"


