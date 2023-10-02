#!/bin/bash --login

# Generate required datasets:
# choice:
#     0: create mixed test dataset
#     1: create saliency map dataset
#     2: create blind drug dataset
#     3: create blind cell dataset
# python preprocess.py --choice 0
# python preprocess.py --choice 1
# python preprocess.py --choice 2
# python preprocess.py --choice 3

ep=300

tr_file=train_sz_1
python training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch $ep --log_interval 20 --cuda_name "cuda:0" --tr_file $tr_file --vl_file val_data --te_file test_data --gout lc_out/$tr_file --set mix --root lc_data/mix_drug_cell

tr_file=train_sz_2
python training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch $ep --log_interval 20 --cuda_name "cuda:0" --tr_file $tr_file --vl_file val_data --te_file test_data --gout lc_out/$tr_file --set mix --root lc_data/mix_drug_cell

tr_file=train_sz_3
python training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch $ep --log_interval 20 --cuda_name "cuda:0" --tr_file $tr_file --vl_file val_data --te_file test_data --gout lc_out/$tr_file --set mix --root lc_data/mix_drug_cell

tr_file=train_sz_4
python training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch $ep --log_interval 20 --cuda_name "cuda:0" --tr_file $tr_file --vl_file val_data --te_file test_data --gout lc_out/$tr_file --set mix --root lc_data/mix_drug_cell

tr_file=train_sz_5
python training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch $ep --log_interval 20 --cuda_name "cuda:0" --tr_file $tr_file --vl_file val_data --te_file test_data --gout lc_out/$tr_file --set mix --root lc_data/mix_drug_cell

tr_file=train_sz_6
python training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch $ep --log_interval 20 --cuda_name "cuda:0" --tr_file $tr_file --vl_file val_data --te_file test_data --gout lc_out/$tr_file --set mix --root lc_data/mix_drug_cell

tr_file=train_sz_7
python training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch $ep --log_interval 20 --cuda_name "cuda:0" --tr_file $tr_file --vl_file val_data --te_file test_data --gout lc_out/$tr_file --set mix --root lc_data/mix_drug_cell

