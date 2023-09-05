#!/bin/bash --login

# Generate required datasets:
# choice:
#     0: create mixed test dataset
#     1: create saliency map dataset
#     2: create blind drug dataset
#     3: create blind cell dataset

outdir="data_processed"
python preprocess.py --choice 0 --outdir $outdir
python preprocess.py --choice 2 --outdir $outdir
python preprocess.py --choice 3 --outdir $outdir
