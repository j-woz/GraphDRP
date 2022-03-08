#!/bin/bash --login

# Generate required datasets:
# choice:
#     0: create mixed test dataset
#     1: create saliency map dataset
#     2: create blind drug dataset
#     3: create blind cell dataset
python preprocess.py --choice 0
python preprocess.py --choice 1
python preprocess.py --choice 2
python preprocess.py --choice 3

