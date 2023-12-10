# Install computational environment
The installation instructions can be found in `conda_env_py37.sh`.<br>
Step 1. Create conda env.
```sh
conda create -n GraphDRP_py37 python=3.7 pip --yes

# Resources:
+ README.md: this file.
+ data: GDSC dataset

###  source codes:
+ preprocess.py: create data in pytorch format
+ utils.py: include TestbedDataset used by create_data.py to create data, performance measures and functions to draw loss, pearson by epoch.
+ models/ginconv.py, gat.py, gat_gcn.py, and gcn.py: proposed models GINConvNet, GATNet, GAT_GCN, and GCNNet receiving graphs as input for drugs.
+ training.py: train a GraphDRP model.
+ saliancy_map.py: run this to get saliency value.


## Dependencies
+ [Torch](https://pytorch.org/)
+ [Pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)
+ [Rdkit](https://www.rdkit.org/)
+ [Matplotlib](https://matplotlib.org/)
+ [Pandas](https://pandas.pydata.org/)
+ [Numpy](https://numpy.org/)
+ [Scipy](https://docs.scipy.org/doc/)

# Step-by-step running:

## 1. Create data in pytorch format
```sh
python preprocess.py --choice 0
=======
# Resources:
+ README.md: this file.
+ data: CSA dataset

###  source codes:
+ models/ginconv.py, gat.py, gat_gcn.py, and gcn.py: proposed models GINConvNet, GATNet, GAT_GCN, and GCNNet receiving graphs as input for drugs.
+ graphdrp_preprocess_improve.py: create data from cell and drug files for pytorch data loaders
+ graphdrp_train_improve.py: train a GraphDRP model.
+ graphdrp_infer_improve.py: infer responses with a trained GraphDRP model.
+ csa_graphdrp_preprocess_improve.py: create data from cell and drug files for pytorch data loaders for a cross-study analysis (CSA).
+ csa_graphdrp_train_improve.py: train a collection of GraphDRP models for CSA.
+ improve:
    + framework.py: core utilities for creating an Improve benchmark and Candle-style kewyword handling.
    + csa.py: common utilities for creating a CSA type of workflow.
    + dataloader.py: functionality for reading raw drug or cell data.
    + rdkit\_utils.py: functionality for manipulation of drug smiles strings.
    + torch\_utils.py: functionality for constructing pytorch data loaders.
    + viz\_utils.py: generic plotting functionality. 
    + metrics.py: functionality to evaluate model performance. 


## Dependencies
+ [Torch](https://pytorch.org/)
+ [Pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)
+ [Matplotlib](https://matplotlib.org/)
+ [Pandas](https://pandas.pydata.org/)
+ [Numpy](https://numpy.org/)
+ [Scipy](https://docs.scipy.org/doc/)
+ [Rdkit](https://www.rdkit.org/)
+ [Candle-lib](https://github.com/ECP-CANDLE/candle_lib)


# Data Set:
Data can be downloaded from [ftp-site](https://ftp.mcs.anl.gov/pub/candle/public/improve/tmp)

The required data tree is shown next:

```
csa_data/raw_data/
├── index.html
├── splits
│   ├── CCLE_all.txt
│   ├── CCLE_split_0_test.txt
│   ├── CCLE_split_0_train.txt
│   ├── CCLE_split_0_val.txt
│   ├── CCLE_split_1_test.txt
│   ├── CCLE_split_1_train.txt
│   ├── CCLE_split_1_val.txt
│   ├── ...
│   ├── GDSCv2_split_9_test.txt
│   ├── GDSCv2_split_9_train.txt
│   ├── GDSCv2_split_9_val.txt
│   └── index.html
├── x_data
│   ├── cancer_copy_number.tsv
│   ├── cancer_discretized_copy_number.tsv
│   ├── cancer_DNA_methylation.tsv
│   ├── cancer_gene_expression.tsv
│   ├── cancer_miRNA_expression.tsv
│   ├── cancer_mutation_count.tsv
│   ├── cancer_mutation_long_format.tsv
│   ├── cancer_mutation.parquet
│   ├── cancer_RPPA.tsv
│   ├── drug_ecfp4_nbits512.tsv
│   ├── drug_info.tsv
│   ├── drug_mordred_descriptor.tsv
│   ├── drug_SMILES.tsv
│   └── index.html
└── y_data
    ├── index.html
    └── response.tsv
```

# Step-by-step running:

## 1. Define required environment variable to point towards data folder, e.g.
```export IMPROVE_DATA_DIR="./csa_data/"```

## 2. Preprocess raw data to construct pytorch data loaders
```python graphdrp_preprocess_improve.py```

This creates a mixed (drugs and cells) dataset and saves file pytorch data loaders (.pt), including training, validation and test sets. They are stored at out_GraphDRP/ as follows:

```
out_GraphDRP/
├── processed
│   ├── test_data.pt
│   ├── train_data.pt
│   └── val_data.pt
├── test_y_data.csv
├── train_y_data.csv
└── val_y_data.csv
```

## 3. Train a GraphDRP model
```python graphdrp_train_improve.py --epochs 10```

This trains a GraphDRP model using the processed data. The checkpointing functionality helps to store the model that has the best MSE performance on validation data. The parameters for training are taken from the default configuration file (graphdrp_default_model.txt) reproduced here:

```
[Global_Params]
model_name = "GraphDRP"
model_arch = "GINConvNet"
batch_size = 256
cuda_name = "cuda:7"
epochs = 2
learning_rate = 0.0001
log_interval = 20
ckpt_save_interval = 5
model_outdir = "./out_GraphDRP/"
test_batch = 256
val_batch = 256
optimizer = "adam"
loss = "mse"
patience = 20
test_data_processed ="test_data.pt"
train_data_processed = "train_data.pt"
val_data_processed = "val_data.pt"
test_data_df = "test_y_data.csv"
val_data_df = "val_y_data.csv"
train_ml_data_dir = "./out_GraphDRP/"
val_ml_data_dir = "./out_GraphDRP/"
test_ml_data_dir = "./out_GraphDRP/"

[Preprocess]
x_data = ["cancer_gene_expression.tsv", "drug_SMILES.tsv"]
y_data = ["response.tsv"]
response_file = "response.tsv"
cell_file = "cancer_gene_expression.tsv"
drug_file = "drug_SMILES.tsv"
gene_system_identifier = ["Gene_Symbol"]
use_lincs = True
data_set = "gCSI"
split_id = 0
canc_col_name = "improve_sample_id"
drug_col_name = "improve_chem_id"
y_col_name = "auc"
y_data_suffix = "y_data"

```
Step 2. Run the commands in `conda_env_py37.sh`.
```sh
bash conda_env_py37.sh

This returns file pytorch format (.pt) stored at data/processed including training, validation, test set.

## 2. Train a GraphDRP model
```sh
python training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0001 --num_epoch 300 --log_interval 20 --cuda_name "cuda:0"
=======

For a run of 100 epochs the following results are generated:


To train a model using training data. The model is chosen if it gains the best MSE for testing data. 
=======
out_GraphDRP/
├── best -> ./out_GraphDRP/epochs/097
├── epochs
│   ├── 088
│   │   ├── ckpt-info.json
│   │   └── model.h5
│   ├── 090
│   │   ├── ckpt-info.json
│   │   └── model.h5
│   ├── 095
│   │   ├── ckpt-info.json
│   │   └── model.h5
│   ├── 097
│   │   ├── ckpt-info.json
│   │   └── model.h5
│   └── 100
│       ├── ckpt-info.json
│       └── model.h5
├── last -> ./out_GraphDRP/epochs/100
├── model.pt
├── out_GraphDRP
│   └── ckpt.log
├── processed
│   ├── test_data.pt
│   ├── train_data.pt
│   └── val_data.pt
├── test_y_data.csv
├── train_y_data.csv
├── val_predicted.csv
├── val_scores.json
└── val_y_data.csv
```

Note that arguments can be modified by command line as the `epochs` keyword demonstrates. Additionally, different models can be selected via the `model_arch` keyword



## 4. Infer over trained model 
```python graphdrp_infer_improve.py```

The scripts uses processed data and the trained model to evaluate performance which is stored in files: `test_scores.json` and `test_predicted.csv`.

# Run the scripts
## Preprocessing.
Transform the benchmark data into model input data files.
```python
python frm_preprocess_tr_vl_te.py
```
=======
[Global_Params]
model_name = "CSA_GraphDRP"
source_data = ["gCSI"]
target_data = ["gCSI"]
split_ids = [3, 7]
model_config = "graphdrp_default_model.txt"
``` 
>>>>>>> cristina/candleize-ap

<<<<<<< HEAD
## Inference.
Run inference.
```python
python frm_infer.py
```
||||||| 5feb8c2
# IMPROVE framework development
* Create computational env
* Run frm_workflow_within.sh or frm_workflow_cross.sh
=======
This is a sample of the output produced by the preprocessing and training scripts when using `--epochs 50`:

```
csa_data/ml_data
├── gCSI-gCSI
│   ├── split_3
│   │   ├── processed
│   │   │   ├── test_data.pt
│   │   │   ├── train_data.pt
│   │   │   └── val_data.pt
│   │   ├── test_y_data.csv
│   │   ├── train_y_data.csv
│   │   └── val_y_data.csv
│   └── split_7
│       ├── processed
│       │   ├── test_data.pt
│       │   ├── train_data.pt
│       │   └── val_data.pt
│       ├── test_y_data.csv
│       ├── train_y_data.csv
│       └── val_y_data.csv
└── models
    └── gCSI-gCSI
        ├── split_3
        │   ├── best -> ./csa_data/ml_data/models/gCSI-gCSI/split_3/epochs/049
        │   ├── csa_data
        │   │   └── ml_data
        │   │       └── models
        │   │           └── gCSI-gCSI
        │   │               └── split_3
        │   │                   └── ckpt.log
        │   ├── epochs
        │   │   ├── 045
        │   │   │   ├── ckpt-info.json
        │   │   │   └── model.h5
        │   │   ├── 046
        │   │   │   ├── ckpt-info.json
        │   │   │   └── model.h5
        │   │   ├── 047
        │   │   │   ├── ckpt-info.json
        │   │   │   └── model.h5
        │   │   ├── 049
        │   │   │   ├── ckpt-info.json
        │   │   │   └── model.h5
        │   │   └── 050
        │   │       ├── ckpt-info.json
        │   │       └── model.h5
        │   ├── last -> ./csa_data/ml_data/models/gCSI-gCSI/split_3/epochs/050
        │   ├── model.pt
        │   ├── val_predicted.csv
        │   └── val_scores.json
        └── split_7
            ├── best -> ./csa_data/ml_data/models/gCSI-gCSI/split_7/epochs/050
            ├── csa_data
            │   └── ml_data
            │       └── models
            │           └── gCSI-gCSI
            │               └── split_7
            │                   └── ckpt.log
            ├── epochs
            │   ├── 043
            │   │   ├── ckpt-info.json
            │   │   └── model.h5
            │   ├── 045
            │   │   ├── ckpt-info.json
            │   │   └── model.h5
            │   ├── 046
            │   │   ├── ckpt-info.json
            │   │   └── model.h5
            │   ├── 049
            │   │   ├── ckpt-info.json
            │   │   └── model.h5
            │   └── 050
            │       ├── ckpt-info.json
            │       └── model.h5
            ├── last -> ./csa_data/ml_data/models/gCSI-gCSI/split_7/epochs/050
            ├── model.pt
            ├── val_predicted.csv
            └── val_scores.json
```
