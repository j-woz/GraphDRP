#!/bin/bash --login

set -e

# Manually run these commands before running this sciprt
# conda create -n GraphDRP_py37 python=3.7 pip --yes
# conda activate GraphDRP_py37

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch --yes
conda install pyg -c pyg -c conda-forge --yes

conda install -c conda-forge matplotlib --yes
conda install -c conda-forge h5py=3.1 --yes

conda install -c bioconda pubchempy --yes
conda install -c rdkit rdkit --yes
conda install -c anaconda networkx --yes
conda install -c conda-forge pyarrow=10.0 --yes

conda install -c pyston psutil --yes

# CANDLE
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop
# pip install git+https://github.com/ECP-CANDLE/candle_lib@candle_data_dir

# My packages
# conda install -c conda-forge ipdb=0.13.9 --yes
# conda install -c conda-forge jupyterlab=3.2.0 --yes
# conda install -c conda-forge python-lsp-server=1.2.4 --yes

# Check installs
# python -c "import torch; print(torch.__version__)"
# python -c "import torch; print(torch.version.cuda)"
# python -c "import torch_geometric; print(torch_geometric.__version__)"
# python -c "import networkx; print(networkx.__version__)"
# python -c "import matplotlib; print(matplotlib.__version__)"
# python -c "import h5py; print(h5py.version.info)"
# python -c "import pubchempy; print(pubchempy.__version__)"
# python -c "import rdkit; print(rdkit.__version__)"


# ----------------------------------------
# Some help from here:
# https://www.youtube.com/watch?v=lu2DzaqBeDg
# https://github.com/kaust-rccl/conda-environment-examples/tree/pytorch-geometric-and-friends

# # creates the conda environment
# PROJECT_DIR=$PWD
# conda env create --prefix $PROJECT_DIR/env --file $PROJECT_DIR/environment.yml --force

# # activate the conda env before installing PyTorch Geometric via pip
# conda activate $PROJECT_DIR/env
# TORCH=1.6.0
# CUDA=cu102
# python -m pip install torch-scatter --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
# python -m pip install torch-sparse --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
# python -m pip install torch-cluster --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
# python -m pip install torch-spline-conv --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
# python -m pip install torch-geometric --no-cache-dir
