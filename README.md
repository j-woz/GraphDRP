# Install computational environment
The installation instructions can be found in `conda_env_py37.sh`.<br>
Step 1. Create conda env.
```sh
conda create -n GraphDRP_py37 python=3.7 pip --yes
```
Step 2. Run the commands in `conda_env_py37.sh`.
```sh
bash conda_env_py37.sh
```

# Run the scripts
## Preprocessing.
Transform the benchmark data into model input data files.
```python
python frm_preprocess_tr_vl_te.py
```

## Training.
Train the model.
```python
python frm_train.py
```

## Inference.
Run inference.
```python
python frm_infer.py
```
