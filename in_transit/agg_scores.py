# %%
from pathlib import Path
import numpy as np
import pandas as pd

# %%
fdir = Path(__file__).resolve().parent
res_dir = fdir/"ap_res"

scheme = ["mix", "drug_split", "cell_split"]

# file_scores = res_dir.glob("scores*.json")
# for s in scheme:
#     pass

# %%
print("Mixed")
col_names = ["RMSE", "MSE", "CCp", "CCs"]
df1 = pd.read_csv("result_GCNNet_GDSC_mixed.csv", names=col_names); df1["Model"] = "GCN"
df2 = pd.read_csv("result_GINConvNet_GDSC_mixed.csv", names=col_names); df2["Model"] = "GIN"
df3 = pd.read_csv("result_GATNet_GDSC_mixed.csv", names=col_names); df3["Model"] = "GAT"
df4 = pd.read_csv("result_GAT_GCN_GDSC_mixed.csv", names=col_names); df4["Model"] = "GCN_GAT"
df = pd.concat([df1, df2, df3, df4], axis=0)
df = df.drop(columns=["MSE", "CCs"])
df_mixed = df[["Model", "CCp", "RMSE"]]
print(df_mixed)


# %%
print("Drug-blind")
col_names = ["RMSE", "MSE", "CCp", "CCs"]
df1 = pd.read_csv("result_GCNNet_GDSC_drug_blind.csv", names=col_names); df1["Model"] = "GCN"
df2 = pd.read_csv("result_GINConvNet_GDSC_drug_blind.csv", names=col_names); df2["Model"] = "GIN"
df3 = pd.read_csv("result_GATNet_GDSC_drug_blind.csv", names=col_names); df3["Model"] = "GAT"
df4 = pd.read_csv("result_GAT_GCN_GDSC_drug_blind.csv", names=col_names); df4["Model"] = "GCN_GAT"
df = pd.concat([df1, df2, df3, df4], axis=0)
df = df.drop(columns=["MSE", "CCs"])
df_drug = df[["Model", "CCp", "RMSE"]]
print(df_drug)


# %%
print("Cell-blind")
col_names = ["RMSE", "MSE", "CCp", "CCs"]
df1 = pd.read_csv("result_GCNNet_GDSC_cell_blind.csv", names=col_names); df1["Model"] = "GCN"
df2 = pd.read_csv("result_GINConvNet_GDSC_cell_blind.csv", names=col_names); df2["Model"] = "GIN"
df3 = pd.read_csv("result_GATNet_GDSC_cell_blind.csv", names=col_names); df3["Model"] = "GAT"
df4 = pd.read_csv("result_GAT_GCN_GDSC_cell_blind.csv", names=col_names); df4["Model"] = "GCN_GAT"
df = pd.concat([df1, df2, df3, df4], axis=0)
df = df.drop(columns=["MSE", "CCs"])
df_cell = df[["Model", "CCp", "RMSE"]]
print(df_cell)
