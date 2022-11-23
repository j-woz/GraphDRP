# GraphDRP
GraphDRP is a Graph Convolutional Networks for Drug Response Prediction.


## Structure
GraphDRP consists of two subnetworks that learn from input features of cancer cell lines (CCLs) and drugs to predict IC50, a dose-independent treatment response. The encoded representations by the subnetworks are concatenated and passed through dense layers for the prediction of IC50. The CCLs are represented with 735 binary features including variant and copy number alterations. The drugs are represented with graph molecular structures where nodes represent the atoms of drug molecules and edges represent the bonds. Each atom is represented by 78 features. The CCL subnetwork consists of three 1-D CNNs followed by dense layers. For the drug subnetworks, four different configurations of graph neural networks (GNN) have been explored, including GCN, GAT, and GIN.


## Data Sources
The primary data resources that have been used to construct datasets for model training and testing (i.e., ML data) include:
- GDSC version 6.0 (cell line and drug ids, treatment response, cell line omics data)
- PubChem (drug SMILES)


## Data and preprocessing
CCL features and response data were downloaded from the GDSC website. Refer to Data.md for more info.


## Evaluation
Three evaluation schemes were used for performance evaluation.

- Mixed set: cell lines and drugs can appear in train, validation, and test sets.
- Cell-blind: no overlap on cell lines in train, validation, and test sets.
- Drug-blind: no overlap on drugs in train, validation, and test sets. 


## URLs
- Original GitHub: https://github.com/hauldhut/GraphDRP
- IMPROVE GitHub: https://github.com/JDACS4C-IMPROVE/GraphDRP/tree/develop
- Data: https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/GraphDRP/


## Reference
Nguyen, T.-T. et al. Graph convolutional networks for drug response prediction. *IEEE/ACM Trans Comput Biol Bioinform*. Jan-Feb 2022;19(1):146-154.
