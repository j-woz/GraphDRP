import os

import candle
import frm


def run(params):
    """Execute specified data preprocessing.

    Parameters
    ----------
    params: python dictionary
        A dictionary of Candle keywords and parsed values.
    """

    # ---------------------------
    # Check/Construct output path
    # ---------------------------
    root = params["output_dir"]  # args.outdir
    if params["create_file_tree"]:
        os.makedirs(root, exist_ok=True)
    # else: # Make sure it exists

    # ---------------------------
    # Process Datasets
    # ---------------------------
    stage = ["train", "val", "test"]
    load_drug_data(stage)
    preprocess()
    preprocess_MLmodel_specific()

    load_cell_data()
    preprocess()
    preprocess_MLmodel_specific()

    combine_data() # Filter, extract features and build combination ?
    store_testbed() # PyTorch dataset




def main():
    params = frm.initialize_parameters()
    run(params)
    print(f"\nML data path:\t\n{ml_data_path}")
    print("\nFinished pre-processing (transformed raw DRP data to model input ML data).")


if __name__ == "__main__":
    main()
