"""Functionality for Training in Cross-Study Analysis (CSA) for GraphDRP Model."""

from pathlib import Path

from improve import csa

#from models.gat import GATNet
#from models.gat_gcn import GAT_GCN
#from models.gcn import GCNNet
#from models.ginconv import GINConvNet

from csa_graphdrp_preprocess_improve import not_used_from_model, required_csa

from graphdrp_train_improve import (
    Trainer,
    gdrp_data_conf,
    gdrp_model_conf,
    determine_device,
    build_PT_data_loader,
    evaluate_model,
    store_predictions_df,
    compute_performace_scores,
)

filepath = Path(__file__).resolve().parent



def run(params):
    """Execute graphDRP CSA model training.

    Parameters
    ----------
    params: python dictionary
        A dictionary of Candle/Improve keywords and parsed values.
    """
    # ------------------------------------------
    # Check/Construct output directory structure
    # ------------------------------------------
    totrainq, _ = csa.directory_tree_from_parameters(params, step = "train")
    # Each element of the queue contains a tuple ((source, target, split_id), ipath, opath)
    print(totrainq)

    # -----------------------------
    # [Req]
    # Determine CUDA/CPU device and configure CUDA device if available
    device = determine_device(params["cuda_name"])

    # -----------------------------
    # Train ML data for every split
    # -----------------------------
    while totrainq:
        elem = totrainq.popleft() # This is (DataSplit, InputSplitPath, OutputSplitPath)

        # indtd is dictionary with input_description: path components
        # outdtd is dictionary with output_description: path components
        indtd = {"train": elem[1], "val": elem[1], "df": elem[1]}
        outdtd = {"model": elem[2], "pred": elem[2], "scores": elem[2]}

        # -------------------------------------
        # Create Trainer object and setup train
        # -------------------------------------
        trobj = Trainer(params, device, outdtd["model"])
        trobj.setup_train()
        # -----------------------------
        # Set checkpointing
        # -----------------------------
        if params["ckpt_directory"] is None:
            params["ckpt_directory"] = outdtd["model"]
        initial_epoch = trobj.config_checkpointing(params["ckpt_directory"])

        # -----------------------------
        # Prepare PyTorch dataloaders
        train_data = f"train_{params['data_suffix']}" # TestbedDataset() appends this string with ".pt"
        train_loader = build_PT_data_loader(indtd["train"],
                                        train_data,
                                        params["batch_size"],
                                        shuffle=True)

        # Note! Don't shuffle the val_loader or results will be corrupted
        val_data = f"val_{params['data_suffix']}" # TestbedDataset() appends this string with ".pt"
        val_loader = build_PT_data_loader(indtd["val"],
                                        val_data,
                                        params["val_batch"],
                                        shuffle=False)

        # -----------------------------
        # Train
        # -----------------------------
        trobj.execute_train(train_loader, val_loader, initial_epoch)

        # -----------------------------
        # Load the (best) saved model (as determined based on val data)
        # Compute predictions
        # (groud truth), (predictions)
        val_true, val_pred = evaluate_model(params["model_arch"], device, outdtd["model"], val_loader)

        # Store predictions in data frame
        # Attempt to concat predictions with the cancer and drug ids, and the true values
        # If data frame found, then y_true is read from data frame and returned
        # Otherwise, only a partial data frame is stored (with val_true and val_pred)
        # and y_true is equal to pytorch loaded val_true
        y_true = store_predictions_df(params, indtd, outdtd, val_true, val_pred)
        # Compute performance scores
        metrics = ["mse", "rmse", "pcc", "scc", "r2"]
        compute_performace_scores(y_true, val_pred, metrics, outdtd, "val")


def main():
    params = csa.initialize_parameters(filepath,
                                       default_model="csa_graphdrp_default_model.txt",
                                       additional_definitions = gdrp_data_conf + gdrp_model_conf,
                                       required = required_csa,
                                       topop = not_used_from_model,
                                      )

    run(params)
    print("\nFinished CSA GraphDRP training.")


if __name__ == "__main__":
    main()
