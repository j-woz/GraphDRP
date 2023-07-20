
from pathlib import Path

import torch

import candle

file_path = Path(__file__).resolve().parent

additional_definitions = [
    # Preprocessing
    {"name": "set",
     "default": "mixed",
     "choices": ["mixed", "cell", "drug"],
     "type": str,
     "help": "Validation scheme (data splitting strategy).", },
    {"name": "train_split",
        "nargs": "+",
        "type": str,
        "help": "path to the file that contains the split ids (e.g., 'split_0_tr_id',  'split_0_vl_id').", },
    # Training / Inference
    {"name": "log_interval",
     "action": "store",
     "type": int,
     "help": "Interval for saving o/p", },
    {"name": "cuda_name",
     "action": "store",
     "type": str,
     "help": "Cuda device (e.g.: cuda:0, cuda:1."},
    {"name": "train_ml_data_dir",
     "action": "store",
     "type": str,
     "help": "Datadir where train data is stored."},
    {"name": "val_ml_data_dir",
     "action": "store",
     "type": str,
     "help": "Datadir where val data is stored."},
    {"name": "test_ml_data_dir",
     "action": "store",
     "type": str,
     "help": "Datadir where test data is stored."},
]

required = [
    "train_data",
    "val_data",
    "test_data",
    "train_split",
]


class BenchmarkFRM(candle.Benchmark):
    """ Benchmark for FRM. """

    def set_locals(self):
        """ Set parameters for the benchmark.

        Parameters
        ----------
        required: set of required parameters for the benchmark.
        additional_definitions: list of dictionaries describing the additional parameters for the
            benchmark.
        """
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


def initialize_parameters(default_model="frm_default_model.txt"):
    """Parse execution parameters from file or command line.

    Parameters
    ----------
    default_model : string
        File containing the default parameter definition.

    Returns
    -------
    gParameters: python dictionary
        A dictionary of Candle keywords and parsed values.
    """

    # Build benchmark object
    frm = BenchmarkFRM(
        file_path,
        default_model,
        "python",
        prog="frm",
        desc="frm functionality",
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(frm)

    return gParameters


def predicting(model, device, loader):
    """ Method to run predictions/inference.
    The same method is in frm_train.py
    TODO: put this in some utils script. --> graphdrp?

    Parameters
    ----------
    model : pytorch model
        Model to evaluate.
    device : string
        Identifier for hardware that will be used to evaluate model.
    loader : pytorch data loader.
        Object to load data to evaluate.

    Returns
    -------
    total_labels: numpy array
        Array with ground truth.
    total_preds: numpy array
        Array with inferred outputs.
    """
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print("Make prediction for {} samples...".format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _ = model(data)
            # Is this computationally efficient?
            total_preds = torch.cat((total_preds, output.cpu()), 0)  # preds to tensor
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)  # labels to tensor
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()
