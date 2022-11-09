import os
import candle

file_path = os.path.dirname(os.path.realpath(__file__))

additional_definitions = [
    {"name": "log_interval",
     "action": "store",
     "type": int,
     "help": "Interval for saving o/p"},
    {"name": "cuda_name",
     "action": "store",
     "type": str,
     "help": "Cuda device (e.g.: cuda:0, cuda:1."},
    {"name": "modeling",
     "action": "store",
     "type": int,
     "help": "Integer. 0: GINConvNet, 1: GATNet, 2: GAT_GCN, 3: GCNNet"},    
    #{"name": "root",
    # "action": "store",
    # "type": str,
    # "help": "Needed for preprocessing/dataloader."},
    #{"name": "epochs",
    # "default": 300,
    # "type": int,
    # "help": "Number of epochs.",},
    {"name": "batch_size",
     "default": 256,
     "type": int,
     "help": "Input batch size for training.",},
    {"name": "val_batch",
     "default": 256,
     "type": int,
     "help": "Input batch size for validation.",},    
    {"name": "test_batch",
     "default": 256,
     "type": int,
     "help": "Input batch size for testing.",},
    {"name": "set",
     "default": "mixed",
     "choices": ["mixed", "cell", "drug"],
     "type": str,
     "help": "Validation scheme (data splitting strategy).",},
]

required = [
    "epochs",
    "learning_rate",
    "log_interval",
    "model_name",
    "set",
    "train_data",
    "val_data",
    "test_data",
]


class BenchmarkGraphDRP(candle.Benchmark):
    """ Benchmark for GraphDRP. """

    def set_locals(self):
        """ Set parameters for the benchmark.

        Args:
            required: set of required parameters for the benchmark.
            additional_definitions: list of dictionaries describing the additional parameters for the
            benchmark.
        """
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions
