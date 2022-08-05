import os

file_path = os.path.dirname(os.path.realpath(__file__))

import candle

additional_definitions = [
    {"name": "log_interval", "action": "store", "type": int, "help": "interval for saving o/p"},
    {"name": "modeling", "action": "store", "type": int, "help": "Integer: see model folder for options"},    
    {"name": "root", "action": "store", "type": str, "help": "needed for preprocessing/dataloader"},
    {"name": "trn_batch_size", "default": 32, "type": int, "help": "input batch size for training",},
    {"name": "val_batch_size", "default": 32, "type": int, "help": "input batch size for validation",},    
    {"name": "test_batch_size", "default": 32, "type": int, "help": "input batch size for testing",},    
]

required = [
    "learning_rate",
    "epochs",
    "log_interval",
    "model_name",
    "set",
]


class BenchmarkGraphDRP(candle.Benchmark):
    """Benchmark for GraphDRP"""

    def set_locals(self):
        """Set parameters for the benchmark.

        Args:
            required: set of required parameters for the benchmark.
            additional_definitions: list of dictionaries describing the additional parameters for the
            benchmark.
        """
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions
