import os

file_path = os.path.dirname(os.path.realpath(__file__))

import candle

additional_definitions = [
    {"name": "log_interval", "action": "store", "type": int},
    {"name": "modeling", "action": "store", "type": int},
    {"name": "model", "action": "store", "type": int},
    {"name": "cuda_name", "action": "store", "type": str},
    {"name": "tr_file", "action": "store", "type": str},
    {"name": "vl_file", "action": "store", "type": str},
    {"name": "te_file", "action": "store", "type": str},
    {"name": "gout", "action": "store", "type": str},
    {"name": "root", "action": "store", "type": str},
]

required = [
    "learning_rate",
    "batch_size",
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
