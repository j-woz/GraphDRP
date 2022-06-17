import os

file_path = os.path.dirname(os.path.realpath(__file__))

import candle

additional_definitions = [
    {"name": "log_interval", "action": "store", "type": int},
]

required = [
    "learning_rate",
    "batch_size",
    "epochs",
    "log_interval"
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
