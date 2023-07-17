from pathlib import Path

import candle

file_path = Path(__file__).resolve().parent

additional_definitions = [
    {
        "name": "train_split",
        "nargs": "+",
        "type": str,
        "help": "path to the file that contains the split ids (e.g., 'split_0_tr_id',  'split_0_vl_id').",
    },
]

required = [
    "train_data",
    "val_data",
    "test_data",
    "train_split",
]


class BmkParse(candle.Benchmark):
    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


def initialize_parameters(default_model="default_preprocess.txt"):
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
    bmk = BmkParse(
        file_path,
        default_model,
        "python",
        prog="preprocess",
        desc="Generic pre-processing functionality",
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(bmk)

    return gParameters


def run(params):
    """Execute specified data preprocessing."""


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
