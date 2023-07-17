import os
from pathlib import Path

import candle
import graphdrp as bmk

file_path = Path(__file__).resolve().parent


def initialize_parameters(default_model="csa_params.txt"):
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
    gdrp = bmk.BenchmarkGraphDRP(
        file_path,
        default_model,
        "python",
        prog="pre-process",
        desc="Generic pre-process functionality",
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(gdrp)

    return gParameters


def run(params):
    """Execute specified data preprocessing."""
    root = params["output_dir"]  # args.outdir
    os.makedirs(root, exist_ok=True)


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
