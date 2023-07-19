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
    root = params["output_dir"]  # args.outdir
    os.makedirs(root, exist_ok=True)


def main():
    params = frm.initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
