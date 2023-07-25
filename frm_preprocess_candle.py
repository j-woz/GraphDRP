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

    # Download one file or collection
    # Extract if zip file
    # Load one file or collection
    #   Operation on load (load columns subset)
    #   Define column to load as label
    #   Need to have operations for cell data / drug data ...
    # Pre-process
    #   Reduce / filtrate
    #   Scale / Inpute / Store scaler object
    #   Combine data
    # Log stats loaded / processed data



def main():
    params = frm.initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
