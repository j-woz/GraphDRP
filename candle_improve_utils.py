import json
import candle
import os
from pprint import pprint
from pathlib import Path
# from pathlib import PurePath


file_path = os.path.dirname(os.path.realpath(__file__))


required = []


def remove_suffix(text, suffix):
    if suffix and text.endswith(suffix):
        return text[:-len(suffix)]
    return text


def parser_from_json(json_file):
    """ Custom parser to read a json file and return the list of included keywords.
        Special case for True/False since these are not handled correctly by the default
        python command line parser.
        All keywords defined in json files are subsequently available to be overwritten
        from the command line, using the CANDLE command line parser.
    Parameters
    ----------
    json_file: File to be parsed

    Return
    ----------
    new_defs: Dictionary of parameters

    """
    file = open(json_file,)
    params = json.load(file)
    new_defs = []
    for key in params:
        if params[key][0] == "True" or params[key][0] == "False":
            new_def = {'name': key,
                       'type': (type(candle.str2bool(params[key][0]))),
                       'default': candle.str2bool(params[key][0]),
                       'help': params[key][1]
                       }
        else:
            new_def = {'name': key,
                       'type': (type(params[key][0])),
                       'default': params[key][0],
                       'help': params[key][1]
                       }
        new_defs.append(new_def)

    return new_defs


def construct_improve_dir_path(dir_name, dir_path, params):
    """ Custom function to construct directory paths in IMPROVE
    """
    new_key = dir_name
    old_key = dir_name + '_name'
    new_val = dir_path + '/' + params[old_key]
    print("Appending key:", new_key, new_val)

    params[new_key] = Path(new_val)

    return params


def construct_improve_file_path(file_name, dir_path, suffix, new_suffix, value, params):
    """ Custom function to construct file paths in IMPROVE
        Given a dictionary and a key name, remove the suffix
        and generate a new key with a new suffix appended.
    """
    file = remove_suffix(file_name, suffix)
    file = file + new_suffix
    params[file] = Path(dir_path / value)

    return params


def add_improve_key(key_name, old_suffix, new_suffix, params):
    """ Custom function to construct file paths in IMPROVE
        Given a dictionary and a key name, remove the suffix
        and generate a new key with a new suffix appended.
    """
    new_key = remove_suffix(key_name, old_suffix)
    new_key = new_key + new_suffix
    params[new_key] = params[key_name]

    return params


def build_improve_paths(params):

    # special cases -- no point automating
    params = construct_improve_dir_path("raw_data_dir", "main_data_dir", params)
    params = construct_improve_dir_path("ml_data_dir", "main_data_dir", params)
    params = construct_improve_dir_path("models_dir", "main_data_dir", params)
    params = construct_improve_dir_path("infer_dir", "main_data_dir", params)

    params = construct_improve_dir_path("x_data_dir", "raw_data_dir", params)
    params = construct_improve_dir_path("y_data_dir", "raw_data_dir", params)
    params = construct_improve_dir_path("splits_dir", "raw_data_dir", params)

    dir_path = params["x_data_dir"]
    # loop over cancer features
    new_dict = {}
    for k in params:
        if k.endswith('_fname'):
            # <k>_file_path = dir_path + '/' + <k>-'_fname'
            new_dict = construct_improve_file_path(k, dir_path, '_fname', '_file_path',
                                                   params[k], new_dict)
    # loop over drug features
    for k in params:
        if k.endswith('_file_name'):
            # <k>_file_path = dir_path + '/' + <k>-'_flle_name'
            new_dict = construct_improve_file_path(k, dir_path, '_file_name', '_file_path',
                                                   params[k], new_dict)

    params.update(new_dict)

    return params


class ImproveBenchmark(candle.Benchmark):

    def set_locals(self):
        """ Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        print('Additional definitions built from json files')
        additional_definitions = parser_from_json("candle_improve.json")
        print(additional_definitions, flush=True)
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


def initialize_parameters():
    # Build agent object

    driver = ImproveBenchmark(file_path, 'dummy.txt', 'keras',
                              prog='CANDLE_example', desc='CANDLE example driver script')

    # Initialize parameters
    gParameters = candle.finalize_parameters(driver)
    # benchmark.logger.info('Params: {}'.format(gParameters))
    run_params = gParameters

    return run_params


def main():
    print("Running main")
    params = initialize_parameters()
    params = build_improve_paths(params)
    print("After building paths")
    pprint(params)


if __name__ == "__main__":
    main()
