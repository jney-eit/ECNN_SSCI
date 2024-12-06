import os
import sys

import shutil
import numpy as np
import torch
import torch.nn as nn


def weight_reset(layer):
    """
    Reset parameters of layer
    :param layer: layer
    :return:
    """
    if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
        layer.reset_parameters()


def save_model(model, optim, valid_bers, simulation_params, path, name):
    """
    Save model to pt file
    :param model: trained model
    :param optim: optimizer
    :param valid_bers: list of validation bers
    :param simulation_params:
    :param path: Path to save model to
    :param name: Name of output file
    :return: Name of output file
    """

    if not os.path.isdir(path):
        print('Directory {} does not exist! Saving to local directory {}.'.format(path, os.path.abspath(os.getcwd())))
        path = ''
    cnt = 0
    file = os.path.join(path, '{}_{:0>2d}.pt'.format(name, cnt))
    while os.path.exists(file):
        cnt += 1
        file = os.path.join(path, '{}_{:0>2d}.pt'.format(name, cnt))
    checkpoint = {
        'model_sd': model.state_dict(),
        'optimizer_sd': optim.state_dict(),
        'valid_bers': valid_bers,
        'sim_params':simulation_params
    }
    torch.save(checkpoint, file)
    return file


def load_model(file):
    """
    Load model from file
    :param file: Path of file
    :return: Loaded model
    """
    if not os.path.exists(file):
        raise ValueError('file ({})) does not exist!'.format(file))
    checkpoint = torch.load(file)
    return checkpoint


def count_parameters(model):
    """
    Count all trainable parameters of model
    :param model: model
    :return: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def decision(x, constellation):
    """
    Choose class for predictions and given constellation
    :param x: Predictions
    :param constellation: Constellation
    :return: Decisions
    """

    # dist = torch.abs(constellation.view(1, 1, -1).expand(x.shape[0], -1, -1) - x.view(x.shape[0], -1, 1))
    # dec = torch.argmin(dist, 2)
    dist = torch.abs(constellation.view(1, 1, -1).expand(x.shape[0], -1, -1) - x.view(x.shape[0], -1, 1))
    # dist.shape = (batch_size, features, len(constellation))
    dec = torch.argmin(dist, 2)
    # dec.shape = (batch_size, features)
    # print("dec.shape = {} | dist.shape = {}".format(dec.shape, dist.shape))
    return dec



def normalize(input):
    """
    Normalize torch input to zero mean and unit variance
    :param input: Input
    :return: Normalized input
    """
    std_in = torch.std(input)
    if std_in != 0:
        x = (input - torch.mean(input)) / std_in
    else:
        x = (input - torch.mean(input))
    return x


def normalize_np(input):
    """
    Normalize numpy input to zero mean and unit variance
    :param input: Input
    :return: Normalized input
    """

    std_in = np.std(input)
    if std_in != 0:
        x = (input - np.mean(input)) / std_in
    else:
        x = (input - np.mean(input))
    return x


def get_next_unused_filename(path, model_type):
    """
    Get filename in directory which does not exist yet
    :param path: Directory path
    :param model_type: model type, used as name for file
    :return:
    """

    cnt = 0
    check = os.path.join(path, 'EQ_{}_{:0>4d}.pt'.format(model_type, cnt))
    while os.path.exists(check):
        cnt += 1
        check = os.path.join(path, 'EQ_{}_{:0>4d}.pt'.format(model_type, cnt))

    return check, 'EQ_{}_{:0>4d}.pt'.format(model_type, cnt)


def print_num_of_trainable_params(model, model_type):
    """
    Print number of trainable parameters of model
    :param model: Model
    :param model_type: Type of model
    :return:
    """
    print('============ {} ============='.format(model_type))
    print('Trainable parameters in model: %d' % count_parameters(model))
    print('===================================')


class Logger(object):
    """
    Class do duplicate output to file
    """

    def __init__(self, file, err=False):
        if err is True:
            self.terminal = sys.stderr
        else:
            self.terminal = sys.stdout
        self.log = open(file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        self.log.flush()
        pass



def extract_element(lst, indices):
    """
    Extract element at specified indices
    :param lst:
    :param indices:
    :return:
    """

    # Unpack the indices from the tuple
    i, *rest = indices
    # If there are more indices, recurse to the next level of the list
    if rest:
        return extract_element(lst[i], rest)
    # If there are no more indices, return the element at the current index
    return lst[i]


def print_dict_info(dict_obj, dict_name):
    print(f"Dictionary name: {dict_name}")
    for key, value in dict_obj.items():
        print(f"Key: {key}, Value: {value}")


def create_dict_structure(shape):
    # Recursive function to handle arbitrary dimensions
    def build_level(dimensions):
        if len(dimensions) == 1:
            return [{} for _ in range(dimensions[0])]
        else:
            return [build_level(dimensions[1:]) for _ in range(dimensions[0])]

    return build_level(list(shape))


def create_output_dirs(parameters):
    """
    Create directory to store output of training runs
    :param parameters: Dictionary of parameters
    :return:
    """

    output_files_path = os.path.join('output_files', parameters["run_name"])
    output_model_path = os.path.join('trained_models', parameters["run_name"])
    parameters["output_files_path"] = output_files_path
    parameters["output_model_path"] = output_model_path

    if not os.path.exists(parameters["output_files_path"]):
        os.makedirs(parameters["output_files_path"])

    if not os.path.exists(os.path.join(parameters["output_files_path"], "tmp")):
        os.makedirs(os.path.join(parameters["output_files_path"], "tmp"))

    if not os.path.exists(os.path.join(parameters["output_files_path"], "tmp_kernel")):
        os.makedirs(os.path.join(parameters["output_files_path"], "tmp_kernel"))

    if not os.path.exists(os.path.join(parameters["output_files_path"], "trainer")):
        os.makedirs(os.path.join(parameters["output_files_path"], "trainer"))

    if not os.path.exists(parameters["output_model_path"]):
        os.makedirs(parameters["output_model_path"])


def copy_train_files(parameters, params_path):
    """
    Copy files used for training to output directories to be able to reproduce results
    :param parameters: Dictionary of parameters
    :param params_path:
    :return:
    """

    # Copy relevant files for training to reproduce results later
    shutil.copyfile("custom_dataset.py", os.path.join(parameters["output_files_path"], "custom_dataset.py"))
    shutil.copyfile("tester.py", os.path.join(parameters["output_files_path"], "tester.py"))
    shutil.copyfile(params_path, os.path.join(parameters["output_files_path"], sys.argv[1]))
    shutil.copyfile("trainer/trainer_base.py", os.path.join(parameters["output_files_path"], "trainer/trainer_base.py"))
    shutil.copyfile("trainer/trainer_superwise.py", os.path.join(parameters["output_files_path"],
                                                                 "trainer/trainer_superwise.py"))
    shutil.copyfile("model.py", os.path.join(parameters["output_files_path"], "model.py"))
    shutil.copyfile("custom_dataset.py", os.path.join(parameters["output_files_path"], "custom_dataset.py"))
    shutil.copyfile(params_path, os.path.join(parameters["output_files_path"], sys.argv[1]))
    shutil.copyfile("sweeper.py", os.path.join(parameters["output_files_path"], "sweeper.py"))


def write_constexpr(file, datatype, name, value, num_tabs=70):
    """
    Write c++ constant expression of arbitrary data type
    """
    file.write(str("constexpr " + datatype + " " + name).ljust(num_tabs) + "= " + str(value) + ";\n")



def sample_wise_normalization(inputs):
    eps = 1e-8

    # Check if inputs are numpy array or torch tensor
    if isinstance(inputs, np.ndarray):
        mean_rowwise = np.mean(inputs, axis=1)[:, None]
        std_rowwise = np.std(inputs, axis=1)[:, None]
        std_rowwise[std_rowwise == 0] = eps
        inputs_norm = (inputs - mean_rowwise) / std_rowwise
    elif isinstance(inputs, torch.Tensor):
        mean_rowwise = torch.mean(inputs, dim=1).view(-1, 1)
        std_rowwise = torch.std(inputs, dim=1).view(-1, 1)
        std_rowwise[std_rowwise == 0] = eps
        inputs_norm = (inputs - mean_rowwise) / std_rowwise
    else:
        raise ValueError("Inputs must be numpy array or torch tensor")

    return inputs_norm
