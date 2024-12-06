import os
import sys
import yaml
import torch
import visualization as vis
import sweeper
from trainer.trainer_base import Trainer_Base
from tester import Tester
from utils import Logger, create_output_dirs, copy_train_files
from custom_dataset import CustomDataset
import argparse


def setup_directories():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(file_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    print(f"sys.path updated: {sys.path}")
    os.chdir(file_dir)
    print(f"Changed working directory to: {os.getcwd()}")


def get_device():
    """Determine the best available device to run computations."""
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 0:
            curr_device = torch.cuda.current_device()
            print(f"Using CUDA device: {torch.cuda.get_device_name(curr_device)}")
            return torch.device('cuda')
        else:
            print("No CUDA devices are available.")
    else:
        print("CUDA is not available. Using CPU.")
    return torch.device('cpu')


def train_or_test_single_model(params, device):
    """
    Wrapper function which allows to easily train models with given parameters without having to deal with the Trainer class directly.
    """

    # Import subclasses here to avoid circular dependencies
    from registry import train_class_dict

    training_mode = params["training_mode"]
    model_type = params["model_type"]

    eval_accs_all_runs = []
    selected_pumps_accs_all_runs = []
    model_files = []

    do_calc_mean_val = params["training_mode"] == "mean" or params["additional_input"] == "mean" or ("subtract_mean_from_input" in params and params["subtract_mean_from_input"] is True)

    dataset = CustomDataset(params["dataset_path"],
                            device=device,
                            training_mode=params["training_mode"],
                            test_pump_generalization=params["test_pump_generalization"],
                            pump_eval_indexes=params["pump_eval_idx"],
                            equally_distributed_train_data=params["equally_distributed_train_data"],
                            do_calc_mean_val=do_calc_mean_val,
                            do_remove_pumps_with_only_one_class=params["do_remove_pumps_with_only_one_class"])

    if params["train_model"] is True:
        # Train new model

        for i in range(params['training_runs']):
            print("#################### Starting run number: {} ####################".format(i + 1))

            # Set output file name
            params['current_filename'] = 'EQ_{}.pt'.format(params["model_type"])
            model_file_out = os.path.join(params["output_model_path"], params['current_filename'])

            # Select trainer class
            if (training_mode, model_type) in train_class_dict:
                trainer = train_class_dict[(training_mode, model_type)](params, device, dataset)
            else:
                raise ValueError("training mode {} not valid with model type {}".format(training_mode, model_type))

            print("Trainer class used: {}".format(trainer.__class__.__name__))

            # train model
            trainer_ret = trainer.train()
            eval_accuracies = trainer_ret[1]
            selected_pumps_eval_accuracies = trainer_ret[2]

            eval_accs_all_runs.append(eval_accuracies)
            selected_pumps_accs_all_runs.append(selected_pumps_eval_accuracies)

            # save trainer and model
            trainer.freeze_state(dir=params["output_model_path"], filename=params["current_filename"])
            model_files.append(model_file_out)

        vis.plot_training_accuracies(eval_accs_all_runs, selected_pumps_accs_all_runs, params)
    else:
        # Load pretrained model
        trainer = Trainer_Base.load_state(os.path.join('trained_models', params["load_model_path"]), device, dataset, params["batch_size"])

    if params["test_model"] or params["generate_hardware_files"]:
        tester = Tester(params, trainer, device, file_prefix=params["model_type"])

    if params["test_model"]:
        if params["export_model_as_onnx"] is True:
            fp_model = tester.export_model_to_onnx()
            tester.trainer.model = fp_model

        tester.test()

    return eval_accs_all_runs, selected_pumps_accs_all_runs



def main():

    setup_directories()
    device = get_device()

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Process some parameters.')
    # Add positional argument for param_path
    parser.add_argument('param_path', type=str, help='Path to the parameters file')
    # Add optional argument for unroll_fac, which defines the parallelism of the hardware
    parser.add_argument('--unroll_fac', type=int, default=1, help='Unroll factor (default: 1)')

    # Parse the arguments
    args = parser.parse_args()

    # Open parameter file, used for configuration
    params_path = os.path.join("parameters/", args.param_path)

    with open(params_path, 'r') as params_file:
        parameters = yaml.safe_load(params_file)

    # Sweep parameter file present?
    if 'sweep' in parameters and parameters['sweep'] != 'None':
        do_sweep = True
        parameters['run_name'] = parameters['sweep']['sweep_name']
    else:
        do_sweep = False

    print("Current run: {}".format(parameters['run_name']))

    # Create output directories
    create_output_dirs(parameters)

    # Duplicate output to file
    sys.stdout = Logger(os.path.join(parameters["output_files_path"], "out.txt"))
    sys.stderr = Logger(os.path.join(parameters["output_files_path"], "err.txt"), err=True)

    # Copy relevant files to reproduce results
    copy_train_files(parameters, params_path)

    if do_sweep:
        print("Performing sweep...")
        if parameters["train_model"] is True:
            # Train on parameter sweep
            results = sweeper.train_sweep(parameters, device)
        elif parameters["test_model"] is True:
            results = sweeper.test_sweep(parameters, device)
        print('sweep_results = {}'.format(results))
    else:
        print("Train or test single model...")
        # Train with fixed parameters
        train_or_test_single_model(parameters, device)

    return


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(False)
    main()