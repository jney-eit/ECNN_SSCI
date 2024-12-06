import os
import pickle
import numpy as np

import utils
import visualization as vis
import csv
import itertools
from registry import train_class_dict
import collections
from tester import Tester
from custom_dataset import CustomDataset, get_num_pumps_in_parquet_dataset
from functools import partial

def train_sweep(params, device):
    """
    Wrapper function which allows to easily train models on a parameter sweep without having to deal with the Sweeper class directly.
    """

    sw = Sweeper(params, device)
    print("#################### PARAMETER ({}) SWEEP - TRAINING STARTED ####################".format(sw.sweep['parameter']))
    sw.train_models()
    print("#################### PARAMETER SWEEP - TRAINING DONE ####################")
    print("#################### PARAMETER SWEEP - EVALUATION STARTED ####################")
    sw.evaluate_sweep()
    print("#################### PARAMETER SWEEP - EVALUATION DONE ####################")
    print("#################### PARAMETER SWEEP - SAVING RESULTS... ####################")
    sw.save_results()
    print("#################### PARAMETER SWEEP - SAVING DONE ####################")
    return sw.results


def test_sweep(params, device):
    """
    Wrapper function which allows to easily train models on a parameter sweep without having to deal with the Sweeper class directly.
    """

    sw = Sweeper(params, device, test_only=True)
    sw.instantiate_children()
    print("#################### PARAMETER SWEEP - EVALUATION STARTED ####################")
    sw.evaluate_sweep()
    print("#################### PARAMETER SWEEP - EVALUATION DONE ####################")
    print("#################### PARAMETER SWEEP - SAVING RESULTS... ####################")
    sw.save_results()
    print("#################### PARAMETER SWEEP - SAVING DONE ####################")
    return sw.results


class Sweeper:
    def __init__(self, params, device, results=None, test_only=False):
        """
        Init Sweeper
        :param params: Dictionary of all parameters
        :param device: cpu or gpu
        :param results: Results of previous run?
        """

        # Set sweep parameters of yaml file
        self.sweep = params['sweep']

        if test_only is True: 
            self.test_params = params
        else:
            self.test_params = None

        if results is None:
            # Set default parameters for instantiation of Trainers/Testers later
            self.default_params = params
        else:
            self.default_params = results['params']

        self.device = device
        do_calc_max_abs_val = params["additional_input"] == "max"
        do_calc_mean_val = params["training_mode"] == "mean" or params["additional_input"] == "mean" or ("subtract_mean_from_input" in params and params["subtract_mean_from_input"] is True)

        # If we do not sweep over different eval pumps, we can use same dataset for all sweeps
        if "pump_eval_idx" not in self.sweep["parameter"]:
            self.dataset = CustomDataset(params["input_x_data_path"], params["input_y_data_path"], params["input_z_data_path"],
                                         params["gt_data_path"],
                                         device=device,
                                         training_mode=params["training_mode"],
                                         test_pump_generalization=params["test_pump_generalization"],
                                         pump_eval_indexes=params["pump_eval_idx"],
                                         equally_distributed_train_data=params["equally_distributed_train_data"],
                                         do_calc_mean_val=do_calc_mean_val,
                                         do_remove_pumps_with_only_one_class=params["do_remove_pumps_with_only_one_class"])
        else:
            self.CustomDataset_parameterized = partial(CustomDataset,
                                                       input_x_data_path=params["input_x_data_path"],
                                                       input_y_data_path=params["input_y_data_path"],
                                                       input_z_data_path=params["input_z_data_path"],
                                                       gt_data_path=params["gt_data_path"],
                                                       device=device,
                                                       training_mode=params["training_mode"],
                                                       test_pump_generalization=params["test_pump_generalization"],
                                                       equally_distributed_train_data=params["equally_distributed_train_data"],
                                                       do_calc_mean_val=do_calc_mean_val,
                                                       do_remove_pumps_with_only_one_class=params["do_remove_pumps_with_only_one_class"])

            param_idx_pump_eval_idx = self.sweep["parameter"].index("pump_eval_idx")
            value_pump_eval_idx = self.sweep["values"][param_idx_pump_eval_idx][0]
            if value_pump_eval_idx == "all":
                print("Performing cross validation for all pumps in dataset.")
                # If pump_eval_idx is set to all, we neet to get all pump indices based on length of dataset
                num_pumps_in_dataset = get_num_pumps_in_parquet_dataset(params["input_x_data_path"])
                pump_indices = list(range(num_pumps_in_dataset))
                self.sweep["values"][param_idx_pump_eval_idx] = pump_indices

        training_mode = params["training_mode"]
        model_type = params["model_type"]

        # Select trainer class
        if (training_mode, model_type) in train_class_dict:
            self.trainer_type = train_class_dict[(training_mode, model_type)]
        else:
            raise ValueError("training mode {} not valid with model type {}".format(training_mode, model_type))

        # Calculate the number of dimensions of the sweep
        self.num_dims = len(self.sweep['parameter'])

        # Create a list of indices for each dimension of the sweep
        self.dim_indices = [range(len(self.sweep['values'][i])) for i in range(self.num_dims)]

        # Generate all possible combinations of indices for the sweep dimensions
        self.combinations = itertools.product(*self.dim_indices)
        self.combinations_list = list(itertools.product(*self.dim_indices))

        # Get total number of different model configurations
        self.total_num_model_configs = 1
        for num_values in [len(values) for values in self.sweep['values']]:
            self.total_num_model_configs *= num_values

        # Gen tuple of shape for model in all dimensions
        # Eg. with three parameters, each with two values and 4 runs: (2, 2, 2, 4)
        self.all_models_shape = tuple([len(values) for values in self.sweep['values']] + [self.default_params['training_runs']])

        #print("all_models_shape: {}".format(self.all_models_shape))
        #print("len(all_models_shape): {}".format(len(self.all_models_shape)))
        #exit(10)

        if results:
            self.parents = results['parents']
            self.children = results['children']
            # print("results['children'] = {}".format(results['children']))

        # Sweep evaluation results
        self.results = results


    def train_models(self, save_results=True):
        """
        Train all models of sweeper
        :param save_results: Plot and save validation accuracies
        :return:
        """

        self.results = None

        # self.children = np.full(self.all_models_shape, "", dtype=object)
        self.children = np.full((self.total_num_model_configs, self.default_params['training_runs']), "", dtype=object)

        validation_accuracies = []
        average_num_bits_iterations = []
        averag_num_bits = []
        average_num_bits_per_layer = []
        test_accuracies = []

        params = self.default_params.copy()

        # Iterate over all combinations of indices
        for i, index_combination in enumerate(self.combinations):

            # Set the parameter values for the current combination of indices
            model_file_out = 'EQ_{}'.format(params["model_type"])
            for param_idx, param in enumerate(self.sweep['parameter']):
                param_value = self.sweep['values'][param_idx][index_combination[param_idx]]
                model_file_out += "_{}{}".format(param, param_value)
                params[param] = param_value

                if params["test_pump_generalization"] and param == "pump_eval_idx":
                    print("Loading dataset with pump_eval_indexes={}...".format(param_value))
                    self.dataset = self.CustomDataset_parameterized(pump_eval_indexes=param_value)

            validation_accuracies_current_sweep = []

            current_sweep_values = [str(self.sweep['values'][dim][self.combinations_list[i][dim]]) for dim in range(self.num_dims)]
            config_str_list = [param + "=" + value for param, value in zip(self.sweep["parameter"], current_sweep_values)]
            conifg_str = ', '.join(config_str_list)
            print("#################### Parameter configuration {}/{}: {} ####################".format(i+1, self.total_num_model_configs, conifg_str))

            test_accuracy_current = []
            averag_num_bits_current = []
            average_num_bits_iterations_current = []
            average_num_bits_per_layer_current = []

            for j in range(self.default_params['training_runs']):
                print("#################### Starting training run number {}/{} with parameter set {}/{}  ####################".format(j+1, self.default_params['training_runs'], i+1, self.total_num_model_configs))

                # Instantiate a trainer with the current parameter values
                trainer = self.trainer_type(params, self.device, self.dataset)

                current_filename = model_file_out + '_{:0>4d}'.format(j)
                model_file_out_path = os.path.join(self.default_params["output_model_path"], current_filename + '.pt')

                print('current_filename: {}'.format(current_filename))
                trainer_ret = trainer.train()
                validation_accuracies_current_sweep.append(trainer_ret[1])

                if self.default_params["quantize"]:
                    average_num_bits_iterations_current.append(trainer_ret[3])
                    averag_num_bits_current.append(trainer_ret[0].get_average_num_bits_model())
                    average_num_bits_per_layer_current.append(trainer_ret[0].get_average_num_bits_per_layer())

                    tester = Tester(self.default_params, trainer, self.device, file_prefix=self.default_params["model_type"])
                    test_results = tester.test()

                    nn_has_adjustable_hyperparam = trainer.additional_input == "mean" or trainer.additional_input == "max" or trainer.subtract_mean_from_input is True
                    threshold_alg_is_used = trainer.training_mode == "mean" or trainer.training_mode == "max_ampl"

                    if nn_has_adjustable_hyperparam:
                        test_accuracy_current =  test_results["add_input_fac_best"]["acc"]
                    elif threshold_alg_is_used:
                        test_accuracy_current =  test_results["t_best"]["acc"]
                    else:
                        test_accuracy_current = test_results["acc"]

                trainer.freeze_state(dir=self.default_params["output_model_path"], filename=current_filename)
                self.children[i, j] = model_file_out_path

            if self.default_params["quantize"]:

                test_accuracies.append(test_accuracy_current)
                averag_num_bits.append(averag_num_bits_current)
                average_num_bits_iterations.append(average_num_bits_iterations_current)
                average_num_bits_per_layer.append(average_num_bits_per_layer_current)

            validation_accuracies.append(validation_accuracies_current_sweep)


        self.results = {
            'params': self.default_params,
            'default':  utils.create_dict_structure(self.all_models_shape),
            'param_best': utils.create_dict_structure(self.all_models_shape),
            'param_fixed': utils.create_dict_structure(self.all_models_shape),
            'param_fpr': utils.create_dict_structure(self.all_models_shape),
            'param_external': utils.create_dict_structure(self.all_models_shape),
            'children': self.children
        }

        if save_results:
            # Save validation bers
            if len(self.sweep['parameter']) == 1:
                if self.default_params["quantize"]:
                    vis.plot_average_num_bits_sweep(self.default_params, average_num_bits_iterations, self.default_params["output_files_path"])
                    vis.plot_acc_vs_bits_sweep(self.default_params, test_accuracies, averag_num_bits, self.default_params["output_files_path"])
                    vis.write_average_num_bits_per_layer(self.default_params, average_num_bits_per_layer, self.default_params["output_files_path"])

        return

    def instantiate_children(self):
        self.children = np.full((self.total_num_model_configs, self.default_params['training_runs']), "", dtype=object)
        params = self.default_params.copy()

        # Iterate over all combinations of indices
        for i, index_combination in enumerate(self.combinations):

            # Set the parameter values for the current combination of indices
            model_file_out = 'EQ_{}'.format(params["model_type"])
            for param_idx, param in enumerate(self.sweep['parameter']):
                param_value = self.sweep['values'][param_idx][index_combination[param_idx]]
                model_file_out += "_{}{}".format(param, param_value)
                params[param] = param_value

            for j in range(self.default_params['training_runs']):
                current_filename = model_file_out + '_{:0>4d}'.format(j)
                model_file_out_path = os.path.join(os.path.join('trained_models', params["load_model_path"]), current_filename + '.pt')

                self.children[i, j] = model_file_out_path


    def evaluate_sweep(self):
        """
        Evaluate all trained models
        :return:
        """

        self.results = {
            'params': self.default_params,
            'default':  utils.create_dict_structure(self.all_models_shape),
            'param_best': utils.create_dict_structure(self.all_models_shape),
            'param_fixed': utils.create_dict_structure(self.all_models_shape),
            'param_fpr': utils.create_dict_structure(self.all_models_shape),
            'param_external': utils.create_dict_structure(self.all_models_shape),
            'children': self.children
        }

        def update_and_print_results(key, results):
            utils.extract_element(self.results[key], index_combination)[j].update(results)
            utils.print_dict_info(results, key)

        for i, index_combination in enumerate(self.combinations_list):

            for param_idx, param in enumerate(self.sweep['parameter']):
                param_value = self.sweep['values'][param_idx][index_combination[param_idx]]

                if self.default_params["test_pump_generalization"] and param == "pump_eval_idx":
                    print("Loading dataset with pump_eval_indexes={}...".format(param_value))
                    self.dataset = self.CustomDataset_parameterized(pump_eval_indexes=param_value)


            for j in range(self.default_params['training_runs']):
                child_tester = Tester.load_from_trainer_file(self.children[i,j], self.device, self.dataset, test_params=self.test_params)

                nn_has_adjustable_hyperparam = child_tester.trainer.additional_input == "mean" or child_tester.trainer.additional_input == "max" or child_tester.trainer.subtract_mean_from_input is True
                threshold_alg_is_used = child_tester.trainer.training_mode == "mean" or child_tester.trainer.training_mode == "max_ampl"

                if self.default_params["test_pump_generalization"] and (nn_has_adjustable_hyperparam or threshold_alg_is_used):
                    test_results = child_tester.test()

                    if nn_has_adjustable_hyperparam:
                        results_best =  test_results["add_input_fac_best"]
                        results_fixed = test_results["add_input_fac_1"]
                        results_fpr = test_results["add_input_fac_fpr"]
                    else:
                        results_best = test_results["t_best"]
                        results_fixed = test_results["t_train"]
                        results_fpr = test_results["t_fpr"]

                    update_and_print_results('param_best', results_best)
                    update_and_print_results('param_fixed', results_fixed)
                    update_and_print_results('param_fpr', results_fpr)

                    if self.default_params["test_external_pump_param"] is True:
                        if nn_has_adjustable_hyperparam:
                            results_external = test_results["add_input_fac_external"]
                        else:
                            results_external = test_results["t_external"]

                        update_and_print_results('param_external', results_external)
                else:
                    test_ret = child_tester.test()
                    update_and_print_results('default', test_ret)


    @staticmethod
    def load_validation_bers(sweeper_file, device):
        """
        Loads all individual model files and validation_bers from training
        :param sweeper_file: Path of sweeper file
        :param device: cpu or gpu
        :return: tuple of bers and sweep parameters,  bers has shape: #evaluations_during_training, #sweep_configs, #models_per_config
        """

        # Load Sweep
        sw = Sweeper.load_sweep(sweeper_file, device)

        validation_bers = None
        training_iters = None
        for i in range(sw.children.shape[0]):
            for j in range(sw.children.shape[1]):
                model = sw.children[i,j]
                te = Tester.load_from_trainer_file(model, device)
                if validation_bers is None:
                    validation_bers = np.ones((len(te.validation_bers),sw.children.shape[0], sw.children.shape[1]))
                if training_iters is None:
                    training_iters = np.arange(len(te.validation_bers)) * te.track_loss_intervall
                validation_bers[:, i, j] = te.validation_bers
        return validation_bers, training_iters, sw.sweep


    def save_results_csv(self, name):
        """
        Save results of all models to csv file
        :param name: Name of sweep
        :return:
        """

        results_file = os.path.join(self.default_params["output_files_path"], 'results_{}.csv'.format(name))
        with open(results_file, 'w') as f:
            writer = csv.writer(f)

            nn_has_adjustable_hyperparam = self.default_params["additional_input"] == "mean" or self.default_params["additional_input"] == "max" or self.default_params["subtract_mean_from_input"] is True
            threshold_alg_is_used = self.default_params["training_mode"] == "mean" or self.default_params["training_mode"] == "max_ampl"

            # values: [[3, 4, 5], [9, 15, 21], [3, 4, 5]]
            for i, index_combination in enumerate(self.combinations_list):
                print("index_combination: {}".format(index_combination))
                current_sweep_params = [str(self.sweep['values'][dim][index_combination[dim]]) for dim in range(self.num_dims)]

                for j in range(self.default_params['training_runs']):

                    if self.default_params["test_pump_generalization"] and (nn_has_adjustable_hyperparam or threshold_alg_is_used):
                        results_best_dict =  utils.extract_element(self.results['param_best'], index_combination)[j]
                        results_fixed_dict = utils.extract_element(self.results['param_fixed'], index_combination)[j]
                        results_fpr_dict = utils.extract_element(self.results['param_fpr'], index_combination)[j]

                        # Write header
                        if i == 0 and j == 0:
                            dict_names = ["param_best", "param_fixed", "param_fpr"]

                            if self.default_params["test_external_pump_param"] is True:
                                dict_names += ["param_external"]

                            # Prepare the header
                            headers = []
                            sub_headers = []
                            headers += self.sweep["parameter"]
                            sub_headers += self.sweep["parameter"]
                            for name in dict_names:
                                headers += [name] + [''] * (len(results_best_dict) - 1)
                                sub_headers += list(results_best_dict.keys())

                            writer.writerow(headers)
                            writer.writerow(sub_headers)

                        current_row = current_sweep_params
                        current_row += [str(value) for value in results_best_dict.values()]
                        current_row += [str(value) for value in results_fixed_dict.values()]
                        current_row += [str(value) for value in results_fpr_dict.values()]

                        if self.default_params["test_external_pump_param"] is True:
                            results_external_dict = utils.extract_element(self.results['param_external'], index_combination)[j]
                            current_row += [str(value) for value in results_external_dict.values()]
                    else:
                        results_dict = utils.extract_element(self.results['default'], index_combination)[j]

                        # Write header
                        if i == 0 and j == 0:

                            # Prepare the header
                            header = []
                            header += self.sweep["parameter"]
                            header += list(results_dict.keys())
                            writer.writerow(header)

                        current_row = current_sweep_params + [str(value) for value in results_dict.values()]

                    writer.writerow(current_row)


    def save_results(self):
        """
        Save results of all models to pickle file and csv file
        :return:
        """
        name = os.path.split(self.default_params["output_files_path"])[-1]
        results_file = os.path.join(self.default_params["output_files_path"], 'results_{}.pickle'.format(name))
        with open(results_file, 'wb+') as f:
            pickle.dump(self.results, f)
        print("Writing parameter sweep results to {}.".format(results_file))

        # Sweeper.save_ber_csv(results, location, params, name)
        self.save_results_csv(name)

        return

    @staticmethod
    def load_results(filename):
        """
        Load results from pickle file
        :param filename: Path of pickle file
        :return: Loaded results
        """
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        return results

    @staticmethod
    def load_sweep(name, device):
        """
        Load old sweeper from file
        :param name: Name of sweeper
        :param device: cpu or gpu
        :return: Sweeper instance
        """
        filename = os.path.join("output_files", name, "results_{}.pickle".format(name))

        if not os.path.exists(filename):
            raise ValueError('file ({})) does not exist!'.format(filename))

        # Load results from file
        results = Sweeper.load_results(filename)

        # Instantiate Sweeper
        sw = Sweeper(results['params'], device, results=results)

        return sw


    @staticmethod
    def append_sweep(sw, sw_a, device):
        """
        I don't get it
        :param sw:
        :param sw_a: what's the difference to sw?
        :param device:
        :return:
        """
        if not sw.results['params']['model_type'] == sw_a.results['params']['model_type']:
            return None
        if not sw.results['params']['training_iters'] == sw_a.results['params']['training_iters']:
            return None
        if not sw.results['params']['sweep']['parameter'] == sw_a.results['params']['sweep']['parameter']:
            return None
        
        r = sw.results.copy()
        r['params']['sweep']['values'].extend(sw_a.results['params']['sweep']['values'])
        r['BER'] = np.vstack((r['BER'], sw_a.results['BER']))
        r['parents'].extend(sw_a.results['parents'])
        r['children'] = np.vstack((r['children'], sw_a.results['children']))

        # save appended results and create new directory, if needed
        run_name = 'appended_sweep'
        r['params']['sweep']['sweep_name'] = run_name
        # Check if run_name already exists
        cnt = 0
        check = os.path.join('output_files', '{}_{:0>4d}'.format(run_name, cnt))
        while os.path.exists(check):
            cnt += 1
            check = os.path.join('output_files', '{}_{:0>4d}'.format(run_name, cnt))
        r['params']['run_name'] = '{}_{:0>4d}'.format(run_name, cnt)
        # Make output directories
        output_files_path = os.path.join('output_files', r['params']['run_name'])
        if not os.path.exists(output_files_path):
            os.makedirs(output_files_path)
        r['params']['output_files_path'] = output_files_path
        # Save actual sweep results into new directory
        Sweeper.save_results(r, r['params']["output_files_path"])
        
        # Instantiate Sweeper from new results
        sw_new = Sweeper.load_sweep(r['params']['run_name'], device)

        return sw_new
    
        

