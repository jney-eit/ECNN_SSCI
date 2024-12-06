import csv
import os

import numpy as np
import torch
import trainer.trainer_base
from trainer.trainer_threshold import Threshold_Trainer
import time
from torch.utils.data import DataLoader

from model import CNN_1D
from visualization import plot_param_sweep

class Tester():
    """
    Class to run several test and generate hardware of trained model
    """

    def __init__(self, params, tr, device, file_prefix=''):
        """
        Init tester class
        :param params: Dictionary of parameters
        :param tr: Trainer object
        :param device: cpu or gpu
        :param file_prefix: prefix of file to store
        """

        # Set all params as instance variables
        for key, value in params.items():
            setattr(self, key, value)

        self.params = params
        self.trainer = tr

        # For comparing to hw we use a similar loader with batch size one
        if tr.test_pump_generalization:
            self.selected_pumps_bs1_loader = torch.utils.data.DataLoader(tr.dataset, batch_size=1,
                                                                     sampler=tr.dataset.selected_pumps_sampler)

        self.device = device
        self.track_loss_intervall = max(self.training_iters // 100, 10)
        self.file_prefix = file_prefix


    def read_external_params(self):

        external_params_dict = {}

        with open(self.external_pump_param_path, mode='r') as file:
            reader = csv.reader(file, delimiter=';')
            for row in reader:
                pump_idx = int(row[0])
                value = row[1]
                external_params_dict[pump_idx] = float(value)

        return external_params_dict


    def generate_input_factors(self, test_combined_hw=False):

        if test_combined_hw == False:
            # Generate input factors < 1
            in_factors_below_one = [1 / x for x in reversed(range(2, 42, 4))]

            # Generate input factors between 0.6 and 2
            in_factors_mid_range = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1,
                             1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]

            # Generate input factors >= 2
            in_factors_above_one = list(range(2, 42, 4))

            in_factors = in_factors_below_one + in_factors_mid_range + in_factors_above_one
        else:
            in_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                                        2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        return in_factors

    def eval_additional_input_fac(self, test_combined_hw=False):
        """
        Evaluate different add_input_facs for trained network and find best one
        :return:
        """

        if not os.path.exists(os.path.join(self.output_files_path, "eval_add_input_fac")):
            os.makedirs(os.path.join(self.output_files_path, "eval_add_input_fac"))

        # Combine all lists
        add_input_fac_list = self.generate_input_factors(test_combined_hw)

        acc_list, tp_list, fp_list, tn_list, fn_list  = [], [], [], [], []

        results = {}
        max_acc = -1

        # Get optimal add_input_fac and add input fac based on external pump param
        for idx, add_input_fac in enumerate(add_input_fac_list):
            acc, tp, fp, tn, fn = self.trainer.evaluate(self.trainer.selected_pumps_loader,
                                                                self.trainer.dataset.classes.to(self.device),
                                                                max_iterations=20,
                                                                additional_input_fac=add_input_fac)[0:5]

            acc_list.append(acc)
            tp_list.append(tp)
            fp_list.append(fp)
            tn_list.append(tn)
            fn_list.append(fn)

            if add_input_fac == 1:
                results["add_input_fac_1"] = {"param": 1, "acc": acc, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

            if acc > max_acc:
                max_acc = acc
                results["add_input_fac_best"] = {"param": add_input_fac, "acc": acc, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

            print('Add input fac: {:4f} | selected pumps acc: {:.5f} | tp_selected_pump: {} | fp_selected_pump: {} | tn_selected_pump: {} | fn_selected_pump: {}'.format(
                    add_input_fac, acc, tp, fp, tn, fn), flush=True)

        fpr_best = 1
        fpr_based_add_input_fac_is_set = False

        fpr_loader = self.selected_pumps_bs1_loader if test_combined_hw is True else self.trainer.selected_pumps_loader

        # Get fpr-based add-input fac
        for idx, add_input_fac in enumerate(reversed(add_input_fac_list)):
            acc, tp, fp, tn, fn = self.trainer.evaluate(fpr_loader,
                                                                self.trainer.dataset.classes.to(self.device),
                                                                max_iterations=50,
                                                                additional_input_fac=add_input_fac)[0:5]

            # Get add_input_fac based on rise of false positive rate
            fpr = fp / (fp + tn) if fp + tn != 0 else 0

            # If false positive rate is < 10 % we set the fpr
            if fpr_based_add_input_fac_is_set is False and fpr < self.params["add_input_fac_fpr_threshold"]:
                results["add_input_fac_fpr"] = {"param":  add_input_fac, "acc": acc, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
                fpr_best = fpr

                fpr_based_add_input_fac_is_set = True

            print('Add input fac: {:4f} | selected pumps acc: {:.4f} | fpr: {:.4f}'.format(
                    add_input_fac, acc, fpr), flush=True)


        # If no good input factor is found, we set to highest input factor
        if not fpr_based_add_input_fac_is_set:
            add_input_fac = add_input_fac_list[-1]
            acc, tp, fp, tn, fn = self.trainer.evaluate(fpr_loader,
                                                                self.trainer.dataset.classes.to(self.device),
                                                                max_iterations=50,
                                                                additional_input_fac=add_input_fac)[0:5]

            # Get add_input_fac based on rise of false positive rate
            fpr = fp / (fp + tn) if fp + tn != 0 else 0
            fpr_best = fpr
            results["add_input_fac_fpr"] = {"param": add_input_fac, "acc": acc, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

        # Check if fpr is low enough
        results["ecnn_successfull"] = fpr_best <= 0.5

        plot_param_sweep(add_input_fac_list, acc_list, tp_list, fp_list, tn_list, fn_list, self.output_files_path + "/eval_add_input_fac/eval_add_input_fac_pump{}.png".format(self.trainer.dataset.selected_pump_indexes))

        def print_results(res, name):
            print('{} acc: {:.5f} with input fac: {:.3f} | tp_selected_pump: {} | fp_selected_pump: {} | '
                  'tn_selected_pump: {} | fn_selected_pump: {}'.format(name, res["acc"], res["param"], res["tp"], res["fp"], res["tn"], res["fn"]), flush=True)

        print_results(results["add_input_fac_best"], "add_input_fac_best")
        print_results(results["add_input_fac_fpr"], "add_input_fac_fpr")
        print_results(results["add_input_fac_1"], "add_input_fac_1")

        return results


    def eval_thresholds(self, test_combined_hw=False):
        """
        Evaluate different threshold for threshold-based algorithm and find best one
        """

        if not os.path.exists(os.path.join(self.output_files_path, "eval_t")):
            os.makedirs(os.path.join(self.output_files_path, "eval_t"))

        results = {}

        if test_combined_hw:
            threshold_trainer = Threshold_Trainer(self.params, self.device, self.trainer.dataset)
        else:
            threshold_trainer = self.trainer

        # add results for training threshold
        if not test_combined_hw:
            # Calculate results for training-based threshold
            acc, tp, fp, tn, fn = threshold_trainer.evaluate(threshold_trainer.selected_pumps_loader,
                                                        threshold_trainer.dataset.classes.to(self.device),
                                                        max_iterations=20)[0:5]

            results["t_train"] = {'param': threshold_trainer.t_train, 'acc': acc, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

            # Calculate results for optimal threshold
            t_opt_acc, tp, fp, tn, fn, t_opt = threshold_trainer.evaluate(threshold_trainer.selected_pumps_loader,
                                                                     threshold_trainer.dataset.classes.to(self.device),
                                                                     do_use_opt_t=True,
                                                                     max_iterations=20)[0:6]

            # add results for optimal threshold
            results["t_best"] = {'param': t_opt, 'acc': t_opt_acc, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

        # Calculate fpr-based results
        if test_combined_hw is False:
            t_list = np.arange(20, 0, -0.2).tolist()
        else:
            t_list = list(reversed([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
                                        7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15, 16, 17, 18, 19, 20]))

        acc_list, tp_list, fp_list, tn_list, fn_list  = [], [], [], [], []

        fpr_based_t_is_set = False
        fpr_loader = self.selected_pumps_bs1_loader if test_combined_hw is True else threshold_trainer.selected_pumps_loader

        for idx, t in enumerate(t_list):
            acc, tp, fp, tn, fn = threshold_trainer.evaluate(fpr_loader, threshold_trainer.dataset.classes.to(self.device), t_eval=t, max_iterations=20)[0:5]

            acc_list.append(acc)
            tp_list.append(tp)
            fp_list.append(fp)
            tn_list.append(tn)
            fn_list.append(fn)

            # Get threshold based on rise of false positive rate
            fpr = fp / (fp + tn) if fp + tn != 0 else 0

            # If false positive rate is larger than fpr_threshold, assume previous value was a good threshold
            if fpr_based_t_is_set is False and fpr > self.params["t_fpr_threshold"]:
                if idx >= 1:
                    results["t_fpr"] = {"param":  t_list[idx - 1], "acc": acc_list[idx - 1], 'tp': tp_list[idx - 1], 'fp': fp_list[idx - 1], 'tn': tn_list[idx - 1], 'fn': tn_list[idx - 1]}
                else:
                    results["t_fpr"] = {"param":  t, "acc": acc, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
                fpr_based_t_is_set = True

        if not fpr_based_t_is_set:
            t = t_list[0]
            acc, tp, fp, tn, fn = threshold_trainer.evaluate(fpr_loader, threshold_trainer.dataset.classes.to(self.device), t_eval=t, max_iterations=20)[0:5]
            results["t_fpr"] = {"param": t, "acc": acc, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

        plot_param_sweep(t_list, acc_list, tp_list, fp_list, tn_list, fn_list, self.output_files_path + "/eval_t/eval_t_pump{}.png".format(threshold_trainer.dataset.selected_pump_indexes))

        def print_results(res, name):
            print('{} acc: {:.5f} with threshold: {:.3f} | tp_selected_pump: {} | fp_selected_pump: {} | '
                  'tn_selected_pump: {} | fn_selected_pump: {}'.format(name, res["acc"], res["param"], res["tp"], res["fp"], res["tn"], res["fn"]), flush=True)

        if test_combined_hw is False:
            print_results(results["t_best"], "t_best")
            print_results(results["t_train"], "t_train")
        print_results(results["t_fpr"], "t_fpr")

        return results


    def test(self):
        """
        Evaluate accuracy, true positive, false positive, true negative and false negative of model
        :return:
        """
        print("self.trainer.selected_pumps_loader: {}".format(self.trainer.selected_pumps_loader))
        if self.trainer.selected_pumps_loader is not None:
            print("Testing model for pump with index {}...".format(self.trainer.dataset.selected_pump_indexes))

            if self.model_type == "CNN_1D" and self.additional_input == "mean" and self.test_combined_hw:
                print("Testing combined hardware")
                ecnn_results = self.eval_additional_input_fac(True)
                if ecnn_results["ecnn_successfull"] == False:
                    print("ECNN performance not sufficient, using threshold")
                    threshold_results = self.eval_thresholds(True)
                    results = threshold_results
                else:
                    print("ECNN performance is sufficient")
                    results = ecnn_results
            else:
                if self.trainer.additional_input == "mean" or self.trainer.additional_input == "max" or self.trainer.subtract_mean_from_input is True:
                    results = self.eval_additional_input_fac()

                elif self.trainer.training_mode == "mean" or self.trainer.training_mode == "max_ampl":
                    results = self.eval_thresholds()

                else:
                    sp_eval_acc, tp, fp, tn, fn = self.trainer.evaluate(self.trainer.selected_pumps_loader,
                                              self.trainer.dataset.classes.to(self.device),
                                              max_iterations=20)[0:5]

                    print('selected pumps acc: {:.5f} | tp_selected_pump: {} | fp_selected_pump: {} | tn_selected_pump: {} | fn_selected_pump: {}'.format(sp_eval_acc, tp, fp, tn, fn), flush=True)

                    # Init results dict with all values
                    results = {"eval_acc": sp_eval_acc, "tp": tp, "fp": fp, "tn": tn, "fn": fn}
        else:
            eval_acc, tp, fp, tn, fn = self.trainer.evaluate(self.trainer.eval_loader, self.trainer.dataset.classes.to(self.device), max_iterations=40)[0:5]
            print('eval acc: {:.5f} | tp: {} | fp: {} | tn: {} | fn: {}'.format(eval_acc, tp, fp, tn, fn), flush=True)

            results = {"eval_acc": eval_acc, "tp": tp, "fp": fp, "tn": tn, "fn": fn}

        return results


    def export_model_to_onnx(self):

        fp_model = CNN_1D(1, depth=self.params["depth"], device=self.device, channels_in=self.params["channels_in"],
                          kernel_size=self.params["kernel_size"], max_channels=self.params["intermediate_channels"], in_sequence_len=self.params["input_symbols"],
                          use_bias=self.params["use_bias"], last_layer_type=self.params["last_layer_type"])

        if self.trainer.quantize:
            for idx, layer in enumerate(self.trainer.model.conv1d_layers):
                fp_model.conv1d_layers[idx].weight = torch.nn.Parameter(layer.q_weight)
                fp_model.conv1d_layers[idx].bias = torch.nn.Parameter(layer.q_bias)

        dummy_input = torch.randn(1, self.params["channels_in"], self.params["input_symbols"], device=self.device)
        torch.onnx.export(fp_model, dummy_input, "sipsensin-cnn.onnx", verbose=True)
        return fp_model

    @staticmethod
    def load_from_trainer_file(trainer_file, device, dataset, test_params=None):
        """
        Init tester based on trainer file
        :param trainer_file: path of trainer file
        :param device: cpu or gpu
        :param dataset: dataset
        :return: tester object
        """

        if not os.path.exists(trainer_file):
            raise ValueError('file ({})) does not exist!'.format(trainer_file))

        # Load Trainer
        tr = trainer.trainer_base.Trainer_Base.load_state(trainer_file, device, dataset)



        # Instantiate Tester from Trainer
        parameters = vars(tr)

        def set_test_param(key):
            if key in test_params:
                parameters[key] = test_params[key]

        # If we test only we want to sweep over specific test params 
        if test_params is not None:
            set_test_param("add_input_fac_fpr_threshold")
            set_test_param("t_fpr_threshold")
            set_test_param("test_external_pump_param")
            set_test_param("external_pump_param_path")

        te = Tester(parameters, tr, device, file_prefix=parameters["model_type"])

        return te



