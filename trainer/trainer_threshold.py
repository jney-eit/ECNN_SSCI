from trainer.trainer_base import Trainer_Base

import torch
import numpy as np

class Threshold_Trainer(Trainer_Base):
    """
    Trainer to train autoencoder model
    """

    def __init__(self, params, device, dataset):

        super(Threshold_Trainer, self).__init__(params, device, dataset)

        # Number of simulation samples
        self.N_sim = self.input_symbols
        # Threshold calculated based on training data
        self.t_train = 0


    def calc_optimal_threshold(self, losses_0, losses_1):
        losses_0_np = np.array(losses_0)
        losses_1_np = np.array(losses_1)

        class_0_labels = np.zeros(len(losses_0))
        class_1_labels = np.ones(len(losses_1))

        labels = np.concatenate((class_0_labels, class_1_labels))
        losses = np.concatenate((losses_0_np, losses_1_np))

        # Sort losses and labels
        sorted_indices = np.argsort(losses)
        losses = losses[sorted_indices]
        labels = labels[sorted_indices]

        # Calculate accuracy for all possible thresholds
        thresholds = np.unique(losses)
        acc = np.zeros(len(thresholds))

        for i, t in enumerate(thresholds):
            pred_labels = np.zeros(len(losses))
            pred_labels[losses > t] = 1
            acc[i] = np.sum(pred_labels == labels) / len(labels)

        # Find optimal threshold
        opt_index = np.argmax(acc)
        opt_threshold = thresholds[opt_index]

        return opt_threshold, acc[opt_index]

    # Function for optimal threshold claculation based on input sampels and gts
    def calc_optimal_threshold_from_data(self, input_samples, gt_samples):
        mse_0 = []
        mse_1 = []

        for b in range(input_samples[0].shape[0]):
            input_samples_mean_expanded = input_samples[1][b].unsqueeze(1).repeat(1, 800)
            mse = torch.mean(torch.square(input_samples_mean_expanded - input_samples[0][b]))

            if gt_samples[b] == self.dataset.classes[0]:
                mse_0.append(mse.cpu().detach().numpy())
            else:
                mse_1.append(mse.cpu().detach().numpy())

        # calc optimal threshold
        opt_threshold, opt_acc = self.calc_optimal_threshold(mse_0, mse_1)

        return opt_threshold, opt_acc



    def train(self, plot_loss_and_ber=True, retrain_out_path="", plt_files_extension=""):
        """
        Train generator in supervised fashion to check if complexity is sufficient
        """

        t_list = []
        opt_train_t_list = []

        it = 0
        while it < self.training_iters:
            for x, gt in self.train_loader:

                curr_opt_t, _ = self.calc_optimal_threshold_from_data(x, gt)
                opt_train_t_list.append(curr_opt_t)
                self.t_train = np.mean(opt_train_t_list)


                if (it + 1) % self.track_loss_intervall == 0:
                    # Compute validation BER

                    print("\n")
                    print("#############################################################################")
                    val_index = int(it // self.track_loss_intervall)
                    eval_iterations = 50

                    self.eval_accuracy[val_index], tp, fp, tn, fn = self.evaluate(self.eval_loader, self.dataset.classes.to(self.device), self.t_train, max_iterations=eval_iterations)

                    t_list.append(self.t_train)

                    if self.selected_pumps_loader:
                        self.selected_pumps_eval_accuracy[val_index] = self.evaluate(self.selected_pumps_loader, self.dataset.classes.to(self.device), self.t_train, max_iterations=eval_iterations)[0]
                    else:
                        self.selected_pumps_eval_accuracy[val_index] = 0

                    # report training status
                    print('Iteration {:5d} of {:5d} | eval acc: {:.2f} | selected pumps acc: {:.2f} | t_train: {:.2f} | t_opt: {:.2f}'.format(
                        it, self.training_iters, self.eval_accuracy[val_index], self.selected_pumps_eval_accuracy[val_index], self.t_train, curr_opt_t), flush=True)

                it += 1
                if it == self.training_iters:
                    break

        return self.model, self.eval_accuracy, self.selected_pumps_eval_accuracy, self.losses



    def evaluate(self, eval_loader, classes, do_use_opt_t=False, t_eval=None, max_iterations=40):
        """
        Evaluate neural network and calculate accuracy
        """

        error_count = 0
        num_values = 0
        num_iterations = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0


        # Calculate average optimal threshold for test dataset
        t_opt_list = []
        if do_use_opt_t is True:
            for input_samples, gt_samples in eval_loader:
                t_curr, _ = self.calc_optimal_threshold_from_data(input_samples, gt_samples)
                t_opt_list.append(t_curr)
            t_opt = np.mean(t_opt_list)
            t = t_opt
            print("Using optimal t for evaluation, t: {}".format(t))
        elif t_eval is None:
            t = self.t_train
            print("Using trained t for evaluation, t: {}".format(t))
        else:
            t = t_eval
            print("Using eval_t, t: {}".format(t))

        for input_samples, gt_samples in eval_loader:

            if num_iterations == max_iterations:
                break
            num_iterations += 1

            for b in range(input_samples[0].shape[0]):



                input_samples_mean_expanded = input_samples[1][b].unsqueeze(1).repeat(1, 800)
                mse = torch.mean(torch.square(input_samples_mean_expanded - input_samples[0][b]))

                if mse < t:
                    pred_class = classes[0]
                else:
                    pred_class = classes[1]

                error_count += torch.sum(torch.ne(pred_class, gt_samples[b])).item()
                tp += torch.sum(torch.logical_and(pred_class == classes[1], gt_samples[b] == classes[1])).item()
                fp += torch.sum(torch.logical_and(pred_class == classes[1], gt_samples[b] == classes[0])).item()
                tn += torch.sum(torch.logical_and(pred_class == classes[0], gt_samples[b] == classes[0])).item()
                fn += torch.sum(torch.logical_and(pred_class == classes[0], gt_samples[b] == classes[1])).item()
                num_values += 1

        accuracy = (num_values - error_count) / num_values
        print('acc: {:.5f} | num_values: {} | tp: {} | fp: {} | tn: {} | fn: {}'.format(accuracy, num_values, tp, fp, tn, fn), flush=True)

        if do_use_opt_t is True:
            return accuracy, tp, fp, tn, fn, t_opt
        else:
            return accuracy, tp, fp, tn, fn

