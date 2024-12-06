from trainer.trainer_base import Trainer_Base

import torch.nn as nn
import torch

class Sup_Trainer(Trainer_Base):
    """
    Trainer to train model using supervised Mean-Square-Error loss
    """

    def __init__(self, params, device, dataset):

        super(Sup_Trainer, self).__init__(params, device, dataset)

        self.loss_func = nn.MSELoss()

        # Number of simulation samples
        self.N_sim = self.input_symbols

    def model_forward(self, input_samples, additional_input_fac=1):
        """
        Forward model and return output
        """

        # Get model output
        if self.additional_input == "mean":
            input_mean_dif = (input_samples[0] - input_samples[1].unsqueeze(2).repeat(1, 1, 800))
            input_mean_dif = input_mean_dif * additional_input_fac
            model_in = torch.cat((input_samples[0], input_mean_dif), dim=1)
        elif self.subtract_mean_from_input is True:
            input_mean_dif = (input_samples[0] - input_samples[1].unsqueeze(2).repeat(1, 1, 800))
            model_in = input_mean_dif * additional_input_fac
        else:
            model_in = input_samples

        #torch.onnx.export(self.model, model_in, "sample_model.onnx")
        #onnx_program.save("model.onnx")
        #exit(10)
        model_out = self.model(model_in)
        return model_out

    def train(self):
        """
        Train generator in supervised fashion
        """

        eval_accuracy_prev = 0
        best_finetuned_acc = 0

        it = 0
        last_lr =  self.learning_rate
        while it < self.training_iters:
            for input_samples, gt_samples in self.train_loader:

                # Ensure model is in training mode
                self.model.train()

                # Reset gradients
                self.optimizer.zero_grad()

                model_out = self.model_forward(input_samples)

                # Compute loss
                loss = self.calculate_loss(model_out, gt_samples)

                # Compute gradients
                loss.backward()
                self.optimizer.step()

                # Learning rate scheduler
                if self.use_lrs:
                    self.lr_sched.step()
                    if last_lr != self.lr_sched.get_last_lr():
                        last_lr = self.lr_sched.get_last_lr()
                        print("learning rate set to {} at iteration {}.".format(self.lr_sched.get_last_lr(), it))

                if (it + 1) % self.track_loss_intervall == 0:
                    # Compute validation BER

                    print("\n")
                    print("#############################################################################")

                    val_index = int(it // self.track_loss_intervall)
                    eval_iterations = 50

                    self.eval_accuracy[val_index], tp, fp, tn, fn, _, _ = self.evaluate(self.eval_loader, self.dataset.classes.to(self.device), max_iterations=eval_iterations)
                    self.losses[val_index] = loss.detach()

                    # report training status
                    print('Iteration {:5d} of {:5d} | eval accuracy: {:.5f} | loss: {:.5f}'.format(it, self.training_iters, self.eval_accuracy[val_index], self.losses[val_index]), flush=True)

                    if self.selected_pumps_loader:
                        print("Testing model for pump with index {}...".format(self.pump_eval_idx))
                        print("len(self.selected_pumps_loader): {}".format(len(self.selected_pumps_loader)))

                        self.selected_pumps_eval_accuracy[val_index], tp, fp, tn, fn, _, _ =  self.evaluate(self.selected_pumps_loader, self.dataset.classes.to(self.device), max_iterations=eval_iterations)

                        print('selected pumps acc: {:.5f} | tp_selected_pump: {} | fp_selected_pump: {} | ' 'tn_selected_pump: {} | fn_selected_pump: {}'.format(self.selected_pumps_eval_accuracy[val_index], tp, fp, tn, fn), flush=True)
                    else:
                        self.selected_pumps_eval_accuracy[val_index] = 0

                it += 1
                if it == self.training_iters:
                    break

        return self.model, self.eval_accuracy, self.selected_pumps_eval_accuracy, self.losses


    def evaluate(self, eval_loader, classes, max_iterations=40, additional_input_fac=1):
        """
        Evaluate neural network and calculate accuracy
        """

        error_count = 0
        num_values = 0
        num_iterations = 0
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

        self.model.eval()

        output_samples_0 = []
        output_samples_1 = []

        for input_samples, gt_samples in eval_loader:

            if num_iterations == max_iterations:
                break
            num_iterations += 1

            out = self.model_forward(input_samples, additional_input_fac=additional_input_fac)

            for b in range(out.shape[0]):
                dist = torch.abs(classes - out[b])
                pred_classes = classes[torch.argmin(dist)]

                if gt_samples[b] == classes[0]:
                    output_samples_0.extend(out[b].detach().cpu().numpy().flatten().tolist())
                else:
                    output_samples_1.extend(out[b].detach().cpu().numpy().flatten().tolist())

                error_count += torch.sum(torch.ne(pred_classes, gt_samples[b])).item()
                true_positive += torch.sum(torch.logical_and(pred_classes == classes[1], gt_samples[b] == classes[1])).item()
                false_positive += torch.sum(torch.logical_and(pred_classes == classes[1], gt_samples[b] == classes[0])).item()
                true_negative += torch.sum(torch.logical_and(pred_classes == classes[0], gt_samples[b] == classes[0])).item()
                false_negative += torch.sum(torch.logical_and(pred_classes == classes[0], gt_samples[b] == classes[1])).item()
                num_values += torch.numel(out[b])

        accuracy = (num_values - error_count) / num_values

        return accuracy, true_positive, false_positive, true_negative, false_negative, output_samples_0, output_samples_1


    def calculate_loss(self, predictions, labels):
        """
        Calculate mean square error loss and quantization loss
        :param predictions: Predicted symbols
        :param labels: Ground truth labels
        :return: loss
        """

        eps = 1e-6
        loss = self.loss_func(predictions.view(-1, 1).float(), labels.view(-1, 1))
        loss += eps
        return loss
