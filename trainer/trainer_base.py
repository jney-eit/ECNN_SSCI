import os
from torch.utils.data import DataLoader
import torch
import torch.optim.lr_scheduler as lrs
import utils
import copy

class Trainer_Base():
    """
    Base training class that can be used by different training approaches e.g. supervised, unsupverised
    """

    def __init__(self, params, device, dataset):

        # We need to import it here to avoid circular dependencies
        from registry import model_selector, optimizer_selector

        self.device = device

        self.dataset = dataset
        params["in_sequence_len"] = self.dataset.sequence_len

        # Save params for reuse in test runs
        self.params = params

        # Set all params of yaml file as instance variables
        for key, value in params.items():
            if key == "model_type":
                self.model_name = value
                self.model = model_selector(self.model_name, params, self.device)
                # Save previous model to reset if BER too high
                self.prev_model = model_selector(self.model_name, params,  self.device)
                # Save best model to be set as final one
                self.best_model = model_selector(self.model_name, params, self.device)

                print("model = {}".format(self.model))

            setattr(self, key, value)

        self.current_filename = 'None'
        print("self.test_pump_generalization: {}".format(self.test_pump_generalization))
        print("batch size: {}".format(self.batch_size))

        self.initialize_loaders()

        # Frequency to print loss
        self.track_loss_intervall = max(self.training_iters // 200, 100)
        self.training_iters = int(self.training_iters)

        # Change training iterations if quantization is applied
        if hasattr(self, 'quantize') and self.quantize is True:
            self.training_iters = self.fp_iterations + self.quant_iterations + self.finetune_iterations
            # List of average quantization values
            self.average_num_bits = {"params": [], "act": []}
        else:
            self.quantize = False

        # Validation bit error rate
        self.eval_accuracy = torch.zeros(int(self.training_iters // self.track_loss_intervall + 1), device=self.device)
        self.selected_pumps_eval_accuracy = torch.zeros(int(self.training_iters // self.track_loss_intervall + 1), device=self.device)

        # Learning rate scheduler
        self.use_lrs = self.lrs_milestones and self.lrs_gamma

        # Optimizer
        if self.model is not None:
            self.optimizer = optimizer_selector(self.optimize_func, self.model, self.learning_rate)

        if self.model is not None and self.use_lrs:
            print("using learning rate scheduler with milestones: {}".format(self.lrs_milestones))
            self.lr_sched = lrs.MultiStepLR(self.optimizer, milestones=self.lrs_milestones, gamma=self.lrs_gamma)

        self.losses = torch.zeros(self.training_iters, device=device, requires_grad=False)
        self.retraining_losses = torch.zeros(self.training_iters, device=device, requires_grad=False)

        # Training labels
        self.labels_true = torch.ones((self.batch_size, 1), device=self.device)
        self.labels_false = torch.zeros((self.batch_size, 1), device=self.device)

        # Number of simulation samples
        self.N_sim = self.input_symbols

        if self.model is not None:
            # Print trainable parameters
            utils.print_num_of_trainable_params(self.model, self.model_type)


    def initialize_loaders(self):
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.dataset.train_sampler, drop_last=True)
        self.eval_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.dataset.eval_sampler, drop_last=True)
        if self.test_pump_generalization:
            # Don't drop last batch, otherwise we have 0 samples for some pumps
            self.selected_pumps_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.dataset.selected_pumps_sampler, drop_last=False)
        else:
            self.selected_pumps_loader = None


    def set_bitwidth_trainable(self):
        """
        Activate training of bitwidths of weights and activations
        :return:
        """

        self.model.set_bitwidth_trainable(True)
        self.prev_model.set_bitwidth_trainable(True)
        if self.fix_input_bits:
            self.model.fix_input_bits(self.input_int_bits, self.input_frac_bits)

        if self.fix_output_bits:
            self.model.fix_output_bits(self.output_int_bits, self.output_frac_bits)

    def fix_bitwidth(self):
        """
        Fix bitwidth of weights and activations
        :return:
        """

        print("Fixing bitwidth to integer for finetuning and reset weights.")
        self.model.fix_learned_bits()
        self.prev_model.fix_learned_bits()

    def lower_bound_model_accuracy(self, val_index, eval_accuracy_prev):
        """
        Loads previous model if accuracy of current model is less than 80 % of current one
        :param val_index: Current evaluation accuracy index
        :param eval_accuracy_prev: Previous evaluation accuracy
        :return: New evaluation accuracy
        """

        if self.eval_accuracy[val_index] < 0.8 * eval_accuracy_prev:
            # Load model of previous iteration
            self.model.load_state_dict(copy.deepcopy(self.prev_model.state_dict()))
            print("Loading previous model: validation accuracy prev: {:.5f}, validation accuracy current: {:.5f}".format(eval_accuracy_prev, self.eval_accuracy[val_index]))
            eval_accuracy_new = eval_accuracy_prev
        else:
            # Save current model
            eval_accuracy_new = self.eval_accuracy[val_index]
            self.prev_model.load_state_dict(copy.deepcopy(self.model.state_dict()))

        return eval_accuracy_new

    def save_model_if_best(self, val_index, best_finetuned_acc):
        """
        Save the current model if it has the best accuracy of all training iterations
        :param val_index: Current evaluation accuracy index
        :param best_finetuned_acc: Best model accuracy of all previous runs
        :return: New best accuracy
        """

        if best_finetuned_acc < self.eval_accuracy[val_index]:
            print("Current model is best finetuned model with accuracy of: {}".format(self.eval_accuracy[val_index]))
            best_finetuned_acc_new = self.eval_accuracy[val_index]
            self.best_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        else:
            best_finetuned_acc_new = best_finetuned_acc

        return best_finetuned_acc_new


    def evaluate(self, **kwargs):
        """
        Needs to be implemented by child class
        """
        raise NotImplementedError("Instantiate a specific Trainer (e.g. Sup_Trainer) object for evaluation!")

    def train(self, **kwargs):
        """
        Needs to be implemented by child class
        """
        raise NotImplementedError("Instantiate a specific Trainer (e.g. Sup_Trainer) object for training!")

    @staticmethod
    def load_state(trainer_file, device, dataset, batch_size=None, trainer_type=None):
        """
        Load previous trainer including model
        :param trainer_file: .pt file containing state dict of trainer
        :param device: device ("cpu" or "gpu") the model is loaded for
        :param dataset: dataset
        :param batch_size: samples per batch
        :param trainer_type: type of child trainer class e.g. Sup_Trainer
        :return: Trainer object
        """

        if not os.path.exists(trainer_file):
            raise ValueError('file ({})) does not exist!'.format(trainer_file))

        print("Loading state of {}...".format(trainer_file))

        # Load Trainer from file
        checkpoint = torch.load(trainer_file, map_location=device)
        checkpoint["device"] = device

        if batch_size:
            checkpoint["batch_size"] = batch_size

        # Import subclasses here to avoid circular dependencies
        from registry import train_class_dict

        training_mode = checkpoint["training_mode"]
        model_type = checkpoint["model_type"]

        # Trainer type to use with corresponding model
        if trainer_type is None:

            # Select trainer class
            if (training_mode, model_type) in train_class_dict:
                trainer_type = train_class_dict[(training_mode, model_type)]
            else:
                raise ValueError("training mode {} not valid with model type {}".format(training_mode, model_type))

        if dataset.test_pump_generalization:
            print("Adding selected pumps loader to loaded trainer...")
            # Don't drop last batch, otherwise we have 0 samples for some pumps
            checkpoint["selected_pumps_loader"] = torch.utils.data.DataLoader(dataset, batch_size=checkpoint["batch_size"], sampler=dataset.selected_pumps_sampler, drop_last=False)
        else:
            checkpoint["selected_pumps_loader"] = None

        tr = trainer_type(checkpoint, device, dataset)
        if model_type != "None":
            tr.model.load_state_dict(checkpoint["model_state_dict"])
            tr.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            tr.t_train = checkpoint["t_train"]


        return tr


    def freeze_state(self, dir=None, filename=None, overwrite=True):
        """
        Save current trainer configuration and model in .pt file
        :param dir: Directory where model is saved
        :param filename: Filename of .pt file
        :param overwrite: Overwrite existing .pt or store with new index
        :return:
        """

        # We need to import it here to avoid circular dependencies
        from registry import model_selector, optimizer_selector

        state_dict = {}

        def clone_dict_or_list(x):
            if isinstance(x, dict):
                out = {}
                for key, val in x.items():
                    if isinstance(val, torch.Tensor):
                        out[key] = val.clone()
                    elif isinstance(val, dict) or isinstance(val, list):
                        out[key] = clone_dict_or_list(val)
                    else:
                        out[key] = copy.deepcopy(val)

                return out

            if isinstance(x, list):
                out = []
                for val in x:
                    if isinstance(val, torch.Tensor):
                        out.append(val.clone())
                    elif isinstance(val, dict) or isinstance(val, list):
                        out.append(clone_dict_or_list(val))
                    else:
                        out.append(copy.deepcopy(val))
                return out


        for attr, value in vars(self).items():
            #print("Copying {} with value {}".format(attr, value))
            # If the attribute is a Tensor, clone it
            if isinstance(value, torch.Tensor):
                state_dict[attr] = value.clone()
            # If the attribute is a dict, each element could be a tensor
            elif isinstance(value, dict) or isinstance(value, list):
                state_dict[attr] = clone_dict_or_list(value)
            # Deepcopy fails for quantized model, so we copy it like this
            elif attr == "model" or attr == "prev_model" or attr == "best_model":
                if self.model_name != "None":
                    temp_model = model_selector(self.model_name, self.params, self.device)
                    temp_model.load_state_dict(value.state_dict())
                    state_dict[attr] = temp_model
            # Otherwise just copy the attribute
            else:
                state_dict[attr] = copy.deepcopy(value)

        #state_dict = copy.deepcopy(vars(self))
        # Do not save dataset in state_dict
        del state_dict["dataset"]
        del state_dict["train_loader"]
        del state_dict["eval_loader"]
        if self.test_pump_generalization:
            del state_dict["selected_pumps_loader"]

        print("self.dataset: {}".format(self.dataset))

        if self.model_type != "None":
            state_dict["model_state_dict"] = self.model.state_dict()
            state_dict["optimizer_state_dict"] = self.optimizer.state_dict()

        if dir is None:
            dir = os.path.join('trained_models', 'CNN_1D')
        if filename is None:
            if overwrite is True:
                file = os.path.join(dir, 'EQ_{}.pt'.format(self.model_name))
            else:
                file, _ = utils.get_next_unused_filename(dir, self.model_name)
        else:
            if filename.endswith(".pt"):
                file = os.path.join(dir, '{}'.format(filename))
            else:
                file = os.path.join(dir, '{}.pt'.format(filename))

        torch.save(state_dict, file)
        print('Saved model to {}.'.format(file))

        return




