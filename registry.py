

"""
    This file needs to be edited if new Models, Channels or Trainers are included
"""


from model import CNN_1D
from trainer.trainer_superwise import Sup_Trainer
from trainer.trainer_threshold import Threshold_Trainer
import torch.optim as optim

# Map model and training mode combination to specific trainer class
train_class_dict = {
    ('sup', 'CNN_1D'): Sup_Trainer,
    ('mean', 'None'): Threshold_Trainer,
}


# If new model is implemented in model.py it needs to be added here. Make sure to also modify model_selector if required
# Map model name to actual class
model_dict = {
    "CNN_1D": CNN_1D,
    "None": None
}

# If new optimizer is implemented  it needs to be added here.  Make sure to also modify optimizer_selector if required
# Map optimizer name to actual class
optimizer_dict = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "Adadelta": optim.Adadelta,
    "Adamax": optim.Adamax,
    "ASGD": optim.ASGD,
    "RMSprop": optim.RMSprop
}


# If a new model is implemented the parameters should be added here
def model_selector(model_name, params, device):
    """
    Creates instance of model as defined by parameters file
    :param model_name: Type of model e.g. CNN_1D
    :param params: Dictionary of parameters
    :param device: cpu or gpu
    :return:
    """

    if model_name not in model_dict:
        raise NotImplementedError("Unknown model: " + str(model_name))

    # Set arguments to class as dict
    if model_name == "None":
        kwargs = {}
    else:
        kwargs = {"batch_size": params["batch_size"], "device": device}

    if model_name == "CNN_1D":

        kwargs["kernel_size"] = params["kernel_size"]
        kwargs["depth"] = params["depth"]
        kwargs["intermediate_channels"] = params["intermediate_channels"]
        kwargs["in_sequence_len"] = params["in_sequence_len"]
        kwargs["pooling_layer_type"] = params["last_layer_type"]

        if "channels_in" in params:
            kwargs["channels_in"] = params["channels_in"]

        if "multi_outs" in params:
            kwargs["multi_outs"] = params["multi_outs"]

        if params["additional_input"] == "max":
            return model_dict["CNN_1D_Additional_Input_Max"](**kwargs)

        if params["additional_input"] == "mean":
            if params["channels_in"] != 6:
                raise ValueError("Mean additional input only supported for 6 input channels")

        return model_dict[model_name](**kwargs)

    if model_name == "CNN_1D_Additional_Input":
        kwargs["kernel_size"] = params["kernel_size"]
        kwargs["depth"] = params["depth"]
        kwargs["intermediate_channels"] = params["intermediate_channels"]
        kwargs["in_sequence_len"] = params["in_sequence_len"]
        kwargs["pooling_layer_type"] = params["last_layer_type"]

        if "channels_in" in params:
            kwargs["channels_in"] = params["channels_in"]

        if "multi_outs" in params:
            kwargs["multi_outs"] = params["multi_outs"]

    if model_name == "None":
        return None

    return model_dict[model_name](**kwargs)

# If a new optimizer is implemented the parameters should be added here
def optimizer_selector(optimizer_name, model, learning_rate):
    """
    Create optimizer object as defined by parameters
    :param optimizer_name: Type of optimizer e.g. RMSprop
    :param params: Dictionary of parameters
    :param model: NN model optimizer gets applied to
    :return: Optimizer object
    """

    if optimizer_name not in optimizer_dict:
        raise NotImplementedError("Unknown optimizer: " + str(optimizer_name))

    kwargs = {"params": model.parameters(), "lr": learning_rate}

    if optimizer_name == "RMSprop":
        kwargs["alpha"] = 0.9

    return optimizer_dict[optimizer_name](**kwargs)


