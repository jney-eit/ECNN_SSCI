import torch
import torch.nn as nn

N_SIM_MAX = 10000



def weight_reset(layer):
    """
    Reste weights to xavier uniform distribution
    :param layer: layer
    :return:
    """
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        try:
            layer.bias.data.fill_(0.001)
        except Exception as e:
            pass


def init_dirac(layer):
    """
    Init weights of 1d Conv layer with dirac
    :param layer: layer
    :return:
    """
    if isinstance(layer, nn.Conv1d):
        torch.nn.init.dirac_(layer.weight)
        try:
            layer.bias.data.fill_(0.001)
        except Exception as e:
            pass


class CNN_1D(nn.Module):
    """
    1D CNN based on L 1d convolutional layers followed by global average pooling or fully connected layer. Maps sequence of values to one output
    """

    def __init__(self, batch_size, depth, device,  in_sequence_len, channels_in=1, kernel_size=9, intermediate_channels=4, multi_outs=1, pooling_layer_type="global_average"):
        """
        Initialization of CNN_1D
        :param batch_size: Samples per batch
        :param depth: Number of 1d convolutional layers
        :param device: cpu or gpu
        :param channels_in: Number of input channels
        :param kernel_size: Kernel size of 1d convolutional layers (same for each layer)
        :param intermediate_channels: Number of channels of middle convolutional layers
        :param in_sequence_len: Length of sequence
        :param multi_outs: Number of outputs to calculate in parallel
        :param pooling_layer_type: Type of last layer, "global_average" or "fully_connected"
        """
        super().__init__()

        # Number of conv layers
        self.depth = depth
        # Oversampling factor
        self.in_per_out = in_sequence_len

        self.batch_size = batch_size
        self.device = device

        self.conv1d_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.act_layers = nn.ModuleList()

        kernel_sizes = [kernel_size] * self.depth
        paddings = [ks // 2 for ks in kernel_sizes]
        strides = [multi_outs] + [1] * (self.depth - 1)
        channels_in_temp = channels_in
        channels_out_temp = intermediate_channels

        for l in range(self.depth - 1):
            self.conv1d_layers.append(nn.Conv1d(channels_in_temp, channels_out_temp, kernel_sizes[l], stride=strides[l], padding=paddings[l], bias=True, device=self.device))
            channels_in_temp = channels_out_temp
            self.bn_layers.append(nn.BatchNorm1d(channels_out_temp, device=self.device))
            self.act_layers.append(nn.ReLU())

        if kernel_sizes[-1] % 2 == 0:
            paddings[-1] -= 1

        self.conv1d_layers.append(nn.Conv1d(channels_in_temp, multi_outs, kernel_sizes[-1], stride=strides[-1], padding=paddings[-1], bias=True, device=self.device))

        if pooling_layer_type == "global_average":
            self.last_layer = nn.AvgPool1d(in_sequence_len)
        elif pooling_layer_type == "fully_connected":
            self.last_layer = nn.Linear(in_sequence_len, 1, bias=False, device=self.device)
        else:
            raise NotImplementedError("{} not supported as last layer.".format(pooling_layer_type))


    def forward(self, x):
        """
        Calculate inference of CNN
        :param x: inputs
        :return: outputs
        """

        for l in range(self.depth - 1):
            x = self.conv1d_layers[l](x)
            x = self.bn_layers[l](x)
            x = self.act_layers[l](x)

        out = self.conv1d_layers[self.depth - 1](x)
        out = self.last_layer(out)

        # Reshape output so multi_out channels are mapped to sequence
        batch_size = out.shape[0]

        out = out.permute(0,2,1)
        out = out.reshape(batch_size, 1, -1)

        return out



class Interpolate(nn.Module):
    """
    Interpolate layer, used for upsampling
    """
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x















