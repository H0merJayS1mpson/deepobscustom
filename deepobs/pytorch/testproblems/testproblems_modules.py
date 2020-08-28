# -*- coding: utf-8 -*-
"""All torch modules that are used by the testproblems."""

import torch
from torch import nn
from torch.nn import functional as F
from .testproblems_utils import tfconv2d
from .testproblems_utils import tfmaxpool2d
from .testproblems_utils import flatten
from .testproblems_utils import tfconv2d_transpose
from .testproblems_utils import mean_allcnnc
from .testproblems_utils import residual_block
from .testproblems_utils import _truncated_normal_init
from ast import literal_eval


#pass initialization method and params as dict.
class net_mnist_logreg(nn.Sequential):
    def __init__(self, num_outputs, initializations=None):
        super(net_mnist_logreg, self).__init__()

        self.add_module('flatten', flatten())
        self.add_module('dense', nn.Linear(in_features=784, out_features=num_outputs))
        # init
        nn.init.constant_(self.dense.bias, 0.0)
        if initializations is not None and 'dense' in initializations:
            (eval(initializations['dense'][0])(*[self.dense.weight, *initializations['dense'][1:]]))
        else:
            nn.init.constant_(self.dense.weight, 0.0)

class net_cifar10_3c3d(nn.Sequential):
    """  Basic conv net for cifar10/100. The network consists of
    - thre conv layers with ReLUs, each followed by max-pooling
    - two fully-connected layers with ``512`` and ``256`` units and ReLU activation
    - output layer with softmax
  The weight matrices are initialized using Xavier initialization and the biases
  are initialized to ``0.0``."""

    def __init__(self, num_outputs, initializations=None):
        """Args:
            num_outputs (int): The number of outputs (i.e. target classes)."""
        super(net_cifar10_3c3d, self).__init__()

        self.add_module('conv1', tfconv2d(in_channels = 3, out_channels = 64, kernel_size = 5))
        self.add_module('relu1', nn.ReLU())
        self.add_module('maxpool1', tfmaxpool2d(kernel_size = 3, stride = 2, tf_padding_type = 'same'))

        self.add_module('conv2', tfconv2d(in_channels = 64, out_channels = 96, kernel_size = 3))
        self.add_module('relu2', nn.ReLU())
        self.add_module('maxpool2', tfmaxpool2d(kernel_size = 3, stride = 2, tf_padding_type = 'same'))

        self.add_module('conv3', tfconv2d(in_channels = 96, out_channels = 128, kernel_size = 3, tf_padding_type='same'))
        self.add_module('relu3', nn.ReLU())
        self.add_module('maxpool3', tfmaxpool2d(kernel_size = 3, stride = 2, tf_padding_type = 'same'))

        self.add_module('flatten', flatten())

        self.add_module('dense1', nn.Linear(in_features = 3 * 3 * 128, out_features = 512))
        self.add_module('relu4', nn.ReLU())
        self.add_module('dense2', nn.Linear(in_features = 512, out_features = 256))
        self.add_module('relu5', nn.ReLU())
        self.add_module('dense3', nn.Linear(in_features = 256, out_features = num_outputs))

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if initializations is not None and 'Conv2d' in initializations:
                    nn.init.constant_(module.bias, 0.0)
                    (eval(initializations['Conv2d'][0])(*[module.weight, *initializations['Conv2d'][1:]]))
                else:
                    nn.init.constant_(module.bias, 0.0)
                    nn.init.xavier_normal_(module.weight)
            if isinstance(module, nn.Linear):
                if initializations is not None and 'Linear' in initializations:
                    nn.init.constant_(module.bias, 0.0)
                    (eval(initializations['Linear'][0])(*[module.weight, *initializations['Linear'][1:]]))
                else:
                    nn.init.constant_(module.bias, 0.0)
                    nn.init.xavier_uniform_(module.weight)


class net_mnist_2c2d(nn.Sequential):
    """  Basic conv net for (Fashion-)MNIST. The network has been adapted from the `TensorFlow tutorial\
  <https://www.tensorflow.org/tutorials/estimators/cnn>`_ and consists of

    - two conv layers with ReLUs, each followed by max-pooling
    - one fully-connected layers with ReLUs
    - output layer with softmax

  The weight matrices are initialized with truncated normal (standard deviation
  of ``0.05``) and the biases are initialized to ``0.05``."""

    def __init__(self, num_outputs, initializations=None):
        """Args:
            num_outputs (int): The numer of outputs (i.e. target classes)."""

        super(net_mnist_2c2d, self).__init__()
        self.add_module('conv1', tfconv2d(in_channels = 1, out_channels = 32, kernel_size = 5, tf_padding_type='same'))
        self.add_module('relu1', nn.ReLU())
        self.add_module('max_pool1', tfmaxpool2d(kernel_size = 2, stride = 2, tf_padding_type='same'))

        self.add_module('conv2', tfconv2d(in_channels = 32, out_channels = 64, kernel_size = 5, tf_padding_type='same'))
        self.add_module('relu2', nn.ReLU())
        self.add_module('max_pool2', tfmaxpool2d(kernel_size = 2, stride = 2, tf_padding_type='same'))

        self.add_module('flatten', flatten())

        self.add_module('dense1', nn.Linear(in_features = 7*7*64, out_features = 1024))
        self.add_module('relu3', nn.ReLU())

        self.add_module('dense2', nn.Linear(in_features = 1024, out_features = num_outputs))

        # init the layers
        for module in (self.modules()):
            if isinstance(module, nn.Conv2d):
                if initializations is not None and 'Conv2d' in initializations:
                    nn.init.constant_(module.bias, 0.0)
                    (eval(initializations['Conv2d'][0])(*[module.weight, *initializations['Conv2d'][1:]]))
                else:
                    nn.init.constant_(module.bias, 0.05)
                    module.weight.data = _truncated_normal_init(module.weight.data, mean=0, stddev=0.05)
            if isinstance(module, nn.Linear):
                if initializations is not None and 'Linear' in initializations:
                    nn.init.constant_(module.bias, 0.0)
                    (eval(initializations['Linear'][0])(*[module.weight, *initializations['Linear'][1:]]))
                else:
                    nn.init.constant_(module.bias, 0.05)
                    module.weight.data = _truncated_normal_init(module.weight.data, mean=0, stddev=0.05)



class net_vae(nn.Module):
    """  A basic VAE for (Faschion-)MNIST. The network has been adapted from the `here\
  <https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776>`_
  and consists of an encoder:

    - With three convolutional layers with each ``64`` filters.
    - Using a leaky ReLU activation function with :math:`\\alpha = 0.3`
    - Dropout layers after each convolutional layer with a rate of ``0.2``.

  and an decoder:

    - With two dense layers with ``24`` and ``49`` units and leaky ReLU activation.
    - With three deconvolutional layers with each ``64`` filters.
    - Dropout layers after the first two deconvolutional layer with a rate of ``0.2``.
    - A final dense layer with ``28 x 28`` units and sigmoid activation.
"""
    def __init__(self, n_latent, initializations=None):
        """Args:
            n_latent (int): Size of the latent space."""
        super(net_vae, self).__init__()
        self.n_latent = n_latent

        # encoding layers
        self.conv1 = tfconv2d(in_channels = 1, out_channels = 64, kernel_size = 4, stride=2, tf_padding_type='same')
        self.dropout1 = nn.Dropout(p = 0.2)

        self.conv2 = tfconv2d(in_channels = 64, out_channels = 64, kernel_size = 4, stride=2, tf_padding_type='same')
        self.dropout2 = nn.Dropout(p = 0.2)

        self.conv3 = tfconv2d(in_channels = 64, out_channels = 64, kernel_size = 4, stride=1, tf_padding_type='same')
        self.dropout3 = nn.Dropout(p = 0.2)

        self.dense1 = nn.Linear(in_features = 7 * 7 * 64, out_features = self.n_latent)
        self.dense2 = nn.Linear(in_features = 7 * 7 * 64, out_features = self.n_latent)

        # decoding layers
        self.dense3 = nn.Linear(in_features = 8, out_features = 24)
        self.dense4 = nn.Linear(in_features = 24, out_features = 24*2 + 1)

        self.deconv1 = tfconv2d_transpose(in_channels = 1, out_channels = 64, kernel_size = 4, stride = 2, tf_padding_type = 'same')
#        self.deconv1 = nn.ConvTranspose2d(in_channels=1, out_channels=64, kernel_size=4, stride=2,)
        self.dropout4 = nn.Dropout(p = 0.2)

        self.deconv2 = tfconv2d_transpose(in_channels = 64, out_channels = 64, kernel_size = 4, stride = 1, tf_padding_type = 'same')
        self.dropout5 = nn.Dropout(p = 0.2)

        self.deconv3 = tfconv2d_transpose(in_channels = 64, out_channels = 64, kernel_size = 4, stride = 1, tf_padding_type = 'same')
        self.dropout6 = nn.Dropout(p = 0.2)

        self.dense5 = nn.Linear(in_features = 14 * 14 * 64, out_features = 28 * 28)

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if initializations is not None and 'Conv2d' in initializations:
                    nn.init.constant_(module.bias, 0.0)
                    (eval(initializations['Conv2d'][0])(*[module.weight, *initializations['Conv2d'][1:]]))
                else:
                    nn.init.constant_(module.bias, 0.0)
                    nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.ConvTranspose2d):
                if initializations is not None and 'ConvTranspose2d' in initializations:
                    nn.init.constant_(module.bias, 0.0)
                    (eval(initializations['ConvTranspose2d'][0])(*[module.weight, *initializations['ConvTranspose2d'][1:]]))
                else:
                    nn.init.constant_(module.bias, 0.0)
                    nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear):
                if initializations is not None and 'Linear' in initializations:
                    nn.init.constant_(module.bias, 0.0)
                    (eval(initializations['Linear'][0])(*[module.weight, *initializations['Linear'][1:]]))
                else:
                    nn.init.constant_(module.bias, 0.0)
                    nn.init.xavier_uniform_(module.weight)

    def encode(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.3)
        x = self.dropout1(x)

        x = F.leaky_relu(self.conv2(x), negative_slope = 0.3)
        x = self.dropout2(x)

        x = F.leaky_relu(self.conv3(x), negative_slope = 0.3)
        x = self.dropout3(x)

        x = x.view(-1, 7*7*64)

        mean = self.dense1(x)
        std_dev = 0.5*self.dense2(x)
        eps = torch.randn_like(std_dev)
        z = mean + eps * torch.exp(std_dev)

        return z, mean, std_dev

    def decode(self, z):
        x = F.leaky_relu(self.dense3(z), negative_slope = 0.3)
        x = F.leaky_relu(self.dense4(x), negative_slope = 0.3)

        x = x.view(-1, 1, 7, 7)

        x = F.relu(self.deconv1(x))
        x = self.dropout4(x)

        x = F.relu(self.deconv2(x))
        x = self.dropout5(x)

        x = F.relu(self.deconv3(x))
        x = self.dropout6(x)

        x = x.view(-1, 14 * 14 * 64, )

        x = F.sigmoid(self.dense5(x))

        images = x.view(-1, 1, 28, 28)

        return images

    def forward(self, x):
        z, mean, std_dev = self.encode(x)

        image = self.decode(z)

        return image, mean, std_dev


class net_vgg(nn.Sequential):
    def __init__(self, num_outputs, variant, initializations=None):
        super(net_vgg, self).__init__()

        self.add_module('upsampling', nn.UpsamplingBilinear2d(size= (224, 224)))

        self.add_module('conv11', tfconv2d(in_channels = 3, out_channels = 64, kernel_size = 3, tf_padding_type='same'))
        self.add_module('relu11', nn.ReLU())
        self.add_module('conv12', tfconv2d(in_channels = 64, out_channels = 64, kernel_size = 3, tf_padding_type='same'))
        self.add_module('relu12', nn.ReLU())
        self.add_module('max_pool1', tfmaxpool2d(kernel_size = 2, stride = 2, tf_padding_type='same'))

        self.add_module('conv21', tfconv2d(in_channels = 64, out_channels = 128, kernel_size = 3, tf_padding_type='same'))
        self.add_module('relu21', nn.ReLU())
        self.add_module('conv22', tfconv2d(in_channels = 128, out_channels = 128, kernel_size = 3, tf_padding_type='same'))
        self.add_module('relu22', nn.ReLU())
        self.add_module('max_pool2', tfmaxpool2d(kernel_size = 2, stride = 2, tf_padding_type='same'))

        self.add_module('conv31', tfconv2d(in_channels = 128, out_channels = 256, kernel_size = 3, tf_padding_type='same'))
        self.add_module('relu31', nn.ReLU())
        self.add_module('conv32', tfconv2d(in_channels = 256, out_channels = 256, kernel_size = 3, tf_padding_type='same'))
        self.add_module('relu32', nn.ReLU())
        self.add_module('conv33', tfconv2d(in_channels = 256, out_channels = 256, kernel_size = 3, tf_padding_type='same'))
        self.add_module('relu33', nn.ReLU())
        if variant == 19:
            self.add_module('conv34', tfconv2d(in_channels = 256, out_channels = 256, kernel_size = 3, tf_padding_type='same'))
            self.add_module('relu34', nn.ReLU())
        self.add_module('max_pool3', tfmaxpool2d(kernel_size = 2, stride = 2, tf_padding_type='same'))

        self.add_module('conv41', tfconv2d(in_channels = 256, out_channels = 512, kernel_size = 3, tf_padding_type='same'))
        self.add_module('relu41', nn.ReLU())
        self.add_module('conv42', tfconv2d(in_channels = 512, out_channels = 512, kernel_size = 3, tf_padding_type='same'))
        self.add_module('relu42', nn.ReLU())
        self.add_module('conv43', tfconv2d(in_channels = 512, out_channels = 512, kernel_size = 3, tf_padding_type='same'))
        self.add_module('relu43', nn.ReLU())
        if variant == 19:
            self.add_module('conv44', tfconv2d(in_channels = 512, out_channels = 512, kernel_size = 3, tf_padding_type='same'))
            self.add_module('relu44', nn.ReLU())
        self.add_module('max_pool4', tfmaxpool2d(kernel_size = 2, stride = 2, tf_padding_type='same'))

        self.add_module('conv51', tfconv2d(in_channels = 512, out_channels = 512, kernel_size = 3, tf_padding_type='same'))
        self.add_module('relu51', nn.ReLU())
        self.add_module('conv52', tfconv2d(in_channels = 512, out_channels = 512, kernel_size = 3, tf_padding_type='same'))
        self.add_module('relu52', nn.ReLU())
        self.add_module('conv53', tfconv2d(in_channels = 512, out_channels = 512, kernel_size = 3, tf_padding_type='same'))
        self.add_module('relu53', nn.ReLU())
        if variant == 19:
            self.add_module('conv54', tfconv2d(in_channels = 512, out_channels = 512, kernel_size = 3, tf_padding_type='same'))
            self.add_module('relu54', nn.ReLU())
        self.add_module('max_pool5', tfmaxpool2d(kernel_size = 2, stride = 2, tf_padding_type='same'))

        self.add_module('flatten', flatten())

        self.add_module('dense1', nn.Linear(in_features = 7*7*512, out_features = 4096))
        self.add_module('relu1', nn.ReLU())
        self.add_module('dropout1', nn.Dropout(p = 0.5))

        self.add_module('dense2', nn.Linear(in_features = 4096, out_features = 4096))
        self.add_module('relu2', nn.ReLU())
        self.add_module('dropout2', nn.Dropout(p = 0.5))

        self.add_module('dense3', nn.Linear(in_features = 4096, out_features = num_outputs))

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if initializations is not None and 'Conv2d' in initializations:
                    nn.init.constant_(module.bias, 0.0)
                    (eval(initializations['Conv2d'][0])(*[module.weight, *initializations['Conv2d'][1:]]))
                else:
                    nn.init.constant_(module.bias, 0.0)
                    nn.init.xavier_normal_(module.weight)

            if isinstance(module, nn.Linear):
                if initializations is not None and 'Linear' in initializations:
                    nn.init.constant_(module.bias, 0.0)
                    (eval(initializations['Linear'][0])(*[module.weight, *initializations['Linear'][1:]]))
                else:
                    nn.init.constant_(module.bias, 0.0)
                    nn.init.xavier_uniform_(module.weight)


class net_cifar100_allcnnc(nn.Sequential):
    def __init__(self, initializations=None):
        super(net_cifar100_allcnnc, self).__init__()

        self.add_module('dropout1', nn.Dropout(p = 0.2))

        self.add_module('conv1', tfconv2d(in_channels = 3, out_channels = 96, kernel_size = 3, tf_padding_type='same'))
        self.add_module('relu1', nn.ReLU())
        self.add_module('conv2', tfconv2d(in_channels = 96, out_channels = 96, kernel_size = 3, tf_padding_type='same'))
        self.add_module('relu2', nn.ReLU())
        self.add_module('conv3', tfconv2d(in_channels = 96, out_channels = 96, kernel_size = 3, stride=(2,2), tf_padding_type='same'))
        self.add_module('relu3', nn.ReLU())

        self.add_module('dropout2', nn.Dropout(p = 0.5))

        self.add_module('conv4', tfconv2d(in_channels = 96, out_channels = 192, kernel_size = 3, tf_padding_type='same'))
        self.add_module('relu4', nn.ReLU())
        self.add_module('conv5', tfconv2d(in_channels = 192, out_channels = 192, kernel_size = 3, tf_padding_type='same'))
        self.add_module('relu5', nn.ReLU())
        self.add_module('conv6', tfconv2d(in_channels = 192, out_channels = 192, kernel_size = 3, stride=(2,2), tf_padding_type='same'))
        self.add_module('relu6', nn.ReLU())

        self.add_module('dropout3', nn.Dropout(p = 0.5))

        self.add_module('conv7', tfconv2d(in_channels = 192, out_channels = 192, kernel_size = 3))
        self.add_module('relu7', nn.ReLU())
        self.add_module('conv8', tfconv2d(in_channels = 192, out_channels = 192, kernel_size = 1, tf_padding_type='same'))
        self.add_module('relu8', nn.ReLU())
        self.add_module('conv9', tfconv2d(in_channels = 192, out_channels = 100, kernel_size = 1, tf_padding_type='same'))
        self.add_module('relu9', nn.ReLU())

        self.add_module('mean', mean_allcnnc())

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if initializations is not None and 'Conv2d' in initializations:
                    nn.init.constant_(module.bias, 0.0)
                    (eval(initializations['Conv2d'][0])(*[module.weight, *initializations['Conv2d'][1:]]))
                else:
                    nn.init.constant_(module.bias, 0.1)
                    nn.init.xavier_normal_(module.weight)

class net_wrn(nn.Sequential):
    def __init__(self, num_residual_blocks, widening_factor, num_outputs, bn_momentum=0.9, initializations=None):
        super(net_wrn, self).__init__()

        # initial conv
        self.add_module('conv1', tfconv2d(3, 16, 3, bias = False, tf_padding_type='same'))

        self._filters = [16, 16 * widening_factor, 32 * widening_factor, 64 * widening_factor]
        self._strides = [1, 2, 2]

        # loop over three residual groups
        for group_number in range(1, 4):
            # first residual block is special since it has to change the number of output channels for the skip connection
            self.add_module('res_unit' + str(group_number) + str(1),
                                residual_block(in_channels=self._filters[group_number-1],
                                               out_channels=self._filters[group_number],
                                               first_stride=self._strides[group_number-1],
                                               is_first_block=True))

            # loop over further residual blocks of this group
            for residual_block_number in range(1, num_residual_blocks):
                self.add_module('res_unit' + str(group_number) + str(residual_block_number+1),
                                residual_block(in_channels=self._filters[group_number], out_channels=self._filters[group_number]))
        # last layer
        self.add_module('bn', nn.BatchNorm2d(self._filters[3], momentum = bn_momentum))
        self.add_module('relu', nn.ReLU())
        self.add_module('avg_pool', nn.AvgPool2d(8))

        # reshape and dense layer
        self.add_module('flatten', flatten())
        self.add_module('dense', nn.Linear(in_features=self._filters[3], out_features=num_outputs))

        # initialisation
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if initializations is not None and 'Conv2d' in initializations:
                    nn.init.constant_(module.bias, 0.0)
                    (eval(initializations['Conv2d'][0])(*[module.weight, *initializations['Conv2d'][1:]]))
                else:
                    nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0) # gamma
                nn.init.constant_(module.bias, 0.0) # beta
                nn.init.constant_(module.running_mean, 0.0)
                nn.init.constant_(module.running_var, 1.0)
            if isinstance(module, nn.Linear):
                if initializations is not None and 'Linear' in initializations:
                    nn.init.constant_(module.bias, 0.0)
                    (eval(initializations['Linear'][0])(*[module.weight, *initializations['Linear'][1:]]))
                else:
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0.0)

class net_char_rnn(nn.Module):
    def __init__(self, seq_len, hidden_dim, vocab_size, num_layers, initializations=None):
        super(net_char_rnn, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim)
        self.lstm = nn.LSTM(input_size = hidden_dim, hidden_size = hidden_dim, num_layers=num_layers, dropout=0.2, batch_first = True)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=vocab_size)
        # TODO init layers?

    def forward(self, x, state = None):
        """state is a tuple for hidden and cell state for initialisation of the lstm"""
        x = self.embedding(x)
        # if no state is provided, default the state to zeros
        if state is None:
            x, new_state = self.lstm(x)
        else:
            x, new_state = self.lstm(x, state)
        x = self.dense(x)
        return x, new_state

class net_quadratic_deep(nn.Module):
    r"""This arhcitecture creates an output which corresponds to a loss functions of the form

    :math:`0.5* (\theta - x)^T * Q * (\theta - x)`

    with Hessian ``Q`` and "data" ``x`` coming from the quadratic data set, i.e.,
    zero-mean normal.
    The parameters are initialized to 1.
"""

    def __init__(self, dim, Hessian, initializations=None):
        """Args:
            dim (int): Number of parameters of the network (Dimension of the quadratic problem).
            Hessian (np.array): The matrix for the quadratic form."""

        super(net_quadratic_deep, self).__init__()
        self.theta = nn.Parameter(torch.ones(dim, requires_grad = True))
        self.Hessian = Hessian

    def forward(self, x):
        q = self.theta - x
        out_batched = 0.5*torch.diag(torch.mm(q, torch.mm(self.Hessian, torch.transpose(q, 0, 1))))

        return out_batched


class net_mlp(nn.Sequential):
    """  A basic MLP architecture. The network is build as follows:

    - Four fully-connected layers with ``1000``, ``500``,``100`` and ``num_outputs``
      units per layer, where ``num_outputs`` is the number of ouputs (i.e. class labels).
    - The first three layers use ReLU activation, and the last one a softmax
      activation.
    - The biases are initialized to ``0.0`` and the weight matrices with
      truncated normal (standard deviation of ``3e-2``)"""
    def __init__(self, num_outputs, initializations=None):
        super(net_mlp, self).__init__()

        self.add_module('flatten', flatten())
        self.add_module('dense1', nn.Linear(784, 1000))
        self.add_module('relu1', nn.ReLU())
        self.add_module('dense2', nn.Linear(1000, 500))
        self.add_module('relu2', nn.ReLU())
        self.add_module('dense3', nn.Linear(500, 100))
        self.add_module('relu3', nn.ReLU())
        self.add_module('dense4', nn.Linear(100, num_outputs))

        for module in self.modules():
            if isinstance(module, nn.Linear):
                if initializations is not None and 'Linear' in initializations:
                    nn.init.constant_(module.bias, 0.0)
                    (eval(initializations['Linear'][0])(*[module.weight, *initializations['Linear'][1:]]))
                else:
                    nn.init.constant_(module.bias, 0.0)
                    module.weight.data = _truncated_normal_init(module.weight.data, mean = 0, stddev=3e-2)



class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super().__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class Bottleneck_DenseNet(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


def DenseNet_Cifar():
    return DenseNet(Bottleneck_DenseNet, [6,12,24,16], growth_rate=12, num_classes=10)


class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10, initializations=None):
        super().__init__()
        block = BasicBlock
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.do = nn.Dropout(0.5)

        if initializations is not None and 'Conv2d' in initializations:
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    (eval(initializations['Conv2d'][0])(*[module.weight, *initializations['Conv2d'][1:]]))
                # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                #     nn.init.constant_(m.weight, 1)
                #     nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #out = self.do(out)
        out = self.linear(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def ResNet18():
    return ResNet([2, 2, 2, 2], num_classes=10)

def ResNet34(initializations=None):
    return ResNet([3, 4, 6, 3], num_classes=10, initializations=initializations)
