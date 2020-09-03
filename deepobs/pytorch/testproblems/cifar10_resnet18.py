from torch import nn
import torch
from .testproblems_modules import ResNet34
from ..datasets.cifar10 import cifar10
from .testproblem import TestProblem
from .testproblem import UnregularizedTestproblem


class cifar10_resnet18(UnregularizedTestproblem):

    def __init__(self, batch_size):

        super(cifar10_resnet18, self).__init__(batch_size)

    def set_up(self, initializations=None):
        self.data = cifar10(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = ResNet18(initializations=initializations)
        self.net.to(self._device)
        self.net = torch.nn.DataParallel(self.net)
        self.regularization_groups = self.get_regularization_groups()