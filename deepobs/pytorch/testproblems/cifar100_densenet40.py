# -*- coding: utf-8 -*-
"""A vanilla CNN architecture for CIFAR-10."""

from torch import nn
import torch
from .testproblems_modules import DenseNet
from ..datasets.cifar100 import cifar100
from .testproblem import TestProblem
from .testproblem import UnregularizedTestproblem


class cifar100_densenet40(UnregularizedTestproblem):


    def __init__(self, batch_size):

        super(cifar100_densenet40, self).__init__(batch_size)

    def set_up(self, initializations=None):
        """Set up the vanilla CNN test problem on Cifar-10."""
        self.data = cifar10(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = DenseNet(num_classes=100, initializations=initializations)
        self.net.to(self._device)
        self.net = torch.nn.DataParallel(self.net)
        self.regularization_groups = self.get_regularization_groups()
