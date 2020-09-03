# -*- coding: utf-8 -*-
"""A vanilla CNN architecture for CIFAR-10."""

from torch import nn
import torch
from .testproblems_modules import ResNet34
from ..datasets.cifar100 import cifar100
from .testproblem import TestProblem
from .testproblem import UnregularizedTestproblem


class cifar100_resnet34(UnregularizedTestproblem):

    def __init__(self, batch_size):

        super(cifar100_resnet34, self).__init__(batch_size)

    def set_up(self, initializations=None):
        self.data = cifar100(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = ResNet34(num_classes=100, initializations=initializations)
        self.net.to(self._device)
        self.net = torch.nn.DataParallel(self.net)
        self.regularization_groups = self.get_regularization_groups()
