# -*- coding: utf-8 -*-

from torch import nn
from .testproblems_modules import resnet34
from ..datasets.cifar10 import cifar10
from .testproblem import TestProblem


class cifar10_resi34(TestProblem):

    def __init__(self, batch_size, weight_decay=0.002):
        super(cifar10_resi34, self).__init__(batch_size, weight_decay)

    def set_up(self, initializations=None):
        """Set up the vanilla CNN test problem on Cifar-10."""
        self.data = cifar10(self._batch_size)
        self.loss_function = nn.CrossEntropyLoss
        self.net = resnet34()
        self.net.to(self._device)
        self.regularization_groups = self.get_regularization_groups()

    def get_regularization_groups(self):
        """Creates regularization groups for the parameters.

        Returns:
            dict: A dictionary where the key is the regularization factor and the value is a list of parameters.
        """
        no, l2 = 0.0, self._weight_decay
        group_dict = {no: [], l2: []}

        for parameters_name, parameters in self.net.named_parameters():
            # penalize only the non bias layer parameters
            if 'bias' not in parameters_name:
                group_dict[l2].append(parameters)
            else:
                group_dict[no].append(parameters)
        return group_dict
