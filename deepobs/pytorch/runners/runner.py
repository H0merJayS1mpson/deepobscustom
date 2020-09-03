"""Module implementing StandardRunner."""

from __future__ import print_function
import torch
import importlib
import abc
from deepobs import config as global_config
from .. import config
from .. import testproblems
from . import runner_utils
from deepobs.abstract_runner.abstract_runner import Runner
import numpy as np
import warnings
from random import seed
from copy import deepcopy


class PTRunner(Runner):
    """The abstract class for runner in the pytorch framework."""

    def __init__(self, optimizer_class, hyperparameter_names):
        super(PTRunner, self).__init__(optimizer_class, hyperparameter_names)

    @abc.abstractmethod
    def training(self, tproblem,
                 hyperparams,
                 num_epochs,
                 print_train_iter,
                 train_log_interval,
                 tb_log,
                 tb_log_dir,
                 **training_params):
        return

    @staticmethod
    def create_testproblem(testproblem, initializations, batch_size, weight_decay, random_seed):
        """Sets up the deepobs.pytorch.testproblems.testproblem instance.

        Args:
            testproblem (str): The name of the testproblem.
            batch_size (int): Batch size that is used for training
            weight_decay (float): Regularization factor
            random_seed (int): The random seed of the framework

        Returns:
            deepobs.pytorch.testproblems.testproblem: An instance of deepobs.pytorch.testproblems.testproblem
        """
        # set the seed and GPU determinism
        if config.get_is_deterministic():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Find testproblem by name and instantiate with batch size and weight decay.
        try:
            testproblem_mod = importlib.import_module(testproblem)
            testproblem_cls = getattr(testproblem_mod, testproblem)
            print("Loading local testproblem.")
        except:
            testproblem_cls = getattr(testproblems, testproblem)

        # if the user specified a weight decay, use that one
        if weight_decay is not None:
            tproblem = testproblem_cls(batch_size, weight_decay)
        # else use the default of the testproblem
        else:
            tproblem = testproblem_cls(batch_size)

        # Set up the testproblem.
        tproblem.set_up(initializations)
        return tproblem

    # Wrapper functions for the evaluation phase.
    @staticmethod
    def evaluate(tproblem, phase, get_next_batch=True):
        """Evaluates the performance of the current state of the model
        of the testproblem instance.
        Has to be called in the beggining of every epoch within the
        training method. Returns the losses and accuracies.

        Args:
            tproblem (testproblem): The testproblem instance to evaluate
            phase (str): The phase of the evaluation. Must be one of 'TRAIN', 'VALID' or 'TEST'
        Returns:
            float: The loss of the current state.
            float: The accuracy of the current state.
            :param get_next_batch:

        """

        if phase == 'TEST':
            tproblem.test_init_op()
            msg = "TEST:"
        elif phase == 'TRAIN':
            tproblem.train_eval_init_op()
            msg = "TRAIN:"
        elif phase == 'VALID':
            tproblem.valid_init_op()
            msg = "VALID:"
        # evaluation loop over every batch of the corresponding evaluation set
        loss = 0.0
        accuracy = 0.0
        batchCount = 0.0
        i = 0
        while True:
            try:
                batch_loss, batch_accuracy = tproblem.get_batch_loss_and_accuracy(get_next_batch=get_next_batch)
                batchCount += 1.0
                loss += batch_loss.item()
                accuracy += batch_accuracy
            except StopIteration:
                break

        loss /= batchCount
        accuracy /= batchCount
        if accuracy != 0.0:
            print("{0:s} loss {1:g}, acc {2:f}".format(msg, loss, accuracy))
        else:
            print("{0:s} loss {1:g}".format(msg, loss))

        return loss, accuracy

    def evaluate_all(self,
                     epoch_count,
                     num_epochs,
                     tproblem,
                     train_losses,
                     valid_losses,
                     test_losses,
                     train_accuracies,
                     valid_accuracies,
                     test_accuracies,
                     get_next_batch=True):

        print("********************************")
        print("Evaluating after {0:d} of {1:d} epochs...".format(epoch_count, num_epochs))

        loss_, acc_ = self.evaluate(tproblem, phase='TRAIN', get_next_batch=get_next_batch)
        train_losses.append(loss_)
        train_accuracies.append(acc_)

        loss_, acc_ = self.evaluate(tproblem, phase='VALID', get_next_batch=get_next_batch)
        valid_losses.append(loss_)
        valid_accuracies.append(acc_)

        loss_, acc_ = self.evaluate(tproblem, phase='TEST', get_next_batch=get_next_batch)
        test_losses.append(loss_)
        test_accuracies.append(acc_)

        print("********************************")


class StandardRunner(PTRunner):
    """A standard runner. Can run a normal training loop with fixed
    hyperparams. It should be used as a template to implement custom runners.
    """

    def __init__(self, optimizer_class, hyperparameter_names):
        super(StandardRunner, self).__init__(optimizer_class, hyperparameter_names)

    def training(self,
                 tproblem,
                 hyperparams,
                 num_epochs,
                 print_train_iter,
                 train_log_interval,
                 tb_log,
                 tb_log_dir):

        opt = self._optimizer_class(tproblem.net.parameters(), **hyperparams)

        # Lists to log train/test loss and accuracy.
        train_losses = []
        valid_losses = []
        test_losses = []
        train_accuracies = []
        valid_accuracies = []
        test_accuracies = []

        minibatch_train_losses = []

        if tb_log:
            try:
                from torch.utils.tensorboard import SummaryWriter
                summary_writer = SummaryWriter(log_dir=tb_log_dir)
            except ImportError as e:
                warnings.warn('Not possible to use tensorboard for pytorch. Reason: ' + e.msg, RuntimeWarning)
                tb_log = False
        global_step = 0

        for epoch_count in range(num_epochs + 1):
            # Evaluate at beginning of epoch.
            self.evaluate_all(epoch_count,
                              num_epochs,
                              tproblem,
                              train_losses,
                              valid_losses,
                              test_losses,
                              train_accuracies,
                              valid_accuracies,
                              test_accuracies)

            # Break from train loop after the last round of evaluation
            if epoch_count == num_epochs:
                break

            ### Training ###

            # set to training mode
            tproblem.train_init_op()
            batch_count = 0
            while True:
                try:
                    opt.zero_grad()
                    batch_loss, _ = tproblem.get_batch_loss_and_accuracy()
                    batch_loss.backward()
                    opt.step()

                    if batch_count % train_log_interval == 0:
                        minibatch_train_losses.append(batch_loss.item())
                        if print_train_iter:
                            print("Epoch {0:d}, step {1:d}: loss {2:g}".format(epoch_count, batch_count, batch_loss))
                        if tb_log:
                            summary_writer.add_scalar('loss', batch_loss.item(), global_step)

                    batch_count += 1
                    global_step += 1

                except StopIteration:
                    break

            if not np.isfinite(batch_loss.item()):
                self._abort_routine(
                    epoch_count,
                    num_epochs,
                    train_losses,
                    valid_losses,
                    test_losses,
                    train_accuracies,
                    valid_accuracies,
                    test_accuracies,
                    minibatch_train_losses)
                break
            else:
                continue

        if tb_log:
            summary_writer.close()
        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            'valid_losses': valid_losses,
            "test_losses": test_losses,
            "minibatch_train_losses": minibatch_train_losses,
            "train_accuracies": train_accuracies,
            'valid_accuracies': valid_accuracies,
            "test_accuracies": test_accuracies
        }

        return output


class LearningRateScheduleRunner(PTRunner):
    """A runner for learning rate schedules. Can run a normal training loop with fixed hyperparams or a learning rate
    schedule. It should be used as a template to implement custom runners.
    """

    def __init__(self, optimizer_class, hyperparameter_names):

        super(LearningRateScheduleRunner, self).__init__(optimizer_class, hyperparameter_names)

    def _add_training_params_to_argparse(self, parser, args, training_params):
        try:
            args['lr_sched_epochs'] = training_params['lr_sched_epochs']
        except KeyError:
            parser.add_argument(
                "--lr_sched_epochs",
                nargs="+",
                type=int,
                help="""One or more epoch numbers (positive integers) that mark
          learning rate changes. The base learning rate has to be passed via
          '--learing_rate' and the factors by which to change have to be passed
          via '--lr_sched_factors'. Example: '--lr 0.3 --lr_sched_epochs 50 100
          --lr_sched_factors 0.1 0.01' will start with a learning rate of 0.3,
          then decrease to 0.1*0.3=0.03 after training for 50 epochs, and
          decrease to 0.01*0.3=0.003' after training for 100 epochs.""")

        try:
            args['lr_sched_factors'] = training_params['lr_sched_factors']
        except KeyError:
            parser.add_argument(
                "--lr_sched_factors",
                nargs="+",
                type=float,
                help="""One or more factors (floats) by which to change the learning
          rate. The base learning rate has to be passed via '--learing_rate' and
          the epochs at which to change the learning rate have to be passed via
          '--lr_sched_factors'. Example: '--lr 0.3 --lr_sched_epochs 50 100
          --lr_sched_factors 0.1 0.01' will start with a learning rate of 0.3,
          then decrease to 0.1*0.3=0.03 after training for 50 epochs, and
          decrease to 0.01*0.3=0.003' after training for 100 epochs.""")

    def training(self,
                 tproblem,
                 hyperparams,
                 num_epochs,
                 print_train_iter,
                 train_log_interval,
                 tb_log,
                 tb_log_dir,
                 # the following are the training_params
                 lr_sched_epochs=None,
                 lr_sched_factors=None):
        """Performs the training and stores the metrices.

        Args:
            tproblem (deepobs.[tensorflow/pytorch].testproblems.testproblem): The testproblem instance to train on.
            hyperparams (dict): The optimizer hyperparameters to use for the training.
            num_epochs (int): The number of training epochs.
            print_train_iter (bool): Whether to print the training progress at every train_log_interval
            train_log_interval (int): Mini-batch interval for logging.
            tb_log (bool): Whether to use tensorboard logging or not
            tb_log_dir (str): The path where to save tensorboard events.
            lr_sched_epochs (list): The epochs where to adjust the learning rate.
            lr_sched_factors (list): The corresponding factors by which to adjust the learning rate.

        Returns:
            dict: The logged metrices. Is of the form: \
                {'test_losses' : [...], \
                'valid_losses': [...], \
                 'train_losses': [...],  \
                 'test_accuracies': [...], \
                 'valid_accuracies': [...], \
                 'train_accuracies': [...] \
                 } \
            where the metrices values are lists that were filled during training.
        """

        opt = self._optimizer_class(tproblem.net.parameters(), **hyperparams)
        if lr_sched_epochs is not None:
            lr_schedule = runner_utils.make_lr_schedule(optimizer=opt, lr_sched_epochs=lr_sched_epochs,
                                                        lr_sched_factors=lr_sched_factors)

        # Lists to log train/test loss and accuracy.
        train_losses = []
        valid_losses = []
        test_losses = []
        train_accuracies = []
        valid_accuracies = []
        test_accuracies = []

        minibatch_train_losses = []

        for epoch_count in range(num_epochs + 1):
            # Evaluate at beginning of epoch.
            self.evaluate_all(epoch_count,
                              num_epochs,
                              tproblem,
                              train_losses,
                              valid_losses,
                              test_losses,
                              train_accuracies,
                              valid_accuracies,
                              test_accuracies)

            # Break from train loop after the last round of evaluation
            if epoch_count == num_epochs:
                break

            ### Training ###
            if lr_sched_epochs is not None:
                # get the next learning rate
                lr_schedule.step(epoch_count)

                if epoch_count in lr_sched_epochs:
                    print("Setting learning rate to {0}".format(lr_schedule.get_lr()))

            # set to training mode
            tproblem.train_init_op()
            batch_count = 0
            while True:
                try:
                    opt.zero_grad()
                    batch_loss, _ = tproblem.get_batch_loss_and_accuracy()

                    batch_loss.backward()
                    opt.step()

                    if batch_count % train_log_interval == 0:
                        minibatch_train_losses.append(batch_loss.item())
                        if print_train_iter:
                            print("Epoch {0:d}, step {1:d}: loss {2:g}".format(epoch_count, batch_count, batch_loss))
                    batch_count += 1

                except StopIteration:
                    break

            # break from training if it goes wrong
            if not np.isfinite(batch_loss.item()):
                self._abort_routine(epoch_count,
                                    num_epochs,
                                    train_losses,
                                    valid_losses,
                                    test_losses,
                                    train_accuracies,
                                    valid_accuracies,
                                    test_accuracies)
                break
            else:
                continue

        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "test_losses": test_losses,
            "minibatch_train_losses": minibatch_train_losses,
            "train_accuracies": train_accuracies,
            'valid_accuracies': valid_accuracies,
            "test_accuracies": test_accuracies
        }

        return output


class CustomRunner(PTRunner):
    """A standard runner. Can run a normal training loop with fixed
    hyperparams. It should be used as a template to implement custom runners.
    """

    def evaluate(tproblem, phase, get_next_batch=True):
        """Evaluates the performance of the current state of the model
        of the testproblem instance.
        Has to be called in the beggining of every epoch within the
        training method. Returns the losses and accuracies.

        Args:
            tproblem (testproblem): The testproblem instance to evaluate
            phase (str): The phase of the evaluation. Must be one of 'TRAIN', 'VALID' or 'TEST'
        Returns:
            float: The loss of the current state.
            float: The accuracy of the current state.
            :param get_next_batch:

        """

        if phase == 'TEST':
            tproblem.test_init_op()
            msg = "TEST:"
        elif phase == 'TRAIN':
            tproblem.train_eval_init_op()
            msg = "TRAIN:"
        elif phase == 'VALID':
            tproblem.valid_init_op()
            msg = "VALID:"
        # evaluation loop over every batch of the corresponding evaluation set
        loss = 0.0
        accuracy = 0.0
        batchCount = 0.0
        while True:
            try:
                batch_loss, batch_accuracy = tproblem.get_batch_loss_and_accuracy(get_next_batch=get_next_batch)
                batchCount += 1.0
                loss += batch_loss.item()
                accuracy += batch_accuracy
            except StopIteration:
                break

        loss /= batchCount
        accuracy /= batchCount
        if accuracy != 0.0:
            print("{0:s} loss {1:g}, acc {2:f}".format(msg, loss, accuracy))
        else:
            print("{0:s} loss {1:g}".format(msg, loss))

        return loss, accuracy

    def evaluate_all(self,
                     epoch_count,
                     num_epochs,
                     tproblem,
                     train_losses,
                     valid_losses,
                     test_losses,
                     train_accuracies,
                     valid_accuracies,
                     test_accuracies,
                     get_next_batch=True):

        print("********************************")
        print("Evaluating after {0:d} of {1:d} epochs...".format(epoch_count, num_epochs))

        loss_, acc_ = CustomRunner.evaluate(tproblem, phase='TRAIN', get_next_batch=get_next_batch)
        train_losses.append(loss_)
        train_accuracies.append(acc_)

        loss_, acc_ = CustomRunner.evaluate(tproblem, phase='VALID', get_next_batch=get_next_batch)
        valid_losses.append(loss_)
        valid_accuracies.append(acc_)

        loss_, acc_ = CustomRunner.evaluate(tproblem, phase='TEST', get_next_batch=get_next_batch)
        test_losses.append(loss_)
        test_accuracies.append(acc_)

        print("********************************")

    def __init__(self, optimizer_class, hyperparameter_names):
        super(CustomRunner, self).__init__(optimizer_class, hyperparameter_names)

    def create_testproblem(self, testproblem, initializations, batch_size, weight_decay, random_seed):
        """Sets up the deepobs.pytorch.testproblems.testproblem instance.

        Args:
            testproblem (str): The name of the testproblem.
            batch_size (int): Batch size that is used for training
            weight_decay (float): Regularization factor
            random_seed (int): The random seed of the framework
            :param initializations: dictionary of the initialazation Methods per layer-Name

        Returns:
            deepobs.pytorch.testproblems.testproblem: An instance of deepobs.pytorch.testproblems.testproblem
        """
        # set the seed and GPU determinism
        if config.get_is_deterministic():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Find testproblem by name and instantiate with batch size and weight decay.
        try:
            testproblem_mod = importlib.import_module(testproblem)
            testproblem_cls = getattr(testproblem_mod, testproblem)
            print("Loading local testproblem.")
        except:
            testproblem_cls = getattr(testproblems, testproblem)

        # if the user specified a weight decay, use that one
        if weight_decay is not None:
            tproblem = testproblem_cls(batch_size, weight_decay)
        # else use the default of the testproblem
        else:
            tproblem = testproblem_cls(batch_size)

        # Set up the testproblem.
        tproblem.set_up(initializations)
        return tproblem

    def training(self,
                 tproblem,
                 hyperparams,
                 num_epochs,
                 print_train_iter,
                 train_log_interval,
                 tb_log,
                 tb_log_dir):

        opt = self._optimizer_class(tproblem.net.parameters(), **hyperparams)

        # Lists to log train/test loss and accuracy.
        train_losses = []
        valid_losses = []
        test_losses = []
        train_accuracies = []
        valid_accuracies = []
        test_accuracies = []
        minibatch_train_losses = []

        if tb_log:
            try:
                from torch.utils.tensorboard import SummaryWriter
                summary_writer = SummaryWriter(log_dir=tb_log_dir)
            except ImportError as e:
                warnings.warn('Not possible to use tensorboard for pytorch. Reason: ' + e.msg, RuntimeWarning)
                tb_log = False
        global_step = 0

        for epoch_count in range(num_epochs + 1):
            # Evaluate at beginning of epoch.
            self.evaluate_all(epoch_count,
                              num_epochs,
                              tproblem,
                              train_losses,
                              valid_losses,
                              test_losses,
                              train_accuracies,
                              valid_accuracies,
                              test_accuracies)

            # Break from train loop after the last round of evaluation
            if epoch_count == num_epochs:
                break

            ### Training ###

            # set to training mode
            tproblem.train_init_op()
            batch_count = 0
            while True:
                try:
                    #opt.zero_grad()
                    def closure(backward=True, get_next_batch=True):
                        opt.zero_grad()
                        batch_loss, _ = tproblem.get_batch_loss_and_accuracy(get_next_batch=get_next_batch)
                        if backward:
                            batch_loss.backward()
                        return batch_loss
                    batch_loss = opt.step(closure)

                    if batch_count % train_log_interval == 0:
                        minibatch_train_losses.append(batch_loss.item())
                        if print_train_iter:
                            print("Epoch {0:d}, step {1:d}: loss {2:g}".format(epoch_count, batch_count, batch_loss))
                        if tb_log:
                            summary_writer.add_scalar('loss', batch_loss.item(), global_step)

                    batch_count += 1
                    global_step += 1

                except StopIteration:
                    break

            if not np.isfinite(batch_loss.item()):
                self._abort_routine(
                    epoch_count,
                    num_epochs,
                    train_losses,
                    valid_losses,
                    test_losses,
                    train_accuracies,
                    valid_accuracies,
                    test_accuracies,
                    minibatch_train_losses)
                break
            else:
                continue

        if tb_log:
            summary_writer.close()
        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            'valid_losses': valid_losses,
            "test_losses": test_losses,
            "minibatch_train_losses": minibatch_train_losses,
            "train_accuracies": train_accuracies,
            'valid_accuracies': valid_accuracies,
            "test_accuracies": test_accuracies
        }
        return output


class CustomLearningRateScheduleRunner(PTRunner):
    """A runner for learning rate schedules. Can run a normal training loop with fixed hyperparams or a learning rate
    schedule. It should be used as a template to implement custom runners.
    """

    def __init__(self, optimizer_class, hyperparameter_names):

        super(CustomLearningRateScheduleRunner, self).__init__(optimizer_class, hyperparameter_names)

    def _add_training_params_to_argparse(self, parser, args, training_params):
        try:
            args['lr_sched_epochs'] = training_params['lr_sched_epochs']
        except KeyError:
            parser.add_argument(
                "--lr_sched_epochs",
                nargs="+",
                type=int,
                help="""One or more epoch numbers (positive integers) that mark
                learning rate changes. The base learning rate has to be passed via
                '--learing_rate' and the factors by which to change have to be passed
                via '--lr_sched_factors'. Example: '--lr 0.3 --lr_sched_epochs 50 100
                --lr_sched_factors 0.1 0.01' will start with a learning rate of 0.3,
                then decrease to 0.1*0.3=0.03 after training for 50 epochs, and
                decrease to 0.01*0.3=0.003' after training for 100 epochs.""")

        try:
            args['lr_sched_factors'] = training_params['lr_sched_factors']
        except KeyError:
            parser.add_argument(
                "--lr_sched_factors",
                nargs="+",
                type=float,
                help=
                """One or more factors (floats) by which to change the learning
                rate. The base learning rate has to be passed via '--learing_rate' and
                the epochs at which to change the learning rate have to be passed via
                '--lr_sched_factors'. Example: '--lr 0.3 --lr_sched_epochs 50 100
                --lr_sched_factors 0.1 0.01' will start with a learning rate of 0.3,
                then decrease to 0.1*0.3=0.03 after training for 50 epochs, and
                decrease to 0.01*0.3=0.003' after training for 100 epochs.""")

    def create_testproblem(self, testproblem, initializations, batch_size, weight_decay, random_seed):
        """Sets up the deepobs.pytorch.testproblems.testproblem instance.

        Args:
            testproblem (str): The name of the testproblem.
            batch_size (int): Batch size that is used for training
            weight_decay (float): Regularization factor
            random_seed (int): The random seed of the framework
            :param initializations: dictionary of the initialazation Methods per layer-Name

        Returns:
            deepobs.pytorch.testproblems.testproblem: An instance of deepobs.pytorch.testproblems.testproblem
        """
        # set the seed and GPU determinism
        if config.get_is_deterministic():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Find testproblem by name and instantiate with batch size and weight decay.
        try:
            testproblem_mod = importlib.import_module(testproblem)
            testproblem_cls = getattr(testproblem_mod, testproblem)
            print("Loading local testproblem.")
        except:
            testproblem_cls = getattr(testproblems, testproblem)

        # if the user specified a weight decay, use that one
        if weight_decay is not None:
            tproblem = testproblem_cls(batch_size, weight_decay)
        # else use the default of the testproblem
        else:
            tproblem = testproblem_cls(batch_size)

        # Set up the testproblem.
        tproblem.set_up(initializations)
        return tproblem

    def evaluate(tproblem, phase, get_next_batch=True):
        """Evaluates the performance of the current state of the model
        of the testproblem instance.
        Has to be called in the beggining of every epoch within the
        training method. Returns the losses and accuracies.

        Args:
            tproblem (testproblem): The testproblem instance to evaluate
            phase (str): The phase of the evaluation. Must be one of 'TRAIN', 'VALID' or 'TEST'
        Returns:
            float: The loss of the current state.
            float: The accuracy of the current state.
            :param get_next_batch:

        """

        if phase == 'TEST':
            tproblem.test_init_op()
            msg = "TEST:"
        elif phase == 'TRAIN':
            tproblem.train_eval_init_op()
            msg = "TRAIN:"
        elif phase == 'VALID':
            tproblem.valid_init_op()
            msg = "VALID:"
        # evaluation loop over every batch of the corresponding evaluation set
        loss = 0.0
        accuracy = 0.0
        batchCount = 0.0
        while True:
            try:
                batch_loss, batch_accuracy = tproblem.get_batch_loss_and_accuracy(get_next_batch=get_next_batch)
                batchCount += 1.0
                loss += batch_loss.item()
                accuracy += batch_accuracy
            except StopIteration:
                break

        loss /= batchCount
        accuracy /= batchCount
        if accuracy != 0.0:
            print("{0:s} loss {1:g}, acc {2:f}".format(msg, loss, accuracy))
        else:
            print("{0:s} loss {1:g}".format(msg, loss))

        return loss, accuracy


    def evaluate_all(self,
                     epoch_count,
                     num_epochs,
                     tproblem,
                     train_losses,
                     valid_losses,
                     test_losses,
                     train_accuracies,
                     valid_accuracies,
                     test_accuracies,
                     get_next_batch=True):

        print("********************************")
        print("Evaluating after {0:d} of {1:d} epochs...".format(epoch_count, num_epochs))

        loss_, acc_ = CustomLearningRateScheduleRunner.evaluate(tproblem, phase='TRAIN', get_next_batch=get_next_batch)
        train_losses.append(loss_)
        train_accuracies.append(acc_)

        loss_, acc_ = CustomLearningRateScheduleRunner.evaluate(tproblem, phase='VALID', get_next_batch=get_next_batch)
        valid_losses.append(loss_)
        valid_accuracies.append(acc_)

        loss_, acc_ = CustomLearningRateScheduleRunner.evaluate(tproblem, phase='TEST', get_next_batch=get_next_batch)
        test_losses.append(loss_)
        test_accuracies.append(acc_)

        print("********************************")

    def training(self,
                 tproblem,
                 hyperparams,
                 num_epochs,
                 print_train_iter,
                 train_log_interval,
                 tb_log,
                 tb_log_dir,
                 # the following are the training_params
                 lr_sched_epochs=None,
                 lr_sched_factors=None):
        """Performs the training and stores the metrices.

        Args:
            tproblem (deepobs.[tensorflow/pytorch].testproblems.testproblem): The testproblem instance to train on.
            hyperparams (dict): The optimizer hyperparameters to use for the training.
            num_epochs (int): The number of training epochs.
            print_train_iter (bool): Whether to print the training progress at every train_log_interval
            train_log_interval (int): Mini-batch interval for logging.
            tb_log (bool): Whether to use tensorboard logging or not
            tb_log_dir (str): The path where to save tensorboard events.
            lr_sched_epochs (list): The epochs where to adjust the learning rate.
            lr_sched_factors (list): The corresponding factors by which to adjust the learning rate.

        Returns:
            dict: The logged metrices. Is of the form: \
                {'test_losses' : [...], \
                'valid_losses': [...], \
                 'train_losses': [...],  \
                 'test_accuracies': [...], \
                 'valid_accuracies': [...], \
                 'train_accuracies': [...] \
                 } \
            where the metrices values are lists that were filled during training.
        """

        opt = self._optimizer_class(tproblem.net.parameters(), **hyperparams)

        if lr_sched_epochs is not None:
            lr_schedule = runner_utils.make_lr_schedule(optimizer=opt, lr_sched_epochs=lr_sched_epochs,
                                                        lr_sched_factors=lr_sched_factors)

        # Lists to log train/test loss and accuracy.
        train_losses = []
        valid_losses = []
        test_losses = []
        train_accuracies = []
        valid_accuracies = []
        test_accuracies = []
        minibatch_train_losses = []

        if tb_log:
            try:
                from torch.utils.tensorboard import SummaryWriter
                summary_writer = SummaryWriter(log_dir=tb_log_dir)
            except ImportError as e:
                warnings.warn('Not possible to use tensorboard for pytorch. Reason: ' + e.msg, RuntimeWarning)
                tb_log = False
        global_step = 0

        for epoch_count in range(num_epochs + 1):
            # Evaluate at beginning of epoch.
            self.evaluate_all(epoch_count,
                              num_epochs,
                              tproblem,
                              train_losses,
                              valid_losses,
                              test_losses,
                              train_accuracies,
                              valid_accuracies,
                              test_accuracies)

            # Break from train loop after the last round of evaluation
            if epoch_count == num_epochs:
                break

            ### Training ###
            if lr_sched_epochs is not None:
                # get the next learning rate
                lr_schedule.step(epoch_count)

                if epoch_count in lr_sched_epochs:
                    print("Setting learning rate to {0}".format(lr_schedule.get_lr()))
            # set to training mode
            tproblem.train_init_op()
            batch_count = 0
            while True:
                try:
                    opt.zero_grad()

                    def closure(backward=True, get_next_batch=True):
                        # opt.zero_grad()
                        batch_loss, _ = tproblem.get_batch_loss_and_accuracy(get_next_batch=get_next_batch)
                        if backward:
                            batch_loss.backward()
                        return batch_loss

                    batch_loss = opt.step(closure)

                    if batch_count % train_log_interval == 0:
                        minibatch_train_losses.append(batch_loss.item())
                        if print_train_iter:
                            print(
                                "Epoch {0:d}, step {1:d}: loss {2:g}".format(epoch_count, batch_count, batch_loss))
                        if tb_log:
                            summary_writer.add_scalar('loss', batch_loss.item(), global_step)

                    batch_count += 1
                    global_step += 1

                except StopIteration:
                    break

            if not np.isfinite(batch_loss.item()):
                self._abort_routine(
                    epoch_count,
                    num_epochs,
                    train_losses,
                    valid_losses,
                    test_losses,
                    train_accuracies,
                    valid_accuracies,
                    test_accuracies,
                    minibatch_train_losses)
                break
            else:
                continue

        if tb_log:
            summary_writer.close()
        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            'valid_losses': valid_losses,
            "test_losses": test_losses,
            "minibatch_train_losses": minibatch_train_losses,
            "train_accuracies": train_accuracies,
            'valid_accuracies': valid_accuracies,
            "test_accuracies": test_accuracies
        }
        return output
        # opt = self._optimizer_class(tproblem.net.parameters(), **hyperparams)
        #
        # if lr_sched_epochs is not None:
        #     lr_schedule = runner_utils.make_lr_schedule(optimizer=opt, lr_sched_epochs=lr_sched_epochs,
        #                                                 lr_sched_factors=lr_sched_factors)
        #
        # # Lists to log train/test loss and accuracy.
        # train_losses = []
        # valid_losses = []
        # test_losses = []
        # train_accuracies = []
        # valid_accuracies = []
        # test_accuracies = []
        #
        # minibatch_train_losses = []
        #
        # for epoch_count in range(num_epochs + 1):
        #     # Evaluate at beginning of epoch.
        #     self.evaluate_all(epoch_count,
        #                       num_epochs,
        #                       tproblem,
        #                       train_losses,
        #                       valid_losses,
        #                       test_losses,
        #                       train_accuracies,
        #                       valid_accuracies,
        #                       test_accuracies)
        #
        #     # Break from train loop after the last round of evaluation
        #     if epoch_count == num_epochs:
        #         break
        #
        #     ### Training ###
        #     if lr_sched_epochs is not None:
        #         # get the next learning rate
        #         lr_schedule.step(epoch_count)
        #
        #         if epoch_count in lr_sched_epochs:
        #             print("Setting learning rate to {0}".format(lr_schedule.get_lr()))
        #
        #     # set to training mode
        #     tproblem.train_init_op()
        #     batch_count = 0
        #     while True:
        #         try:
        #             def closure(backward=True, get_next_batch=True):
        #                 # opt.zero_grad()
        #                 batch_loss, _ = tproblem.get_batch_loss_and_accuracy(get_next_batch=get_next_batch)
        #                 if backward:
        #                     batch_loss.backward()
        #                 return batch_loss
        #
        #             batch_loss = opt.step(closure)
        #
        #
        #             if batch_count % train_log_interval == 0:
        #                 minibatch_train_losses.append(batch_loss.item())
        #                 if print_train_iter:
        #                     print("Epoch {0:d}, step {1:d}: loss {2:g}".format(epoch_count, batch_count, batch_loss))
        #             batch_count += 1
        #
        #         except StopIteration:
        #             break
        #
        #     # break from training if it goes wrong
        #     if not np.isfinite(batch_loss.item()):
        #         self._abort_routine(epoch_count,
        #                             num_epochs,
        #                             train_losses,
        #                             valid_losses,
        #                             test_losses,
        #                             train_accuracies,
        #                             valid_accuracies,
        #                             test_accuracies)
        #         break
        #     else:
        #         continue
        #
        # # Put results into output dictionary.
        # output = {
        #     "train_losses": train_losses,
        #     "valid_losses": valid_losses,
        #     "test_losses": test_losses,
        #     "minibatch_train_losses": minibatch_train_losses,
        #     "train_accuracies": train_accuracies,
        #     'valid_accuracies': valid_accuracies,
        #     "test_accuracies": test_accuracies
        # }
        #
        # return output

class PalRunner(PTRunner):
    """A standard runner. Can run a normal training loop with fixed
    hyperparams. It should be used as a template to implement custom runners.
    """

    def evaluate(tproblem, phase, get_next_batch=True):
        """Evaluates the performance of the current state of the model
        of the testproblem instance.
        Has to be called in the beggining of every epoch within the
        training method. Returns the losses and accuracies.

        Args:
            tproblem (testproblem): The testproblem instance to evaluate
            phase (str): The phase of the evaluation. Must be one of 'TRAIN', 'VALID' or 'TEST'
        Returns:
            float: The loss of the current state.
            float: The accuracy of the current state.
            :param get_next_batch:

        """

        if phase == 'TEST':
            tproblem.test_init_op()
            msg = "TEST:"
        elif phase == 'TRAIN':
            tproblem.train_eval_init_op()
            msg = "TRAIN:"
        elif phase == 'VALID':
            tproblem.valid_init_op()
            msg = "VALID:"
        # evaluation loop over every batch of the corresponding evaluation set
        loss = 0.0
        accuracy = 0.0
        batchCount = 0.0
        i = 0
        while True:
            try:
                batch_loss, batch_accuracy = tproblem.get_batch_loss_and_accuracy(get_next_batch=get_next_batch)
                batchCount += 1.0
                loss += batch_loss.item()
                accuracy += batch_accuracy
            except StopIteration:
                break

        loss /= batchCount
        accuracy /= batchCount
        if accuracy != 0.0:
            print("{0:s} loss {1:g}, acc {2:f}".format(msg, loss, accuracy))
        else:
            print("{0:s} loss {1:g}".format(msg, loss))

        return loss, accuracy

    def evaluate_all(self,
                     epoch_count,
                     num_epochs,
                     tproblem,
                     train_losses,
                     valid_losses,
                     test_losses,
                     train_accuracies,
                     valid_accuracies,
                     test_accuracies,
                     get_next_batch=True):

        print("********************************")
        print("Evaluating after {0:d} of {1:d} epochs...".format(epoch_count, num_epochs))

        loss_, acc_ = CustomRunner.evaluate(tproblem, phase='TRAIN', get_next_batch=get_next_batch)
        train_losses.append(loss_)
        train_accuracies.append(acc_)

        loss_, acc_ = CustomRunner.evaluate(tproblem, phase='VALID', get_next_batch=get_next_batch)
        valid_losses.append(loss_)
        valid_accuracies.append(acc_)

        loss_, acc_ = CustomRunner.evaluate(tproblem, phase='TEST', get_next_batch=get_next_batch)
        test_losses.append(loss_)
        test_accuracies.append(acc_)

        print("********************************")

    def __init__(self, optimizer_class, hyperparameter_names):
        super(PalRunner, self).__init__(optimizer_class, hyperparameter_names)

    def create_testproblem(self, testproblem, initializations, batch_size, weight_decay, random_seed):
        """Sets up the deepobs.pytorch.testproblems.testproblem instance.

        Args:
            testproblem (str): The name of the testproblem.
            batch_size (int): Batch size that is used for training
            weight_decay (float): Regularization factor
            random_seed (int): The random seed of the framework
            :param initializations: dictionary of the initialazation Methods per layer-Name

        Returns:
            deepobs.pytorch.testproblems.testproblem: An instance of deepobs.pytorch.testproblems.testproblem
        """
        # set the seed and GPU determinism
        if config.get_is_deterministic():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Find testproblem by name and instantiate with batch size and weight decay.
        try:
            testproblem_mod = importlib.import_module(testproblem)
            testproblem_cls = getattr(testproblem_mod, testproblem)
            print("Loading local testproblem.")
        except:
            testproblem_cls = getattr(testproblems, testproblem)

        # if the user specified a weight decay, use that one
        if weight_decay is not None:
            tproblem = testproblem_cls(batch_size, weight_decay)
        # else use the default of the testproblem
        else:
            tproblem = testproblem_cls(batch_size)

        # Set up the testproblem.
        tproblem.set_up(initializations)
        return tproblem

    def training(self,
                 tproblem,
                 hyperparams,
                 num_epochs,
                 print_train_iter,
                 train_log_interval,
                 tb_log,
                 tb_log_dir):

        opt = self._optimizer_class(tproblem.net.parameters(), **hyperparams)

        # Lists to log train/test loss and accuracy.
        train_losses = []
        valid_losses = []
        test_losses = []
        train_accuracies = []
        valid_accuracies = []
        test_accuracies = []
        minibatch_train_losses = []

        print(type(tproblem.data))
        global_step = 0
        net = tproblem.net
        criterion = tproblem.loss_function()
        device = tproblem._device
        data = tproblem.data

        if tb_log:
            try:
                from torch.utils.tensorboard import SummaryWriter
                summary_writer = SummaryWriter(log_dir=tb_log_dir)
            except ImportError as e:
                warnings.warn('Not possible to use tensorboard for pytorch. Reason: ' + e.msg, RuntimeWarning)
                tb_log = False

        def valid(epoch_):
            tproblem.valid_init_op()
            valid_loss = 0
            train_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(data._train_eval_dataloader):
                    inputs, targets = inputs.to(device), targets.to(device)

                    if batch_idx == 0 and epoch_ == 0:
                        intitial_loss = criterion(net(inputs), targets)
                        print("Initial Loss ", intitial_loss, "batch ID: ", batch_idx)

                    #inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                acc = correct / total
                train_losses.append(train_loss / (batch_idx+1))
                valid_accuracies.append(acc)
                print("********************************")
                print("Evaluating after {0:d} of {1:d} epochs...".format(epoch_, num_epochs))
                print("TRAIN:" + "loss {0:g}, acc {1:f}".format((train_loss / (batch_idx+1)), acc))

                for batch_idx, (inputs, targets) in enumerate(data._valid_dataloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    valid_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                acc = correct / total
                valid_losses.append(valid_loss / (batch_idx+1))
                valid_accuracies.append(acc)
                print("VALID:" + "loss {0:g}, acc {1:f}".format((valid_loss / (batch_idx+1)), acc))

        def test(epoch_):
            global best_acc
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(data._test_dataloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                acc = correct / total
                test_accuracies.append(acc)
                test_losses.append(test_loss/(batch_idx+1))
                print("TEST:" + "loss {0:g}, acc {1:f}".format(test_loss/(batch_idx+1), acc))
                print("********************************")

        def train(epoch_):
            batch_size = 0
            tproblem.train_init_op()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(data._train_dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                opt.zero_grad()
                batch_size +=1
                # print("batch ID: ", batch_idx)
                # print(inputs.size())

                # if batch_idx == 0 and epoch_ == 0:
                #     intitial_loss = criterion(net(inputs), targets)
                #     print("Initial Loss ", intitial_loss, "batch ID: ", batch_idx)

                def loss_fn(backward=True):
                    out_ = net(inputs)
                    # print(batch_idx, inputs[0])
                    loss_ = criterion(out_, targets)
                    # print(out_[127], targets)
                    # if Vonwo != '':
                    #     print(Vonwo, "   ", batch_idx, "    ", loss_)
                    #     print(targets, out_.max(1))
                    if backward:
                        loss_.backward()
                    return loss_, out_, batch_idx

                loss, outputs = opt.step(loss_fn)

            #     train_loss += loss.item()
            #     _, predicted = outputs.max(1)
            #     total += targets.size(0)
            #     correct += predicted.eq(targets).sum().item()
            # acc = 100. * correct / total
            # train_losses.append(train_loss/batch_size)
            # train_accuracies.append(acc)

            # cur_time = int((time.time() - time_start))
            # logger.debug('train time: {:4.2f} min'.format(cur_time / 60))
            # logger.info(formatted_str('TRAIN:', epoch_, train_loss / (batch_idx + 1), correct / total))
            #
            # for s, t in [('time', cur_time), ('epoch', epoch_)]:
            #     writer.add_scalar('train-%s/accuracy' % s, correct / total, t)
            #     writer.add_scalar('train-%s/train_loss' % s, train_loss / (batch_idx + 1), t)

            minibatch_train_losses=[]
            if tb_log:
                summary_writer.close()
            # Put results into output dictionary.
        for epoch_count in range(num_epochs + 1):
            valid(epoch_count)
            test(epoch_count)
            train(epoch_count)

            if epoch_count == num_epochs:
                print("ENDE")
                break

        output = {
            "train_losses": train_losses,
            'valid_losses': valid_losses,
            "test_losses": test_losses,
            "minibatch_train_losses": minibatch_train_losses,
            "train_accuracies": train_accuracies,
            'valid_accuracies': valid_accuracies,
            "test_accuracies": test_accuracies
        }
        return output
