import numpy as np
from enum import Enum
import sys, os
import matplotlib.pyplot as plt
sys.path.append(os.pardir)
import pylab


class condition_error(Exception):
    pass


class Confidit:
    def __init__(self, x, y, test_x=None, test_y=None, eta = 100.0, test_interval:int=1000):
        """
        :param x: feature patterns for train
        :param y: the correct labels for x
        :param test_x: feature patterns for test
        :param test_y: the correct labels for y
        :param gamma: parameter for exploration
        :param test_interval: the intervals conducting test
        """

        self.x = x
        self.y = y
        self.test_x = test_x
        self.test_y = test_y
        self.data_size = x.shape[0]
        self.n = x.shape[1]

        self.t = 0

        self.eta = eta
        self.alpha = 1.0
        self.interval = test_interval

        self.K = np.max(y) + 1
        self.w = np.zeros((self.K, self.n))
        self.A = np.ones((self.K, self.n))*((1+self.alpha)**2)  # Matrix A (only diagonal element)

    def train(self, t):
        """
        :param t: the number of training.
        """
        seq = np.arange(self.data_size)
        np.random.shuffle(seq)

        ol_ratio_list = []
        ol, cl = 0, 0   # the number of ordinary, complementary labels

        accuracy_ratio_list = []
        correct, false = 0, 0

        sa = 0

        for count, i in enumerate(seq):
            x, y = self.x[i], self.y[i]
            x = x / np.linalg.norm(x)  # the norm of x is 1.

            wx = self._det_fun(self._fun(x))
            predict = self._det_label(wx)

            eta = self.eta
            proposed_label = self._det_proposed_label(x, wx, eta)
            self._update(x, proposed_label, (proposed_label==y))

            if predict == y:
                correct += 1
            else:
                false += 1

            if proposed_label == y:
                ol += 1
            else:
                cl += 1

            sa += (predict == proposed_label)

            if count % self.interval == 0:
                print(count)
                print("ordinary labels ratio", ol / (ol + cl))
                print("accuracy ratio", correct / (correct + false))
                print(sa)
                print('')
                ol_ratio_list.append(ol / (ol + cl))
                accuracy_ratio_list.append(correct / (correct + false))

        final_l = ol / (ol + cl)
        final_ac = correct / (correct + false)
        return ol_ratio_list, accuracy_ratio_list, final_l, final_ac

    def _update(self, x, proposed_label, ordinary:bool=True):
        """
        :param x: feature vector
        :param proposed_label: proposed_label
        :param ordinary: whether the proposed_label is true label or not
        """

        tau = 1.0 if ordinary else -1.0
        fun_x = self._fun(x)
        self.w[proposed_label] = self.w[proposed_label]*self.A[proposed_label] + tau * fun_x
        self.A[proposed_label] += np.square(fun_x)

        self.w[proposed_label] *= np.reciprocal(self.A[proposed_label])

    def _det_proposed_label(self, x, wx, eta):
        """
        determine proposed label by sampling
        :param x: the feature vector
        :param wx: the output of the one-ver-all classifiers
        :param eta: the value of exploration
        :return: proposed label
        """

        A_inverse = np.reciprocal(self.A)
        e_square = eta * (A_inverse*x).dot(x)
        e = np.sqrt(e_square)
        return np.argmax(wx + e)

    def _fun(self, x):
        """
        :param x: the feature pattern
        :return: the feature vector
        """
        return x

    def _det_fun(self, x):
        """
        :param x: the feature vector
        :return: the inner product of w and x
        """
        return x.dot(self.w.T)

    def _det_label(self, wx):
        """
        determine label by the inner product wx
        :param wx:
        :return: the determined label
        """
        if wx.ndim == 1:
            return np.argmax(wx)
        elif wx.ndim == 2:
            return np.argmax(wx, axis=1)
        else:
            raise condition_error
