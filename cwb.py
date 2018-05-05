import numpy as np
from enum import Enum
import sys, os
import matplotlib.pyplot as plt
sys.path.append(os.pardir)
import pylab
from scipy.stats import norm


class condition_error(Exception):
    pass


class CWB:
    def __init__(self, x, y, test_x=None, test_y=None, gamma = 0.2, eta = 0.7, test_interval:int=1000):
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

        self.gamma = gamma
        self.eta = eta
        self.alpha = 1.0
        self.interval = test_interval
        self.a = 1.0

        self.K = np.max(y) + 1
        self.w = np.zeros((self.K, self.n))
        self.S = np.identity(self.n)*self.a

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

            gamma = self.gamma
            proposed_label, p = self._det_proposed_label(predict, gamma)
            self._update(x, predict, proposed_label, p, (proposed_label==y))

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

        return ol_ratio_list, accuracy_ratio_list

    def _update(self, x, predict, proposed_label, p, ordinary:bool=True):
        """
        :param x: feature vector
        :param proposed_label: proposed_label
        :param ordinary: whether the proposed_label is true label or not
        """

        wx = self._det_fun(self._fun(x))
        m = wx[predict] - wx[proposed_label]
        v = x.dot(self.S).dot(x)
        l = max(0, self.eta*np.sqrt(v) - m)
        if ordinary and l > 0.0:
            phi = norm.ppf(q=self.eta)
            v = x.dot(self.S).dot(x)
            alpha = max(0, ((-(1+2*phi*m)+np.sqrt((1+2*phi*m)**2 - 2*phi*(m - phi*v)))/(4*phi*v)))
            beta = 2*alpha*phi
            tau

        fun_x = self._fun(x)
        self.w[proposed_label] = self.w[proposed_label]*self.A[proposed_label] + tau * fun_x
        self.A[proposed_label] += np.square(fun_x)

        self.w[proposed_label] *= np.reciprocal(self.A[proposed_label])

    def _det_proposed_label(self, predict, gamma):
        """
        determine proposed label by sampling
        :param predict: predicted label
        :param gamma: the value of exploration (assumed 0 <= gamma <= 0.5)
        :return: proposed label and p
        """
        assert gamma >= 0.0

        p = np.ones(self.K) * gamma / self.K
        p[predict] += (1.0-gamma)

        cumulative_p = np.copy(p)
        for i in range(1, self.K):
            cumulative_p[i] = cumulative_p[i-1] + p[i]

        assert abs(cumulative_p[self.K-1] - 1.0) <= 0.01
        cumulative_p[self.K-1] = 1.0    # for computation

        r = np.random.random()
        k = 0
        while cumulative_p[k] <= r:
            k += 1

        assert 0 <= k < self.K
        return k, p

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
