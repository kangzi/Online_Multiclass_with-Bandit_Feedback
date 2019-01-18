import numpy as np
from enum import Enum
import sys, os
sys.path.append(os.pardir)
from confidit import Confidit
import warnings
warnings.simplefilter("error", RuntimeWarning)

class kernel(Enum):
    gauss = 0
    polynomial = 1


class condition_error(Exception):
    pass


class Confidit_kernel(Confidit):
    def __init__(self, x, y, g = 1.0, B = 500, eta = 100.0,
                 test_interval:int=1000, normalize:bool = True):
        """
        :param x:
        :param y:
        :param g:
        :param eta:
        :param test_interval:
        """
        super().__init__(x, y, eta, test_interval)
        self.g = g
        self.dim = B
        self.w = np.zeros((self.K, B))
        self.B = B
        self.bags = np.zeros((B, x[0].shape[0]))
        self.kernel = kernel.gauss
        self.normalize = normalize
        self.A = np.ones((self.K, self.B)) * ((1 + self.alpha) ** 2)

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
            if count == 0:
                self.bags = x.reshape(1, -1)
                self.w = np.zeros((self.K, 1))
                self.A = np.ones((self.K, 1)) * (1 + self.alpha)
            elif count < self.B:
                self.bags = np.vstack((self.bags, x.reshape(1, -1)))
                self.w = np.hstack((self.w, np.zeros((self.K, 1))))
                self.A = np.hstack((self.A, np.ones((self.K, 1)) * (1 + self.alpha)))

            try:
                _ = self._fun(x)
            except RuntimeWarning:
                false += 1
                if count % self.interval == 0:
                    print(count)
                    print("ordinary labels ratio", ol / (ol + cl))
                    print("accuracy ratio", correct / (correct + false))
                    print('')
                    ol_ratio_list.append(ol / (ol + cl))
                    accuracy_ratio_list.append(correct / (correct + false))
                continue

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
                print('')
                ol_ratio_list.append(ol / (ol + cl))
                accuracy_ratio_list.append(correct / (correct + false))

        final_l = ol / (ol + cl)
        final_ac = correct / (correct + false)
        return ol_ratio_list, accuracy_ratio_list, final_l, final_ac

    def _fun(self, x):
        if self.kernel == kernel.gauss:
                x = (np.exp(-np.sum(np.square(self.bags - x), axis=1) / self.g))
                if self.normalize:
                    return x / np.linalg.norm(x)
                else:
                    return x
        else:
            raise condition_error