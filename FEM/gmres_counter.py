import numpy as np
from matplotlib import pyplot as plt


class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.callbacks = []
        self.internal_list = []

    def append(self, elem):
        self.internal_list.append(elem)

    def get_list(self):
        return self.internal_list

    def plot(self, label=''):
        if label == '':
            plt.semilogy(np.linspace(1, len(self.internal_list), len(self.internal_list)), self.internal_list)
        else:
            plt.semilogy(np.linspace(1, len(self.internal_list), len(self.internal_list)), self.internal_list, label=label)
            plt.legend()

        plt.xlabel('Iterations')
        plt.ylabel('Absolute Residual')
    def __call__(self, rk=None):
        self.callbacks.append(rk)
        self.internal_list.append(rk)
        self.niter += 1