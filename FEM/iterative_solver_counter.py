import numpy as np
from matplotlib import pyplot as plt


class iterative_solver_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.callbacks = []
        self.internal_list = []

    def append(self, elem):
        self.internal_list.append(elem)

    def get_list(self):
        return self.internal_list

    def setup_plot_params(self, **kwargs):
        self.plot_dict = kwargs
        for key, value in kwargs.items():
            self.__setattr__(key, value)


    def plot(self, label=True, **kwargs):
        if label is False:
            plt.semilogy(np.linspace(1, len(self.internal_list), len(self.internal_list)), self.internal_list, **kwargs)
        else:
            plt.semilogy(np.linspace(1, len(self.internal_list), len(self.internal_list)), self.internal_list, **self.plot_dict)
            plt.legend()

        plt.xlabel('Iterations')
        plt.ylabel('Relative $L_2$ Residual')
    def __call__(self, rk=None):
        self.callbacks.append(rk)
        self.internal_list.append(rk)
        self.niter += 1