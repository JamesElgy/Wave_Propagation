import numpy as np
from matplotlib import pyplot as plt
import time

class iterative_solver_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.callbacks = []
        self.internal_list = []
        self.time_list = []
        self.start_time = time.time_ns()

        self.weighting = 1



    def append(self, elem):
        self.internal_list.append(elem)

    def get_list(self):
        return self.internal_list

    def set_LSE(self, linop, b):
        self.linop = linop
        self.b = b
        self.weighting = np.linalg.norm(self.b)


    def setup_plot_params(self, **kwargs):
        self.plot_dict = kwargs

        try:
            self.xlim = self.plot_dict['xlim']
            del self.plot_dict['xlim']
        except:
            self.xlim = ''
        try:
            self.ylim = self.plot_dict['ylim']
            del self.plot_dict['ylim']
        except:
            self.ylim = ''


        for key, value in kwargs.items():
            self.__setattr__(key, value)


    def plot(self, label=True, **kwargs):
        if label is False:
            plt.semilogy(np.linspace(1, len(self.internal_list), len(self.internal_list)), self.internal_list, **kwargs)
        else:
            plt.semilogy(np.linspace(1, len(self.internal_list), len(self.internal_list)), self.internal_list, **self.plot_dict)
            plt.legend()

        if self.xlim != '':
            plt.xlim(self.xlim)
        if self.ylim != '':
            plt.xlim(self.ylim)

        plt.xlabel('Iterations')
        plt.ylabel('$L_2$ Residual')

    def plot_time(self, label=True, **kwargs):
        times = (np.asarray(self.time_list) - self.time_list[0]) / 1e9 # time_list is in nanoseconds
        if label is False:
            plt.semilogy(times, self.internal_list, **kwargs)
        else:
            plt.semilogy(times, self.internal_list, **self.plot_dict)
            plt.legend()

        plt.xlabel('Time [sec]')
        plt.ylabel('Relative $L_2$ Residual')

    def __call__(self, rk=None):

        if len(rk > 1):
            # solution vector rather than residual
            rk = np.linalg.norm((self.linop(rk)) - self.b)

        self.callbacks.append(rk)
        self.internal_list.append(rk)
        self.niter += 1
        self.time_list.append(time.time_ns())

        # if self.niter % 100 == 0:
        #     print(f'{self.niter}: {rk}')