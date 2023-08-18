# James Elgy - 08/08/2023

import numpy as np
from matplotlib import pyplot as plt
import os

def save_all_figures(path, format='pdf', suffix='', prefix=''):
    """
    Function to save all open figures to disk.
    Files are named as:
    {suffix}{figure_n}{prefix}.{format}
    :param path: path to the desired saving directory.
    :param format: desired file format. pdf, png, jpg
    :param suffix: additional component of the output filename
    :param prefix: additional component of the output filename
    :return:
    """

    if not os.path.isdir(path):
        os.mkdir(path)
    extension = '.' + format
    if format != 'tex':
        for i in plt.get_fignums():
            plt.figure(i)
            filename = prefix + f'figure_{i}' + suffix
            plt.savefig(os.path.join(path, filename) + extension)
    else:
        raise TypeError('Unrecognised file format')



if __name__ == '__main__':
    pass
