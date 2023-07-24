# James Elgy - 21/07/2023

import numpy as np
from matplotlib import pyplot as plt

from FEM.Wave_Propagation import wave_propagation
from Saving.export_to_vtk import export_to_vtk

from ngsolve import *

def test_vtk():
    W = wave_propagation()
    W.run(p=3, wavenumber=np.asarray([-2, -0.5, 2.5]), box_size=5, h=0.5)

    export_to_vtk(W, '../test')



if __name__ == '__main__':
    test_vtk()
