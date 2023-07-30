# James Elgy - 21/07/2023

import numpy as np
from matplotlib import pyplot as plt
# from ngsolve import VTKOutput

def export_to_vtk(wave_prop, savename, refine=True):

    # exact_1_Real = GridFunction(wave_prop.fes)
    # exact_2_Real = GridFunction(wave_prop.fes)
    # exact_3_Real = GridFunction(wave_prop.fes)
    # exact_1_Real.vec.FV().NumPy()[:] = wave_prop.e_exact[0]
    # exact_2_Real.vec.FV().NumPy()[:] = wave_prop.e_exact[1]
    # exact_3_Real.vec.FV().NumPy()[:] = wave_prop.e_exact[2]

    sol_real = wave_prop.sol.real
    sol_imag = wave_prop.sol.imag

    exact_real = wave_prop.e_exact.real
    exact_imag = wave_prop.e_exact.imag

    # sol_2 = wave_prop.sol[1]
    # sol_3 = wave_prop.sol[2]

    output = []
    output.append(sol_real)
    output.append(sol_imag)
    output.append(exact_real)
    output.append(exact_imag)
    # output.append(sol_2)
    # output.append(sol_3)

    if refine == True:
        subs = 3
    else:
        subs = 0

    vtk = VTKOutput(ma=wave_prop.mesh, coefs=output,
                    names=['sol_real', 'sol_imag', 'exact_real', 'exact_imag'], filename=savename, subdivision=subs)
    vtk.Do()



if __name__ == '__main__':
    pass
