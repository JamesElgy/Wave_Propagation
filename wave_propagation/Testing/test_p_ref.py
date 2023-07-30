# James Elgy - 19/07/2023

import numpy as np
from matplotlib import pyplot as plt
from ..FEM.Wave_Propagation import wave_propagation
from ngsolve import GridFunction, HCurl, Integrate, InnerProduct

def run_p_ref(solver='scipy', preconditioner='direct', tol=1e-10):
    plt.figure(1)
    err_array = []
    ndof_array = []

    for inst, kx in enumerate([1]):
        for p in [1,2,3]:

            print(f'Running for order {p}')

            wave_prop = wave_propagation(instance=inst)

            sol = wave_prop.run(p=p, wavenumber=np.asarray([kx, 0, 0]), box_size=2, h=0.08, solver=solver, preconditioner=preconditioner, tol=tol, solver_residual_plot=False)
            nd = wave_prop.fes.ndof
            exact = wave_prop.e_exact

            fes = HCurl(wave_prop.mesh, order=wave_prop.p, dirichlet="default", complex=True)
            exact_grid = GridFunction(fes)
            exact_grid.Set((exact[0], exact[1], exact[2]))

            domain_values = {'sphere': 1, 'pmlregion': 0}
            cf = wave_prop.mesh.MaterialCF(domain_values)

            Integration_Order = np.max([4 * (0 + 1), 3 * (5 - 1)])
            err = Integrate(cf *InnerProduct(wave_prop.sol - wave_prop.e_exact, wave_prop.sol - wave_prop.e_exact), wave_prop.mesh, order=Integration_Order)**0.5
            err = err / Integrate(cf * InnerProduct(wave_prop.e_exact, wave_prop.e_exact), wave_prop.mesh, order=Integration_Order)**0.5
            print(err)

            err_array += [err]
            ndof_array += [nd]

        plt.figure(kx)
        plt.loglog(ndof_array,err_array, marker='x', label=f'{solver}, {preconditioner}')
        plt.ylabel('relative error, $e$')
        plt.xlabel('$N_d$')
        plt.legend()


if __name__ == '__main__':
    # run_p_ref(solver='scipy', preconditioner='bddc', tol=1e-12)
    # run_p_ref(solver='scipy', preconditioner='local', tol=1e-12)
    run_p_ref(solver='scipy', preconditioner='multigrid', tol=1e-12)
