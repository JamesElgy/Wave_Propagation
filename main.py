import wave_propagation as wp
from matplotlib import pyplot as plt
import numpy as np


def compute_eigs(f, fignum):
    # global eigs

    print(f'\033[96mBDDC: {f}\033[0m')
    W = wp.wave_propagation(h=2*np.pi/f)
    W.run(p=1, preconditioner='bddc', wavenumber=np.asarray([1, 0, 0]), box_size=2, h=2*np.pi/f, solver_residual_plot=False)
    eigs = W.compute_eigenspectum(n=30)
    # eigs = np.append(eigs, W.compute_eigenspectum(n=30, which='SM'))
    print(eigs)
    e = np.asarray([complex(n) for n in eigs])
    plt.figure(fignum)
    plt.scatter(e.real, e.imag, label='BDDC', s=2, marker='x')
    plt.xlabel('$\mathrm{Re}(\lambda)$')
    plt.ylabel('$\mathrm{Im}(\lambda)$')

    print('\033[96mLocal\033[0m')
    W = wp.wave_propagation(h=2*np.pi/f)
    W.run(p=1, preconditioner='local', wavenumber=np.asarray([1, 0, 0]), box_size=2, h=2*np.pi/f,
          solver_residual_plot=False)
    eigs = W.compute_eigenspectum(n=30)
    # eigs = np.append(eigs, W.compute_eigenspectum(n=30, which='SM'))
    print(eigs)
    e = np.asarray([complex(n) for n in eigs])
    plt.figure(fignum)
    plt.scatter(e.real, e.imag, label='Local', s=2, marker='+')
    plt.xlabel('$\mathrm{Re}(\lambda)$')
    plt.ylabel('$\mathrm{Im}(\lambda)$')

    print('\033[96mMultigrid\033[0m')
    W = wp.wave_propagation(h=2*np.pi/f)
    W.run(p=1, preconditioner='multigrid', wavenumber=np.asarray([1, 0, 0]), box_size=2, h=2*np.pi/f,
          solver_residual_plot=False)
    eigs = W.compute_eigenspectum(n=30)
    # eigs = np.append(eigs, W.compute_eigenspectum(n=30, which='SM'))
    print(eigs)
    e = np.asarray([complex(n) for n in eigs])
    plt.figure(fignum)
    plt.scatter(e.real, e.imag, label='Multigrid', s=2, marker='^')
    plt.xlabel('$\mathrm{Re}(\lambda)$')
    plt.ylabel('$\mathrm{Im}(\lambda)$')
    plt.title(f'$h = \lambda / {f}$')
    plt.legend()
    plt.show()



def compute_eigs_once(f, fignum, meth='bddc'):
    W = wp.wave_propagation(h=2 * np.pi / f)
    W.run(p=2, preconditioner=meth, wavenumber=np.asarray([1, 0, 0]), box_size=2, h=2 * np.pi / f,
          solver_residual_plot=False)
    eigs = W.compute_eigenspectum(n=100, which='LM')
    # eigs = np.append(eigs, W.compute_eigenspectum(n=100, which='SM'))
    # eigs = np.append(eigs, W.compute_eigenspectum(n=30, which='LR'))
    # print(eigs)
    e = np.asarray([complex(n) for n in eigs])
    plt.figure(fignum)
    plt.scatter(e.real, e.imag, label=f'$h=2\pi / {f}$', s=2, marker='x')
    plt.xlabel('$\mathrm{Re}(\lambda)$')
    plt.ylabel('$\mathrm{Im}(\lambda)$')
    plt.legend()
    plt.title(meth.capitalize())


if __name__ == '__main__':

    W = wp.wave_propagation()
    W.run()
    eigs = W.compute_eigenspectum(use_NGSolve=True)
    print(eigs)
    # wp.Testing.run_p_ref(solver='CG', preconditioner='direct', tol=1e-12)
    # wp.Testing.run_p_ref(solver='GMRES', preconditioner='direct', tol=1e-12)
    # wp.Testing.run_p_ref(solver='scipy', preconditioner='multigrid', tol=1e-12)
    # wp.Testing.run_p_ref(solver='scipy', preconditioner='multigrid', tol=1e-12, use_PML=True)

    #wp.Testing.run_p_ref(solver='scipy', preconditioner='multigrid', tol=1e-12)
    # wp.Testing.run_p_ref(solver='scipy', preconditioner='bddc', tol=1e-12)
    # wp.Testing.run_p_ref(solver='scipy', preconditioner='multigrid', tol=1e-12)
    # for f in [8, 16, 32, 64]:
    #     compute_eigs_once(f, 1, meth='local')
    #     plt.show()
    #     compute_eigs_once(f, 2, meth='bddc')
    #     plt.show()
        # compute_eigs_once(f, 3, meth='multigrid')
