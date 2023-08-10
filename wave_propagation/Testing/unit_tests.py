# James Elgy - 01/08/2023

from .scipy_random_solve import *
from .test_p_ref import *
from .scipy_ngsolve_comparison import *
from ..FEM.Wave_Propagation import wave_propagation as wp
from ..Multigrid.ngsolve_example import *


def test_scipy_random_solve():
    """
    Checks that scipy sparse can solve a simple linear system independent of ngsolve.
    :return:
    """
    err = scipy_random_solve()
    assert err < 1e-10

def test_wave_prop():
    """
    Tests that the wave propagation can run without raising an execption
    :return:
    """
    w = wp()
    w.run(p=0)

def test_scipy_ngsolve_comparison():
    """
    Tests that ngsolve and scipy give the same answer
    :return:
    """
    err = scipy_ngsolve_comparison()
    print(err)
    assert err < 1e-6

def test_p_ref():
    """
    Tests that error is monotonically decreasing as p in increased.
    :return:
    """
    err, ndof = run_p_ref('scipy', preconditioner='multigrid', plot=False)
    assert all(earlier > later for earlier, later in zip(err, err[1:]))

def test_multigrid_local_preconditioner_comparison():
    """
    Compares local and multigrid preconditioners for H1 poisson problem using the example code from the NGSolve
    website. Number of iterations required for multigrid should all be less than the local preconditioner.
    """
    steps_loc = np.asarray(ngsolve_example(levels=9, precond='local'))
    steps_mult = np.asarray(ngsolve_example(levels=9, precond='multigrid'))

    assert all(m < l for m, l in zip(steps_mult[:,1], steps_loc[:,1]))

