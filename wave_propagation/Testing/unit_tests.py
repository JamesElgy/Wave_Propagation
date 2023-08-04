# James Elgy - 01/08/2023

from .scipy_random_solve import *
from .test_p_ref import *
from .scipy_ngsolve_comparison import *
from ..FEM.Wave_Propagation import wave_propagation as wp

def test_scipy_random_solve():
    err = scipy_random_solve()
    assert err < 1e-10

def test_wave_prop():
    w = wp()
    w.run(p=0)

def test_scipy_ngsolve_comparison():
    err = scipy_ngsolve_comparison()
    print(err)
    assert err < 1e-6

def test_p_ref():
    err, ndof = run_p_ref('scipy', preconditioner='multigrid', plot=False)
    assert all(earlier > later for earlier, later in zip(err, err[1:]))




