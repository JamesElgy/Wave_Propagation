import numpy as np
from matplotlib import pyplot as plt
from ngsolve import BilinearForm, Mesh, H1, unit_square, grad, LinearForm, dx, GridFunction, Preconditioner, \
    TaskManager, CGSolver, unit_cube
import time


def ngsolve_example(h=0.5, p=1, levels=1,
                 condense=False,
                 precond="local",
                 wb_edges=True):
    """
    Solve Poisson problem on l refinement levels.
    PARAMETERS:
        h: coarse mesh size
        p: polynomial degree
        l: number of refinement levels
        condense: if true, perform static condensations
        precond: name of a built-in preconditioner
    Returns: (ndof, niterations)
        List of tuples of number of degrees of freedom and iterations
    """
    # mesh = Mesh(unit_square.GenerateMesh(maxh=h))
    mesh = Mesh(unit_cube.GenerateMesh(maxh=h))
    fes = H1(mesh, order=p, dirichlet="bottom|left|back", wb_withedges=wb_edges)

    u, v = fes.TnT()
    a = BilinearForm(grad(u)*grad(v)*dx)
    f = LinearForm(v*dx)
    gfu = GridFunction(fes)

    if precond != 'bddc_mult':
        c = Preconditioner(a, precond) # Register c to a BEFORE assembly
    else:
        c = Preconditioner(a, 'bddc', coarsetype="h1amg")


    steps = []

    for l in range(levels):
        l = 0
        print(l)
        if l > 0:
            mesh.Refine()
        fes.Update()
        gfu.Update()

        with TaskManager():
            a.Assemble()
            f.Assemble()

            # Conjugate gradient solver
            start_time = time.time_ns()
            inv = CGSolver(a.mat, c.mat, maxsteps=1000)
            gfu.vec.data = inv * f.vec
        steps.append ( (fes.ndof, inv.GetSteps(), times) )
    return np.asarray(steps)

if __name__ == '__main__':
    l = 1
    p = 5
    steps_bddc = ngsolve_example(precond='bddc', levels=l, p=p)
    steps_bddcmult = ngsolve_example(precond='bddc_mult', levels=l, p=p, wb_edges=False)
    steps_mult = ngsolve_example(precond='multigrid', levels=l, p=p)

    print(steps_bddcmult)
    print('')
    print(steps_bddc)
    print('')
    print(steps_mult)




