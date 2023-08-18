# James Elgy - 07/08/2023

from matplotlib import pyplot as plt
import numpy as np
from ngsolve import BilinearForm, Mesh, HCurl, LinearForm, GridFunction, Preconditioner, unit_cube, \
    TaskManager, CGSolver, SymbolicLFI, SymbolicBFI, InnerProduct, Integrate, curl, CoefficientFunction, sin, BND,\
    Projector
from ngsolve import y as ng_y
from netgen.occ import Box, Pnt, OCCGeometry

from scipy.sparse.linalg import LinearOperator, gmres, cg
from wave_propagation.FEM.iterative_solver_counter import *
from wave_propagation.Saving.save_figs import *


def Hcurl_example_eliptic(h=0.5, p=1, levels=1,
                 condense=True,
                 precond="local",
                 use_scipy=False,
                 plot_iter = False,
                 counter_figure = 1,
                 tol=1e-8):

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

    Solves curl(curl(A)) + A = F where A is [sin(y), 0, 0] and F = [2sin(y), 0, 0].

    """

    cube = unit_cube
    mesh = Mesh(cube.GenerateMesh(maxh=h))
    fes = HCurl(mesh, order=p, dirichlet="bottom|left|right|top|front|back")

    A_exact = CoefficientFunction((sin(ng_y), 0, 0))
    F_exact = CoefficientFunction((2 * sin(ng_y), 0, 0))

    u, v = fes.TnT()
    a = BilinearForm(fes, condense=condense)
    a += SymbolicBFI(curl(u) * curl(v))
    a += SymbolicBFI(u * v)

    f = LinearForm(fes)
    f += SymbolicLFI(F_exact * v)

    gfu = GridFunction(fes)

    if precond != 'bddc_mult':
        c = Preconditioner(a, precond) # Register c to a BEFORE assembly
    else:
        c = Preconditioner(a, 'bddc', coarsetype="multigrid") # Register c to a BEFORE assembly

    steps = []

    for l in range(levels):
        print(f'level = {l}')
        if l > 0:
            mesh.Refine()
        fes.Update()
        gfu.Update()

        gfu.Set(A_exact, BND)

        if use_scipy is False:
            with TaskManager():
                a.Assemble()
                f.Assemble()


                # Solve steps depend on condense
                if condense:
                    f.vec.data += a.harmonic_extension_trans * f.vec

                res = f.vec.CreateVector()
                res.data = f.vec - (a.mat * gfu.vec)

                # Conjugate gradient solver
                inv = CGSolver(a.mat, c.mat, maxsteps=10000, precision=tol)
                gfu.vec.data += inv * res

                if condense:
                    gfu.vec.data += a.harmonic_extension * gfu.vec
                    gfu.vec.data += a.inner_solve * f.vec
        else:
            a.Assemble()
            f.Assemble()
            proj = Projector(mask=fes.FreeDofs(coupling=condense), range=True)

            # Solve steps depend on condense
            if condense:
                f.vec.data += a.harmonic_extension_trans * f.vec

            res = f.vec.CreateVector()
            res.data = f.vec - (a.mat * gfu.vec)

            tmp1 = f.vec.CreateVector()
            tmp2 = f.vec.CreateVector()

            def matvec(v):
                tmp1.FV().NumPy()[:] = v
                tmp2.data = a.mat * tmp1
                tmp2.data = proj *tmp2
                return tmp2.FV().NumPy()

            res.data = proj * res
            A = LinearOperator((a.mat.height, a.mat.width), matvec)
            u = gfu.vec.CreateVector()
            counter = iterative_solver_counter()  # timing is done using ns precision, so counter is initialised immediately before solver.
            counter.set_LSE(A, res.FV().NumPy())

            u.FV().NumPy()[:], succ = cg(A, res.FV().NumPy(), tol=tol, maxiter=10000, M=c, callback=counter)
            gfu.vec.data += u
            if condense:
                gfu.vec.data += a.harmonic_extension * gfu.vec
                gfu.vec.data += a.inner_solve * f.vec

            if plot_iter == True:
                plt.figure(counter_figure)
                counter.setup_plot_params(label=f'Refinement Level = {l}, NDOF={fes.ndof}')
                counter.plot(label=True)
                plt.title(f'{precond}')
                plt.axhline(tol*np.linalg.norm(res.FV().NumPy()), color='r', linestyle='--')
                plt.figure(counter_figure+1)
                counter.plot_time(label=True)
                plt.title(f'{precond}')
                plt.axhline(tol*np.linalg.norm(res.FV().NumPy()), color='r', linestyle='--')


        err = Integrate(InnerProduct(A_exact - gfu, A_exact - gfu), mesh) ** 0.5
        err /= Integrate(InnerProduct(A_exact, A_exact),mesh) ** 0.5

        if use_scipy is False:
            steps.append((fes.ndof, inv.GetSteps(), err))
        else:
            steps.append((fes.ndof, counter.niter, err))

    del mesh, a, f, A_exact, F_exact, fes, u,  res, c, gfu
    return steps

if __name__ == '__main__':
    print('______')

    plt.figure(999)
    for h in [0.5]:
        p = 1

        print('')
        steps_mult = np.asarray(
            Hcurl_example_eliptic(h=h, levels=5, p=p, precond='local', use_scipy=True, plot_iter=True, counter_figure=7))
        plt.figure(999)
        plt.loglog(steps_mult[:, 0], steps_mult[:, 1], "-*", label=f'Multigrid h={h}')

        print('')
        steps_mult = np.asarray(Hcurl_example_eliptic(h=h, levels=5, p=p, precond='multigrid', use_scipy=True, plot_iter=True, counter_figure=1))
        plt.figure(999)
        plt.loglog(steps_mult[:, 0], steps_mult[:, 1], "-*", label=f'Multigrid h={h}')

        steps_mult = np.asarray(Hcurl_example_eliptic(h=h, levels=5, p=p, precond='bddc_mult', use_scipy=True, plot_iter=True, counter_figure=3))
        plt.figure(999)
        plt.loglog(steps_mult[:, 0], steps_mult[:, 1], "-*", label=f'BDDC-m h={h}')

        steps_mult = np.asarray(Hcurl_example_eliptic(h=h, levels=5, p=p, precond='bddc', use_scipy=True, plot_iter=True, counter_figure=5))
        plt.figure(999)
        plt.loglog(steps_mult[:, 0], steps_mult[:, 1], "-*", label=f'BDDC h={h}')

    plt.legend()
    plt.xlabel('$N_d$')
    plt.ylabel('$N_{iter}$')


    save_all_figures(f'Results', format='pdf', prefix=f'p={p}_')