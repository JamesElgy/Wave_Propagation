from ngsolve import *
from matplotlib import pyplot as plt
import numpy as np

from scipy.sparse.linalg import LinearOperator, gmres, cg
from wave_propagation.FEM.iterative_solver_counter import *

def run_example(precond='bddc_mult', p=2, use_scipy=False):
    cube = unit_cube
    mesh = Mesh(cube.GenerateMesh(maxh=0.5))
    fes = HCurl(mesh, order=p, dirichlet="bottom|left|right|top|front|back")

    # Setting known exact solution
    A_exact = CoefficientFunction((sin(y), 0, 0))
    F_exact = CoefficientFunction((2 * sin(y), 0, 0))

    u, v = fes.TnT()
    a = BilinearForm(fes, condense=True)
    a += SymbolicBFI(curl(u) * curl(v))
    a += SymbolicBFI(u * v)

    f = LinearForm(fes)
    f += SymbolicLFI(F_exact * v)

    gfu = GridFunction(fes)

    # Setting preconditioner
    if precond == 'bddc_mult':
        c = Preconditioner(a, 'bddc', coarsetype="multigrid")
    else:
        c = Preconditioner(a, precond)

    # Generating refined mesh and updating fes
    for l in range(5):
        if l > 0:
            mesh.Refine()
        fes.Update()
        gfu.Update()

        gfu.Set(A_exact, BND)

        # Solving
        if use_scipy is False:
            with TaskManager():
                a.Assemble()
                f.Assemble()

                f.vec.data += a.harmonic_extension_trans * f.vec
                res = f.vec.CreateVector()
                res.data = f.vec - (a.mat * gfu.vec)
                # Conjugate gradient solver
                inv = CGSolver(a.mat, c.mat, maxsteps=10000, precision=1e-7)
                gfu.vec.data += inv * res
                gfu.vec.data += a.harmonic_extension * gfu.vec
                gfu.vec.data += a.inner_solve * f.vec
        else:
            a.Assemble()
            f.Assemble()
            proj = Projector(mask=fes.FreeDofs(coupling=True), range=True)

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

            u.FV().NumPy()[:], succ = cg(A, res.FV().NumPy(), tol=1e-8, maxiter=10000, M=c, callback=counter)
            gfu.vec.data += u
            gfu.vec.data += a.harmonic_extension * gfu.vec
            gfu.vec.data += a.inner_solve * f.vec

            plt.figure(999)
            counter.setup_plot_params(label=f'Refinement Level = {l}, NDOF={fes.ndof}')
            counter.plot(label=True)
            plt.title(f'{precond}')
            plt.axhline(1e-8*np.linalg.norm(res.FV().NumPy()), color='r', linestyle='--')

        # Computing error:
        err = Integrate(InnerProduct(A_exact - gfu, A_exact - gfu), mesh) ** 0.5
        err /= Integrate(InnerProduct(A_exact, A_exact),mesh) ** 0.5

        # print(l, fes.ndof, inv.GetSteps(), err)

if __name__ == '__main__':
    print('\nBDDC Mult')
    run_example(precond='bddc_mult', use_scipy=True)
    plt.show()

    # print('\nBDDC')
    # run_example(precond='bddc')
    #
    # print('\nMultigrid')
    # run_example(precond='multigrid')
    #
    # print('\nComparing for p=0')
    # print('\nBDDC Mult')
    # run_example(precond='bddc_mult', p=0)
    #
    # print('\nBDDC')
    # run_example(precond='bddc', p=0)
    #
    # print('\nMultigrid')
    # run_example(precond='multigrid', p=0)