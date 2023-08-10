from ngsolve import *

def run_example(precond='bddc_mult', p=2):
    cube = unit_cube
    nmesh = cube.GenerateMesh(maxh=0.5)
    nmesh.BoundaryLayer(boundary=".*", thickness=[0.01], domains='default', material='default')
    mesh = Mesh(nmesh)
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

        # Computing error:
        err = Integrate(InnerProduct(A_exact - gfu, A_exact - gfu), mesh) ** 0.5
        err /= Integrate(InnerProduct(A_exact, A_exact),mesh) ** 0.5

        print(l, fes.ndof, inv.GetSteps(), err)

if __name__ == '__main__':
    print('\nBDDC Mult')
    run_example(precond='bddc_mult')

    print('\nBDDC')
    run_example(precond='bddc')

    print('\nMultigrid')
    run_example(precond='multigrid')

    print('\nComparing for p=1')
    print('\n')
    run_example(precond='bddc_mult', p=1)
    print('')
    run_example(precond='bddc', p=1)
    print('')
    run_example(precond='multigrid', p=1)