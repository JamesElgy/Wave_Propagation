from ngsolve import *
import matplotlib.pyplot as plt

def SolveProblem(h=0.5, p=1, levels=1,
                 condense=True,
                 precond="local"):
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
    # fes = H1(mesh, order=p, dirichlet="bottom|left")
    fes = HCurl(mesh, order=p, dirichlet="bottom|left|right|top", complex=True)

    u, v = fes.TnT()
    # a = BilinearForm(grad(u)*grad(v)*dx, condense=condense)
    a = BilinearForm(fes, condense=condense)
    a += SymbolicBFI(curl(u) * curl(v))
    a += SymbolicBFI(-1 * u * v)

    # f = LinearForm(0*dx)
    f = LinearForm(fes)
    f += SymbolicLFI(0)

    ey = exp(1j * (1 * x))
    exact = CoefficientFunction((0, ey, 0))

    gfu = GridFunction(fes)
    gfu.Set(exact, BND)
    Draw(gfu)
    c = Preconditioner(a, precond) # Register c to a BEFORE assembly

    steps = []

    for l in range(levels):
        if l > 0:
            mesh.Refine()
        fes.Update()
        gfu.Update()

        with TaskManager():
            a.Assemble()
            f.Assemble()

            # Solve steps depend on condense
            if condense:
                f.vec.data += a.harmonic_extension_trans * f.vec

            res = f.vec.CreateVector()
            res.data = f.vec - (a.mat * gfu.vec)

            # Conjugate gradient solver
            inv = CGSolver(a.mat, c.mat, maxsteps=1000)
            gfu.vec.data += inv * res

            if condense:
                gfu.vec.data += a.harmonic_extension * gfu.vec
                gfu.vec.data += a.inner_solve * f.vec
        if fes.ndof < 15000:
            Redraw()

        # Computing error:
        err = Integrate(InnerProduct(exact - gfu, exact - gfu),mesh) ** 0.5 / Integrate(InnerProduct(exact, exact),mesh) ** 0.5
        steps.append ( (fes.ndof, inv.GetSteps(), err) )

        gfu = GridFunction(fes)
        gfu.Set(exact, BND) # Resetting GFU

    return steps


if __name__ == '__main__':
    res_mg = SolveProblem(levels=5, precond="bddc", p=2)

    for n in res_mg:
        print(n)