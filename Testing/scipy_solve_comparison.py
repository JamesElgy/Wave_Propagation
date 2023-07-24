from ngsolve import *
from netgen.geom2d import unit_square
import scipy as sp
import inspect

def report(xk):
    frame = inspect.currentframe().f_back
    print(frame.f_locals['resid'])

def static_condensation_test():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.4))
    fes = H1(mesh, order=4, dirichlet='bottom|right')
    u, v = fes.TnT()

    a = BilinearForm(fes, condense=True)
    a += grad(u) * grad(v) * dx

    c = Preconditioner(a, type="direct")

    a.Assemble()

    U = x*x*(1-y)*(1-y)          # U = manufactured solution
    DeltaU = 2*((1-y)*(1-y)+x*x) # Source: DeltaU = ∆U

    f = LinearForm(fes)
    f += -DeltaU * v * dx
    f.Assemble()


    # Direct inverse:
    u = GridFunction(fes)
    u.Set(U, BND)               # Dirichlet b.c: u = U on boundary

    r = f.vec.CreateVector()
    r.data = f.vec - a.mat * u.vec
    r.data += a.harmonic_extension_trans * r

    u.vec.data += a.mat.Inverse(fes.FreeDofs(True)) * r
    u.vec.data += a.harmonic_extension * u.vec
    u.vec.data += a.inner_solve * r

    err = sqrt(Integrate((U-u)*(U-u),mesh))  # Compute error
    print(f'Direct: {err}')

    # CGSolver
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.4))
    fes = H1(mesh, order=4, dirichlet='bottom|right')
    u, v = fes.TnT()

    a = BilinearForm(fes, condense=True)
    a += grad(u) * grad(v) * dx

    c = Preconditioner(a, type="direct")

    a.Assemble()

    U = x*x*(1-y)*(1-y)          # U = manufactured solution
    DeltaU = 2*((1-y)*(1-y)+x*x) # Source: DeltaU = ∆U

    f = LinearForm(fes)
    f += -DeltaU * v * dx
    f.Assemble()

    u = GridFunction(fes)
    u.Set(U, BND)               # Dirichlet b.c: u = U on boundary

    r = f.vec.CreateVector()
    r.data = f.vec - a.mat * u.vec
    r.data += a.harmonic_extension_trans * r

    u.vec.data += CGSolver(mat=a.mat, pre=c.mat, precision=1e-16, maxsteps=5000,) * r
    u.vec.data += a.harmonic_extension * u.vec
    u.vec.data += a.inner_solve * r

    err = sqrt(Integrate((U-u)*(U-u),mesh))  # Compute error
    print(f'CG: {err}')

    # Scipy
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.4))
    fes = H1(mesh, order=4, dirichlet='bottom|right')
    u, v = fes.TnT()

    a = BilinearForm(fes, condense=True)
    a += grad(u) * grad(v) * dx

    c = Preconditioner(a, type="direct")
    a.Assemble()

    U = x*x*(1-y)*(1-y)           # U = manufactured solution
    DeltaU = 2*((1-y)*(1-y)+x*x)  # Source: DeltaU = ∆U

    f = LinearForm(fes)
    f += -DeltaU * v * dx
    f.Assemble()

    u = GridFunction(fes)
    u.Set(U, BND)               # Dirichlet b.c: u = U on boundary

    r = f.vec.CreateVector()
    r.data = f.vec - a.mat * u.vec
    r.data += a.harmonic_extension_trans * r

    # Converting to scipy sparse:

    tmp1 = f.vec.CreateVector()
    tmp2 = f.vec.CreateVector()
    def matvec(v):
        tmp1.FV().NumPy()[:] = v
        tmp2.data = a.mat * tmp1
        return tmp2.FV().NumPy()

    q = u.vec.CreateVector()
    A = sp.sparse.linalg.LinearOperator((a.mat.height,a.mat.width), matvec)
    q.FV().NumPy()[:], succ = sp.sparse.linalg.cg(A, r.FV().NumPy(), maxiter=5000, tol=1e-10)#, callback=report)
    print(succ)


    u.vec.data += q
    u.vec.data += a.harmonic_extension * u.vec
    u.vec.data += a.inner_solve * r

    err = sqrt(Integrate((U-u)*(U-u),mesh))  # Compute error
    print(f'Scipy CG: {err}')

def poisson(meth='direct'):
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))
    mesh.nv, mesh.ne  # number of vertices & elements

    fes = H1(mesh, order=4, dirichlet="bottom|right")
    u = fes.TrialFunction()  # symbolic object
    v = fes.TestFunction()  # symbolic object
    gfu = GridFunction(fes)  # solution

    U = x * x * (1 - y) * (1 - y)  # U = manufactured solution
    DeltaU = 2 * ((1 - y) * (1 - y) + x * x)  # Source: DeltaU = ∆U
    gfu.Set(U, BND)

    a = BilinearForm(fes)
    a += grad(u) * grad(v) * dx

    if meth == 'CG' or 'scipy':
        pre = Projector(mask=fes.FreeDofs(), range=True)
        c = Preconditioner(a, 'direct')
    a.Assemble()

    f = LinearForm(fes)
    f += -DeltaU * v * dx
    f.Assemble()

    r = f.vec.CreateVector()
    r.data = f.vec - a.mat * gfu.vec

    if meth == 'direct':
        inv = a.mat.Inverse(freedofs=fes.FreeDofs())
        gfu.vec.data += inv * r
    elif meth == 'CG':
        inv = CGSolver(a.mat, c.mat, maxsteps=1000, precision=1e-10)
        gfu.vec.data += inv * r
    elif meth == 'scipy':

        # rows, cols, vals = a.mat.COO()
        # A = sp.sparse.csr_matrix((vals, (rows, cols)))
        # A = sp.sparse.linalg.aslinearoperator(A)
        #
        pre = Projector(mask=fes.FreeDofs(), range=True)
        tmp1 = r.CreateVector()
        tmp2 = r.CreateVector()

        def matvec(v):
            tmp1.FV().NumPy()[:] = v
            tmp2.data = a.mat * tmp1
            tmp2.data = pre * tmp2
            return tmp2.FV().NumPy()



        q = r.CreateVector()
        r.data = pre * r
        A = sp.sparse.linalg.LinearOperator((a.mat.height, a.mat.width), matvec)
        q.FV().NumPy()[:], s = sp.sparse.linalg.cg(A, r.FV().NumPy(), tol=1e-15, maxiter=5000, callback=report)
        print(s)
        gfu.vec.data += q

    err = sqrt(Integrate((U-gfu)*(U-gfu),mesh))  # Compute error
    print(f'{meth}: {err}')



if __name__ == '__main__':


    poisson('direct')
    poisson('CG')
    poisson('scipy')
