# James Elgy - 19/07/2023

from netgen.occ import *
import netgen.meshing as ngmeshing
import inspect

from .iterative_solver_counter import *
from ..Testing.scipy_random_solve import *

# from ngsolve import *
import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
from ngsolve import exp as ng_exp
from ngsolve import x as ng_x
from ngsolve import y as ng_y
from ngsolve import z as ng_z
from ngsolve import BilinearForm, LinearForm, GridFunction, CoefficientFunction, CGSolver, GMRESSolver, Projector,\
    Preconditioner, HCurl, curl, div, Conj, SymbolicBFI, SymbolicLFI, BND, dx, IdentityMatrix, Mesh
from ngsolve.la import EigenValues_Preconditioner

class wave_propagation:

    def __init__(self, **kwargs):
        # Useful internal params:
        self.instance = 0

        # propagation parameters
        kx = 1
        ky = 0
        kz = 0
        self.wavenumber = np.asarray([kx, ky, kz])
        self.amp = np.asarray([0, 1, 0])

        # FEM parameters
        self.h = 0.08 #0.08
        self.p = 1
        self.preconditioner = 'bddc'  # 'local', 'direct', 'multigrid', 'bddc'
        self.solver = 'scipy'  # 'CG' or 'GMRES'
        self.tol = 1e-10
        self.max_iter = 2500
        self.refinement_levels = 0


        # Solver residual plotting
        self.solver_residual_plot = False
        self.solver_residual_plot_fignum = 2
        wavelength = 2*np.pi / np.linalg.norm(self.wavenumber)
        self.solver_residual_plot_label = f'{self.solver}'

        # Geometry parameters
        self.box_size = 1
        self.use_PML = False
        self.PML_thickness = 0.1
        self.PML_decay_rate = 10 * 1j

        # Setting any attributes specified when creating the class:
        for key, value in kwargs.items():
            self.__setattr__(key, value)


        scipy_random_solve()

    def generate_mesh(self):

        print(f'Using h={self.h}')

        half_length = self.box_size / 2
        # box = Box(Pnt(-half_length, -half_length, -half_length), Pnt(half_length, half_length, half_length))
        sph = Sphere(Pnt(0,0,0), r=half_length)
        sph.maxh = self.h
        sph.mat('sphere')
        sph.bc('outer')
        nmesh = OCCGeometry(sph).GenerateMesh()

        self.mesh = Mesh(nmesh)
        numelements = self.mesh.ne  # Count the number elements

        self.mesh.Curve(5)

        curve = 5
        self.mesh.Curve(curve)  # This can be used to set the degree of curved elements
        print("mesh contains " + str(numelements) + " elements")

    def generate_FES(self):
        self.fes = HCurl(self.mesh, order=self.p, dirichlet='outer', complex=True, autoupdate=True)
        self.u = self.fes.TrialFunction()
        self.v = self.fes.TestFunction()
        print(f'NDOF = {self.fes.ndof}')
        self.fes.Update()

    def generate_exact_solution(self):

        phasor = ng_exp(1j * ((self.wavenumber[0] * ng_x) + (self.wavenumber[1] * ng_y) + self.wavenumber[2] * ng_z))
        ex = self.amp[0] * phasor
        ey = self.amp[1] * phasor
        ez = self.amp[2] * phasor

        self.e_exact = CoefficientFunction((ex, ey, ez))#, ey, ez]

    def generate_bilinear_linear_forms(self):
        # LHS = 0
        k_squared = (self.wavenumber[0]**2 + self.wavenumber[1]**2 + self.wavenumber[2]**2)
        A = BilinearForm(self.fes, condense=True)
        A += SymbolicBFI(curl(self.u) * (curl(self.v)), bonus_intorder=2)
        A += SymbolicBFI((-k_squared) * self.u * (self.v), bonus_intorder=2)

        self.P = Preconditioner(A, self.preconditioner)  # Apply the preconditioner  before assembly

        self.mesh, self.fes = self.refine_mesh(self.mesh, self.fes)

        self.A = A.Assemble()

        # RHS:
        F = LinearForm(self.fes)
        F += SymbolicLFI(0)
        self.F = F.Assemble()
        self.P.Update()

        # Setting boundary conditions
        self.sol = GridFunction(self.fes)
        self.sol.Set(self.e_exact, BND)

    def apply_postprojection(self):
        fes = HCurl(self.mesh, order=self.p, dirichlet='outer', complex=False)
        u, v = fes.TnT()
        m = BilinearForm(u * v * dx)
        m.Assemble()
        # build gradient matrix as sparse matrix (and corresponding scalar FESpace)
        gradmat, fesh1 = self.fes.CreateGradient()
        gradmattrans = gradmat.CreateTranspose()  # transpose sparse matrix
        math1 = gradmattrans @ m.mat @ gradmat  # multiply matrices
        math1[0, 0] += 1  # fix the 1-dim kernel
        invh1 = math1.Inverse(inverse="sparsecholesky")

        # build the Poisson projector with operator Algebra:
        proj = IdentityMatrix() - gradmat @ invh1 @ gradmattrans @ m.mat

        self.P = proj @ self.P.mat

    def solve(self):

        # Solve the problem (including static condensation)
        self.F.vec.data += self.A.harmonic_extension_trans * self.F.vec
        res = self.F.vec.CreateVector()
        res.data = self.F.vec - (self.A.mat * self.sol.vec)

        if self.solver == 'CG':
            print('Solving using NGSolve CG')
            inverse = CGSolver(self.A.mat, self.P.mat, precision=self.tol, maxsteps=self.max_iter, printrates=True)
            self.sol.vec.data += inverse * res
        elif self.solver == 'GMRES':
            print('Solving using NGSolve GMRES')
            inverse = GMRESSolver(self.A.mat, self.P.mat, precision=self.tol, maxsteps=self.max_iter, printrates=True)
            self.sol.vec.data += inverse * res
        elif self.solver == 'scipy':
            u = self.scipy_solve(res)
            self.sol.vec.data += u


        self.sol.vec.data += self.A.harmonic_extension * self.sol.vec
        self.sol.vec.data += self.A.inner_solve * self.F.vec
        print("finished solve")

    def refine_mesh(self, mesh, fes):
        numelements = self.mesh.ne
        # Generating multigrid refinement levels
        for l in range(self.refinement_levels):
            if l > 0:
                mesh.Refine()
            print(f'Refinement Factor = {mesh.ne / numelements}')
            numelements = mesh.ne
            fes.Update()

        return mesh, fes

    def compute_eigenspectum(self, n=5, which='LM', tol=1e-10, use_NGSolve=False):

        print('Computing Eigenvalues')

        if use_NGSolve is True:
            lams = EigenValues_Preconditioner(mat=self.A.mat, pre=self.P, tol=tol)
            return lams
        else:

            Precond_A = self.P.mat @ self.A.mat

            tmp1 = self.F.vec.CreateVector()
            tmp2 = self.F.vec.CreateVector()

            def matvec(v):
                tmp1.FV().NumPy()[:] = v
                tmp2.data = Precond_A * tmp1
                tmp2.data =tmp2
                return tmp2.FV().NumPy()

            A = sp.sparse.linalg.LinearOperator((self.A.mat.height, self.A.mat.width), matvec)
            e,_ = sp.sparse.linalg.eigs(A, k=n, which=which, tol=tol)

            return e

    def scipy_solve(self, res):
        print('Solving using scipy GMRES')
        scipy_random_solve()
        # Preconditioner to remove coupled degrees of freedom for static condensation.
        pre = Projector(mask=self.fes.FreeDofs(coupling=True), range=True)

        # Setting up linear operator function that takes v and returns Av
        tmp1 = self.F.vec.CreateVector()
        tmp2 = self.F.vec.CreateVector()
        def matvec(v):
            tmp1.FV().NumPy()[:] = v
            tmp2.data = self.A.mat * tmp1
            tmp2.data = pre * tmp2
            return tmp2.FV().NumPy()

        # rows, cols, vals = self.A.mat.COO()
        # M = sp.sparse.csr_matrix((vals,(rows,cols)))
        #
        #
        #
        # M = M.todense()
        # print(M)
        # print(np.sum(M))
        # print(np.std(M))
        #
        # print(f'cond = {max(np.linalg.eigvals(M)) / min(np.linalg.eigvals(M))}')

        # from IPython import embed
        # embed()

        r2 = res.CreateVector()
        r2.data = pre * res  # Applying pre to both the left and right hand sides.
        u = self.sol.vec.CreateVector()

        # Solve
        A = sp.sparse.linalg.LinearOperator((self.A.mat.height, self.A.mat.width), matvec)

        counter = iterative_solver_counter() # timing is done using ns precision, so counter is initialised immediately before solver.
        u.FV().NumPy()[:], succ = sp.sparse.linalg.gmres(A, r2.FV().NumPy(), tol=self.tol, maxiter=self.max_iter, M=self.P, callback=counter)
        print(f'Passed forward solve check: {np.allclose(A @ (u.FV().NumPy()[:]), r2.FV().NumPy()[:])}')

        # Plotting convergence.
        if self.solver_residual_plot is True:
            ls = ['-', '--', '-.']

            if self.use_PML is True:
                self.solver_residual_plot_label = 'With PML'
            else:
                self.solver_residual_plot_label = 'Without PML'

            plt.figure(self.solver_residual_plot_fignum + self.p)
            counter.setup_plot_params(label=self.solver_residual_plot_label, xlim='')
            counter.plot(label=True)
            plt.show()

            # plt.figure(self.solver_residual_plot_fignum + self.p + 10)
            # counter.setup_plot_params(label=self.solver_residual_plot_label, )
            # counter.plot_time(label=True)
            # plt.show()


        return u

    def run(self, **kwargs):

        for key, value in kwargs.items():
            self.__setattr__(key, value)

        print(f'Using p={self.p}')
        print(f'Using h={self.h}')

        self.generate_mesh()
        self.generate_FES()
        self.generate_exact_solution()
        self.generate_bilinear_linear_forms()
        # self.apply_postprojection()
        # self.solve()

        return self.sol

if __name__ == '__main__':
    W = wave_propagation()
    W.run(p=2)
    eigs = W.compute_eigenspectum()


