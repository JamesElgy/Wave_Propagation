from netgen.csg import unit_cube
from netgen.geom2d import unit_square
from ngsolve import *
from ngsolve.la import EigenValues_Preconditioner

mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))

# ngsglobals.msg_level = 3

class MultiGrid(BaseMatrix):
    def __init__(self, bfa, smoothingsteps=1, cycle=1, endlevel=0):
        super(MultiGrid, self).__init__()
        self.bfa = bfa
        self.inv = bfa.mat.Inverse(bfa.space.FreeDofs())
        self.mats = [bfa.mat]
        self.smoothers = [()]
        self.smoothingsteps = smoothingsteps
        self.cycle = cycle
        self.endlevel = endlevel

    def Update(self):
        self.mats.append(self.bfa.mat) # Appending new sparse matrix to hierarchy.
        blocks = []
        freedofs = self.bfa.space.FreeDofs()
        for i in range(len(freedofs)):
            if freedofs[i]:
                blocks.append((i,))
        self.smoothers.append(self.bfa.mat.CreateBlockSmoother(blocks))

    def Height(self):
        return self.bfa.mat.height

    def Width(self):
        return self.bfa.mat.width

    def Mult(self, b, x):
        self.MGM(len(self.mats) - 1, b, x)

    def MGM(self, level, b, x):
        if level > 0:
            prol = self.bfa.space.Prolongation()
            nc = self.mats[level - 1].height
            d = b.CreateVector()
            w = b.CreateVector()

            x[:] = 0
            # pre-smoothing Jacobi/Gauss-Seidel
            for i in range(self.smoothingsteps):
                self.smoothers[level].Smooth(x, b)

            # coarse grid correction
            for cgc in range(self.cycle):
                d.data = b - self.mats[level] * x
                prol.Restrict(level, d)
                self.MGM(level - 1, d[0:nc], w[0:nc])

                print(level)
                prol.Prolongate(level, w)
                x.data += w

            # post-smoothing Jacobi/Gauss-Seidel
            for i in range(self.smoothingsteps):
                self.smoothers[level].SmoothBack(x, b)
        else:
            x.data = self.inv * b


fes = HCurl(mesh, order=1, dirichlet="left", low_order_space=False)

u, v = fes.TnT()
a = BilinearForm(fes)
a += SymbolicBFI(u * v + curl(u) * curl(v))
a.Assemble()
f = LinearForm(fes)
f += SymbolicLFI(curl(v))
f.Assemble()

pre = MultiGrid(a, smoothingsteps=1)

with TaskManager():
    for l in range(9):
        print("level", l)
        mesh.Refine()
        fes.Update()
        print("ndof = ", fes.ndof)
        a.Assemble()
        f.Assemble()
        pre.Update()

        # lam = EigenValues_Preconditioner(mat=a.mat, pre=pre)
        # print("eigenvalues: ", lam)

        gfu = GridFunction(fes)
        inv = CGSolver(mat=a.mat, pre=pre, printrates=True)
        gfu.vec.data = inv * f.vec

