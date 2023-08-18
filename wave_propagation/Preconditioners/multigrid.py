# James Elgy - 10/08/2023

from ngsolve import BaseMatrix
from ngsolve import BaseMatrix

# ngsglobals.msg_level = 3


class MGPreconditioner(BaseMatrix):
    def __init__(self, fes, level, mat, coarsepre,static_cond, finest_level=0):
        super().__init__()
        self.fes = fes
        self.level = level
        self.mat = mat
        self.coarsepre = coarsepre
        if level > 0:
            self.localpre = mat.CreateSmoother(fes.FreeDofs(coupling=static_cond))
        else:
            self.localpre = mat.Inverse(fes.FreeDofs(coupling=static_cond))

    def Mult(self, d, w):
        if self.level == 0:
            w.data = self.localpre * d
            return

        prol = self.fes.Prolongation().Operator(self.level)

        w[:] = 0
        self.localpre.Smooth(w, d)
        r = d - self.mat * w
        r = r.Evaluate()
        w += prol @ self.coarsepre @ prol.T * r
        self.localpre.SmoothBack(w, d)

    def Shape(self):
        return self.localpre.shape

    def CreateVector(self, col):
        return self.localpre.CreateVector(col)


class MultiGrid(BaseMatrix):
    """
    A recursive implementation of Multigrid preconditioner class taken from
     https://ngsolve.org/forum/ngspy-forum/899-adaptive-mesh-refinement-interpolation-operator

    Adapted by adding coupling argument to remove interior DOFs and endlevel argument to specify a number of desired restrictions.
    endlevel=0 corresponds to doing a direct solve on the coarsest grid in the hierarchy.
    """


    def __init__(self, bfa, smoothingsteps=1, cycle=1, endlevel=0, coupling=True):
        super().__init__()
        self.bfa = bfa
        self.inv = bfa.mat.Inverse(bfa.space.FreeDofs(coupling=coupling), 'sparsecholesky')
        self.mats = [bfa.mat]
        self.smoothers = [()]
        self.smoothingsteps = smoothingsteps
        self.cycle = cycle
        self.endlevel = endlevel
        self.coupling = coupling

    def Update(self, block_dofs=''):
        # self.mats.append(self.bfa.mat)
        # self.smoothers.append(self.bfa.mat.CreateSmoother(self.bfa.space.FreeDofs(coupling=self.coupling)))
        self.mats.append (self.bfa.mat)
        blocks = []
        freedofs = self.bfa.space.FreeDofs(coupling=self.coupling)


        for item in block_dofs.split('|'):
            if item != '':
                if item == 'vertices':
                    dofs = self.bfa.space.mesh.vertices
                elif item == 'edges':
                    dofs = self.bfa.space.mesh.edges
                elif item == 'faces':
                    dofs = self.bfa.space.mesh.faces
                elif item == 'facets':
                    dofs = self.bfa.space.mesh.facets

                for v in dofs:
                    vdofs = set()
                    for el in self.bfa.space.mesh[v].elements:
                        vdofs |= set(d for d in self.bfa.space.GetDofNrs(el)
                                     if freedofs[d])
                    blocks.append(vdofs)

            if item == '':
                for i in range(len(freedofs)):
                    if freedofs[i]:
                        blocks.append ( (i,) )

        self.smoothers.append(self.bfa.mat.CreateBlockSmoother(blocks))


    def Height(self):
        return self.bfa.mat.height

    def Width(self):
        return self.bfa.mat.width

    def Mult(self, b, x):
        self.MGM(len(self.mats) - 1, b, x)

    def MGM(self, level, b, x):
        # print(f'Working on level {level})
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

                # From forum post (https://forum.ngsolve.org/t/what-block-preconditioners-really-do/1525/7), the
                # prolongation operators only project the lowest order dofs. This is also supported by the NGSolve
                # documentation.
                # "NGSolve’s multigrid implementation for a high order method uses h-version multigrid for the lowest
                # order block, and local block-smoothing for the high order bubble basis functions."
                #

                prol.Restrict(level, d)
                self.MGM(level - 1, d[0:nc], w[0:nc])

                # print(level)
                prol.Prolongate(level, w)
                x.data += w

            # post-smoothing Jacobi/Gauss-Seidel
            for i in range(self.smoothingsteps):
                self.smoothers[level].SmoothBack(x, b) # pre and post smoothing contributes most of the time taken
        else:
            x.data = self.inv * b



if __name__ == '__main__':
    pass