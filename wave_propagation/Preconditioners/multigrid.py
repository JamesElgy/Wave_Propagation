# James Elgy - 10/08/2023

from ngsolve import BaseMatrix
from ngsolve import BaseMatrix

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


if __name__ == '__main__':
    pass