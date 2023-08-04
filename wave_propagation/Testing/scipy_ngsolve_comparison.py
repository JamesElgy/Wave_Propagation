# James Elgy - 01/08/2023

from ..FEM.Wave_Propagation import wave_propagation as wp
from ngsolve import Integrate, InnerProduct

def scipy_ngsolve_comparison():
    W1 = wp()
    sol1 = W1.run(p=1, solver='CG', preconditioner='direct')

    W2 = wp()
    sol2 = W2.run(p=1, solver='scipy', preconditioner='direct')

    err = Integrate(InnerProduct(W1.sol - W2.sol, W1.sol-W2.sol), W1.mesh) ** 0.5
    err /= Integrate(InnerProduct(W1.sol, W1.sol), W1.mesh) ** 0.5

    return abs(err)

if __name__ == '__main__':
    err = scipy_ngsolve_comparison()
    print(err)
