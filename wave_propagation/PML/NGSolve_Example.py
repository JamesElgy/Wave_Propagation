# James Elgy - 29/08/2023

from ngsolve import *
from ngsolve.webgui import Draw
from netgen.occ import *
import matplotlib.pyplot as plt
import numpy as np
def main():
    outer = Circle((0, 0), 2).Face()
    outer.edges.name = 'outerbnd'
    inner = Circle((0, 0), 1).Face()
    inner.edges.name = 'innerbnd'
    inner.faces.name = 'inner'
    pmlregion = outer - inner
    pmlregion.faces.name = 'pmlregion'
    geo = OCCGeometry(Glue([inner, pmlregion]), dim=2)

    mesh = Mesh(geo.GenerateMesh(maxh=0.1))
    mesh.Curve(3)
    print(help(pml.Radial))
    print(help(mesh.SetPML))

if __name__ == '__main__':
    main()
