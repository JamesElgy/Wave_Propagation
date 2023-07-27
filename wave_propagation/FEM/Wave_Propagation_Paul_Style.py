# Import libaries
# import netgen.gui
import numpy as np
import netgen.meshing as ngmeshing
from ngsolve import *
from netgen.occ import *
import scipy.sparse as sp
import gc

# Specify order of elements
for Order in [0, 1, 2, 3]:
    # Specify the mesh file
    # Object = "OCC_sphere.vol"
    # # Loading the object file
    # ngmesh = ngmeshing.Mesh(dim=3)
    # ngmesh.Load("VolFiles/" + Object)
    #
    # # Creating the mesh and defining the element types
    # mesh = Mesh("VolFiles/" + Object)
    # curve = 5
    # mesh.Curve(curve)  # This can be used to set the degree of curved elements
    # numelements = mesh.ne  # Count the number elements
    # print(" mesh contains " + str(numelements) + " elements")

    box = Sphere(Pnt(0,0,0), r=20)
    box.maxh = 2
    box.bc('outer')

    nmesh = OCCGeometry(box).GenerateMesh()
    mesh = Mesh(nmesh)
    curve = 5
    mesh.Curve(curve)
    print(mesh.ne)


    # define material constants
    Mu0 = 4. * np.pi * 1e-7
    # Epsilon = 1e-6

    # define tolerances etc
    Maxsteps = 500
    Tolerance = 1e-8
    Solver = "direct"

    print("Order=", Order)

    fes = HCurl(mesh, order=Order, dirichlet="outer", complex=True)

    # Count the number of degrees of freedom
    ndof = fes.ndof
    print("ndof", ndof)

    # Set up the grid function and apply Dirichlet BC
    a = GridFunction(fes)

    k = np.asarray([1,0,0]) * 0.1
    Epsilon = sum(np.asarray(k)**2)
    amp = [0,0,1]
    # Setup boundary condition
    def axout(x, y, z, k, amp):
        p = amp[0] * exp(1j* (k[0]*x + k[1]*y + k[2]*z))
        return p


    def ayout(x, y, z, k, amp):
        p = amp[1] * exp(1j* (k[0]*x + k[1]*y + k[2]*z))
        return p


    def azout(x, y, z, k, amp):
        p = amp[2] * exp(1j* (k[0]*x + k[1]*y + k[2]*z))
        return p

    # curl a = [ 0, 3z**2, 0]
    # curl curl a  = [ -6z,  0,0]

    a.Set((axout(x, y, z, k, amp), ayout(x, y, z, k, amp), azout(x, y, z, k, amp)), BND)

    # Setup source condition
    # src = CoefficientFunction((-6 * z, 0, 0))
    # src = 0
    # Test and trial functions
    u = fes.TrialFunction()
    v = fes.TestFunction()

    Additional_Int_Order = 2
    # Create the linear and bilinear forms (nb we get the linear system TSM.mat a.vec.data = - R = f.vec.data)
    f = LinearForm(fes)
    # f += SymbolicLFI((src * v), bonus_intorder=Additional_Int_Order)
    f += SymbolicLFI(0)


    TSM = BilinearForm(fes, symmetric=True, condense=True)
    TSM += SymbolicBFI((curl(u) * curl(v)), bonus_intorder=Additional_Int_Order)
    TSM += SymbolicBFI(-Epsilon * (u * v), bonus_intorder=Additional_Int_Order)

    if Solver == "direct":
        P = Preconditioner(TSM, "direct")
    if Solver == "bddc":
        P = Preconditioner(TSM, "bddc")  # Apply the bddc preconditioner
    if Solver == "local":
        P = Preconditioner(TSM, "local")  # Apply the local preconditioner

    TSM.Assemble()
    f.Assemble()

    P.Update()

    # Solve the problem (including static condensation)
    f.vec.data += TSM.harmonic_extension_trans * f.vec
    res = f.vec.CreateVector()
    res.data = f.vec - (TSM.mat * a.vec)
    inverse = CGSolver(TSM.mat, P.mat, precision=Tolerance, maxsteps=Maxsteps, printrates=False)
    # inverse = GMRESSolver(TSM.mat, P.mat, printrates=False, precision=Tolerance, maxsteps=Maxsteps)
    a.vec.data += inverse * res
    a.vec.data += TSM.harmonic_extension * a.vec

    a.vec.data += TSM.inner_solve * f.vec
    print("finished solve")

    exact = CoefficientFunction((axout(x, y, z, k, amp), ayout(x, y, z, k, amp), azout(x, y, z, k, amp)))

    # # Compute L2 norm of curl error = curl (a-aexact) = curl (a -aexact) = curl (a)  -bexact
    Integration_Order = np.max([4*(Order+1),3*(curve-1)])
    # Ierrinside = Integrate(InnerProduct(curl(a)-0,curl(a)-0),mesh,order=Integration_Order)

    # print(f'Error in curl = {Ierrinside}')

    Ierrinside = Integrate(InnerProduct(a-exact,a-exact),mesh,order=Integration_Order)**(0.5) / Integrate(InnerProduct(exact,exact),mesh,order=Integration_Order)**(0.5)
    print(f'Error in a = {Ierrinside}')

# Draw(a, mesh, 'sol')
# Draw(exact, mesh, 'exact')
#
# gex = GridFunction(fes)
# gex.Set(exact, BND)
# Draw(gex, mesh, 'exact_grid')
