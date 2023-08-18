# Import libaries

# import netgen.gui
import time

import matplotlib.pyplot as plt

import wave_propagation.Saving.save_figs

plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts, amssymb}')
import numpy as np
import netgen.meshing as ngmeshing
from ngsolve import *
from netgen.occ import *
import scipy.sparse as sp
import gc
from memory_profiler import memory_usage
from itertools import cycle
from wave_propagation.Preconditioners.multigrid import MultiGrid as mg

import pprofile

ngsglobals.msg_level = 3

def generate_mesh(h):
    sph = Sphere(Pnt(0,0,0), r=1)
    sph.mat('copper')
    sph.maxh = h
    box = Box(Pnt(-50,-50,-50), Pnt(50,50,50))
    box.mat('air')
    box.bc('outer')

    geo = Glue([sph, box])
    mesh = Mesh(OCCGeometry(geo).GenerateMesh())
    return mesh


def run(Solver='multigrid', block_dofs='vertices'):
    # Specify order of elements

    for h in [0.5]:#, 0.25, 0.125]:
        err = []
        nd = []
        for Order in [0]:

            times = {'start_definitions': time.time()}

            # Specify the mesh file
            Object = "OCC_sphere.vol"
            # Loading the object file
            ngmesh = ngmeshing.Mesh(dim=3)
            ngmesh.Load("VolFiles/" + Object)

            # Creating the mesh and defining the element types
            mesh = Mesh("VolFiles/" + Object)

            # mesh = generate_mesh(h)

            curve = 4
            mesh.Curve(curve)  # This can be used to set the degree of curved elements
            numelements = mesh.ne  # Count the number elements
            print(" mesh contains " + str(numelements) + " elements")

            # Materials consist of sphere material and air in the order as defined in the mesh
            matlist = ["copper", "air"]
            contrast = 20
            murlist = [contrast, 1]
            inout = []
            for mat in matlist:
                if mat == "air":
                    inout.append(0)
                else:
                    inout.append(1)
            inorout = dict(zip(matlist, inout))
            murmat = dict(zip(matlist, murlist))

            # Coefficient functions
            mur_coef = [murmat[mat] for mat in mesh.GetMaterials()]
            mur = CoefficientFunction(mur_coef)
            # inout = 1 in B, inout =0 in R^3 \ B
            inout_coef = [inorout[mat] for mat in mesh.GetMaterials()]
            inout = CoefficientFunction(inout_coef)

            # define material constants
            Mu0 = 4. * np.pi * 1e-7
            Epsilon = 1e-6

            # Coefficent function for background field
            B0 = 1
            H0 = CoefficientFunction((0, 0, B0 / Mu0))
            alpha = 1
            exactsphere = 1

            # define tolerances etc
            Maxsteps = 500
            Tolerance = 1e-8
            # Solver = "multigrid"

            print("Order=", Order)

            dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
            fes = HCurl(mesh, order=Order, dirichlet="outer", complex=False, gradientdomains=dom_nrs_metal)

            # Count the number of degrees of freedom
            ndof = fes.ndof
            # print("ndof", ndof)

            # Set up the grid function and apply Dirichlet BC
            a = GridFunction(fes)

            # Setup boundary condition
            if exactsphere == 0:
                a.Set((0, 0, 0), BND)
            else:
                def axout(x, y, z):
                    r = sqrt(x ** 2 + y ** 2 + z ** 2);
                    theta = acos(z / r);
                    phi = atan2(y, x);
                    Aphi = 0.5 * B0 * (r - 2 * (1 - contrast) / (2 + contrast) * alpha ** 3 / r ** 2) * sin(theta);
                    return -sin(phi) * Aphi


                def ayout(x, y, z):
                    r = sqrt(x ** 2 + y ** 2 + z ** 2);
                    theta = acos(z / r);
                    phi = atan2(y, x);
                    Aphi = 0.5 * B0 * (r - 2 * (1 - contrast) / (2 + contrast) * alpha ** 3 / r ** 2) * sin(theta);
                    return cos(phi) * Aphi


                def azout(x, y, z):
                    return 0.


                a.Set((axout(x, y, z), ayout(x, y, z), azout(x, y, z)), BND)

            # Setup source condition
            src = CoefficientFunction((0, 0, 0))

            # Test and trial functions
            u = fes.TrialFunction()
            v = fes.TestFunction()

            Additional_Int_Order = 2
            # Create the linear and bilinear forms (nb we get the linear system TSM.mat a.vec.data = - R = f.vec.data)
            f = LinearForm(fes)
            f += SymbolicLFI((src * v), bonus_intorder=Additional_Int_Order)

            TSM = BilinearForm(fes, symmetric=True, condense=True)
            TSM += SymbolicBFI(inout * (mur ** (-1)) * (curl(u) * curl(v)), bonus_intorder=Additional_Int_Order)
            TSM += SymbolicBFI((1 - inout) * (curl(u) * curl(v)), bonus_intorder=Additional_Int_Order)
            TSM += SymbolicBFI(Epsilon * (u * v), bonus_intorder=Additional_Int_Order)

            times['finished_definitions'] = time.time()
            times['start_mesh_refinement'] = time.time()

            if Solver == 'multigrid':
                P = Preconditioner(TSM, 'multigrid')

            print(f'NE = {mesh.ne}')
            for l in range(3):
                ne = mesh.ne
                if l > 0:
                    for el in mesh.Elements():
                        mesh.SetRefinementFlag(el, el.mat == 'copper')
                    mesh.Refine(mark_surface_elements=True)
                    print(f'NE = {mesh.ne}, Refinement Factor = {mesh.ne / ne}')
                    # Draw(mesh)
                    ne = mesh.ne

                fes.Update()
                a.Update()
                TSM.Assemble()
                f.Assemble()

                if l == 0 and Solver == 'mg' or Solver == 'bddc_mg':
                    P = mg(TSM, coupling=True, smoothingsteps=1, cycle=1)
                elif l > 0 and Solver == 'mg' or Solver == 'bddc_mg':
                    P.Update(block_dofs=block_dofs)

            times['finished_mesh_refinement'] = time.time()
            times['start_creating_preconditioner'] = time.time()

            if Solver == 'multigrid':

                TSM.Assemble()
                f.Assemble()
                fes.Update()



            # Resetting BC:
            if exactsphere != 0:
                a.Set((axout(x, y, z), ayout(x, y, z), azout(x, y, z)), BND)
            else:
                a.Set((0,0,0), BND)
            a.Update()

            if Solver == "bddc" or Solver == 'bddc_mg':

                if Solver == 'bddc_mg':
                    P_mg = P
                    P = Preconditioner(TSM, "bddc", coarsetype=P_mg)  # Apply the bddc preconditioner
                else:
                    P = Preconditioner(TSM, "bddc")  # Apply the bddc preconditioner

                print("using bddc")
                TSM.Assemble()
                f.Assemble()
                P.Update()
            if Solver == "local":
                P = Preconditioner(TSM, "local")  # Apply the local preconditioner
                TSM.Assemble()
                f.Assemble()
                P.Update()

            times['finished_creating_preconditioner'] = time.time()
            times['start_solve'] = time.time()


            # Solve the problem (including static condensation)
            f.vec.data += TSM.harmonic_extension_trans * f.vec
            res = f.vec.CreateVector()
            res.data = f.vec - (TSM.mat * a.vec)

            profiler = pprofile.Profile()
            with profiler:
                inverse = CGSolver(TSM.mat, P, precision=Tolerance, maxsteps=Maxsteps, printrates=True)
                a.vec.data += inverse * res

            # profiler.print_stats()
            a.vec.data += TSM.harmonic_extension * a.vec

            a.vec.data += TSM.inner_solve * f.vec
            print("finished solve")

            times['finished_solve'] = time.time()

            apply_postprocessing = False
            if apply_postprocessing is True:
                times['start_postprocessing'] = time.time()

                times['start_assembly'] = time.time()
                # Poission Projection to acount for gradient terms:
                u, v = fes.TnT()
                m = BilinearForm(fes)
                m += u * v * dx
                m.Assemble()
                times['finished_assembly'] = time.time()

                # build gradient matrix as sparse matrix (and corresponding scalar FESpace)
                gradmat, _ = fes.CreateGradient()

                gradmattrans = gradmat.CreateTranspose()  # transpose sparse matrix

                times['start_math1'] = time.time()
                math1 = gradmattrans @ m.mat @ gradmat  # multiply matrices
                times['finished_math1'] = time.time()
                math1[0, 0] += 1  # fix the 1-dim kernel
                invh1 = math1.Inverse(inverse="sparsecholesky") # Inverse is doing an entire inverse. Does not work in same way as CGSolver

                # build the Poisson projector with operator Algebra:
                times['start_projector'] = time.time()
                proj = IdentityMatrix() - gradmat @ invh1 @ gradmattrans @ m.mat
                times['finished_projector'] = time.time()
                a.vec.data = proj * (a.vec)
                # times['finished_projector'] = time.time()
                print("applied projection")
                del proj, math1, invh1, m, gradmattrans, gradmat
                times['finished_postprocessing'] = time.time()



            times['start_calculating_errors'] = time.time()
            compute_errors = False
            if compute_errors is True:
                # Compute errors

                # Compare with the exact solution
                # Construct it inside the object

                if exactsphere == 1:
                    def bxin(x, y, z):
                        r = sqrt(x ** 2 + y ** 2 + z ** 2);
                        phi = atan2(y, x)
                        theta = acos(z / r)
                        Br = 3 * B0 * cos(theta) / (1 + 2 / contrast)
                        Btheta = -3 * B0 * sin(theta) / (1 + 2 / contrast)
                        return sin(theta) * cos(phi) * Br + cos(theta) * cos(phi) * Btheta


                    def byin(x, y, z):
                        r = sqrt(x ** 2 + y ** 2 + z ** 2);
                        phi = atan2(y, x)
                        theta = acos(z / r)
                        Br = 3 * B0 * cos(theta) / (1 + 2 / contrast)
                        Btheta = -3 * B0 * sin(theta) / (1 + 2 / contrast)
                        return sin(theta) * sin(phi) * Br + cos(theta) * sin(phi) * Btheta


                    def bzin(x, y, z):
                        r = sqrt(x ** 2 + y ** 2 + z ** 2);
                        phi = atan2(y, x)
                        theta = acos(z / r)
                        Br = 3 * B0 * cos(theta) / (1 + 2 / contrast)
                        Btheta = -3 * B0 * sin(theta) / (1 + 2 / contrast)
                        return cos(theta) * Br - sin(theta) * Btheta


                    bexactinside = CoefficientFunction((bxin(x, y, z), byin(x, y, z), bzin(x, y, z)))

                    # Construct it outside the object
                    #       Do not use Bexact obtained from differentiating solution - has
                    #       precision issues. Recall that for a sphere (H_alpha - H_0 ) = D2G(x,z)M H_0 is
                    #       exact when H_0 is uniform
                    # def logr(r):
                    #    out=0.
                    #    for n in range(20):
                    #        out = out + 1/(1+2*n)*((r-1)/(r+1))**(n+1+2*n)
                    #    return 2*out

                    # Construct it outside the object
                    # def bxout(x,y,z):
                    #    r = sqrt(x**2+y**2+z**2);
                    #    phi = atan2(y,x)
                    #    theta = acos(z/r)
                    #    Br = B0*(1-2*(1-contrast)/(2+contrast)*alpha**3/r**3)*cos(theta);
                    #    #logr =  2 *( (r-1)/(r+1)  + (1/3)*( (r-1)/(r+1) )**3 + (1/5)* ( (r-1)/(r+1) )**5 + (1/7) *( (r-1)/(r+1) )**7 )
                    #    Btheta = -B0*sin(theta)*(1-(1-contrast)/(2+contrast)*alpha**3*logr(r)/r);
                    #    return sin(theta)*cos(phi)*Br+cos(theta)*cos(phi)*Btheta

                    # def byout(x,y,z):
                    #    r = sqrt(x**2+y**2+z**2);
                    #    phi = atan2(y,x)
                    #    theta = acos(z/r)
                    #    Br = B0*(1-2*(1-contrast)/(2+contrast)*alpha**3/r**3)*cos(theta);
                    #    #logr =  2 *( (r-1)/(r+1)  + (1/3)*( (r-1)/(r+1) )**3 + (1/5)* ( (r-1)/(r+1) )**5 + (1/7) *( (r-1)/(r+1) )**7 )
                    #    Btheta = -B0*sin(theta)*(1-(1-contrast)/(2+contrast)*alpha**3*logr(r)/r);
                    #    return sin(theta)*sin(phi)*Br+cos(theta)*sin(phi)*Btheta

                    # def bzout(x,y,z):
                    #    r = sqrt(x**2+y**2+z**2);
                    #    phi = atan2(y,x)
                    #    theta = acos(z/r)
                    #    Br = B0*(1-2*(1-contrast)/(2+contrast)*alpha**3/r**3)*cos(theta);
                    #    #logr =  2 *( (r-1)/(r+1)  + (1/3)*( (r-1)/(r+1) )**3 + (1/5)* ( (r-1)/(r+1) )**5 + (1/7) *( (r-1)/(r+1) )**7 )
                    #    Btheta = -B0*sin(theta)*(1-(1-contrast)/(2+contrast)*alpha**3*logr(r)/r);
                    #    return cos(theta)*Br-sin(theta)*Btheta

                    Pi = 3.141592653589793


                    def bxout(x, y, z):
                        r = sqrt(x ** 2 + y ** 2 + z ** 2)
                        rx = x / r
                        ry = y / r
                        rz = z / r
                        M = 4 * Pi * alpha ** 3 * (contrast - 1) / (2 + contrast)
                        return M / (4 * Pi * r ** 3) * 3 * rx * rz * B0


                    def byout(x, y, z):
                        r = sqrt(x ** 2 + y ** 2 + z ** 2)
                        rx = x / r
                        ry = y / r
                        rz = z / r
                        M = 4 * Pi * alpha ** 3 * (contrast - 1) / (2 + contrast)
                        return M / (4 * Pi * r ** 3) * 3 * ry * rz * B0


                    def bzout(x, y, z):
                        r = sqrt(x ** 2 + y ** 2 + z ** 2)
                        rx = x / r
                        ry = y / r
                        rz = z / r
                        M = 4 * Pi * alpha ** 3 * (contrast - 1) / (2 + contrast)
                        return M / (4 * Pi * r ** 3) * (3 * rz * rz - 1) * B0 + B0


                    bexactoutside = CoefficientFunction((bxout(x, y, z), byout(x, y, z), bzout(x, y, z)))


                    def axin(x, y, z):
                        r = sqrt(x ** 2 + y ** 2 + z ** 2);
                        theta = acos(z / r);
                        phi = atan2(y, x);
                        Aphi = 3 * B0 * r * sin(theta) / (2 * (1 + 2 / contrast));
                        return -sin(phi) * Aphi  # -(-B0*y/2.)


                    def ayin(x, y, z):
                        r = sqrt(x ** 2 + y ** 2 + z ** 2);
                        theta = acos(z / r);
                        phi = atan2(y, x);
                        Aphi = 3 * B0 * r * sin(theta) / (2 * (1 + 2 / contrast));
                        return cos(phi) * Aphi  # -(B0*x/2.)


                    def azin(x, y, z):
                        return 0. - B0 * 0.


                    aexactoutside = CoefficientFunction((axout(x, y, z), ayout(x, y, z), azout(x, y, z)))
                    aexactinside = CoefficientFunction((axin(x, y, z), ayin(x, y, z), azin(x, y, z)))

                    # Compute L2 norm of curl error = curl (a-aexact) = curl (a -aexact) = curl (a)  -bexact
                    Integration_Order = np.max([4 * (Order + 1), 3 * (curve - 1)])

                    Ierrinside = Integrate(inout * InnerProduct(curl(a) - bexactinside, curl(a) - bexactinside), mesh,
                                           order=Integration_Order)
                    Ierroutside = Integrate((1 - inout) * InnerProduct(curl(a) - bexactoutside, curl(a) - bexactoutside), mesh,
                                            order=Integration_Order)
                    Ierrtot = Ierroutside + Ierrinside
                    Ininside = Integrate(inout * InnerProduct(bexactinside, bexactinside), mesh, order=Integration_Order)
                    Inoutside = Integrate((1 - inout) * InnerProduct(bexactoutside, bexactoutside), mesh, order=Integration_Order)
                    Intot = Inoutside + Ininside
                    print("Order=", Order, "Error in curl", Ierrtot / Intot)
                    print("Order=", Order, "Error in curl inside", Ierrinside / Ininside)
                    print("Order=", Order, "Error in curl outside", Ierroutside / Inoutside)
                    err += [Ierrtot / Intot]
                    nd += [fes.ndof]

                    Ierrinside = Integrate(inout * InnerProduct(a - aexactinside, a - aexactinside), mesh, order=Integration_Order)
                    Ierroutside = Integrate((1 - inout) * InnerProduct(a - aexactoutside, a - aexactoutside), mesh,
                                            order=Integration_Order)
                    Ierrtot = Ierrinside + Ierroutside
                    Ininside = Integrate(inout * InnerProduct(aexactinside, aexactinside), mesh, order=Integration_Order)
                    Inoutside = Integrate((1 - inout) * InnerProduct(aexactoutside, aexactoutside), mesh, order=Integration_Order)
                    Intot = Ininside + Inoutside
                    print("Order=", Order, "Error in a", Ierrtot / Intot)


            times['finished_calculating_errors'] = time.time()
            times['finished_time'] = time.time()
        #
        # plt.figure(999)
        # plt.loglog(nd, err, marker='x', label=f'Mesh NE = {mesh.ne}')
        # plt.xlabel('$N_d$')
        # plt.ylabel(r'Curl Error')
        # plt.legend()
        # wave_propagation.Saving.save_figs.save_all_figures('Results', format='pdf')

    return times


def mem_profile(Solver='multigrid', block_dofs='vertices'):
    start = time.time_ns() / 1e9
    mem, times = memory_usage((run, (), {'Solver':Solver, 'block_dofs':block_dofs}), interval=0.1, retval=True)
    stop = time.time_ns() / 1e9
    t = np.linspace(0, stop - start, len(mem))
    plt.figure(1)
    plt.plot(t, mem, label=f'{Solver}')
    plt.legend()
    plt.xlabel('Time, [s]')
    plt.ylabel('Memory Usage, [MB]')


    plt.figure()
    plt.title(f'{Solver}')
    plot_mem_usage(times, mem)


def plot_mem_usage(timings, mem, add_breakdown=True):

    cols = cycle(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple','tab:pink','tab:cyan'])

    def_time = timings['finished_definitions'] - timings['start_definitions']
    mesh_ref_time = timings['finished_mesh_refinement'] - timings['start_mesh_refinement']
    precond_ref_time = timings['finished_creating_preconditioner'] - timings['start_creating_preconditioner']
    solve_time = timings['finished_solve'] - timings['start_solve']
    # postproc_time = timings['finished_postprocessing'] - timings['start_postprocessing']
    # error_time = timings['finished_calculating_errors'] - timings['start_calculating_errors']

    tot_time = timings['finished_calculating_errors'] - timings['start_definitions']

    time_array = np.linspace(0, tot_time, len(mem))


    if add_breakdown is True:
        t0 = 0
        for dt, labels in zip([def_time, mesh_ref_time, precond_ref_time, solve_time],
                              ['Preprocessing', 'Refining mesh', 'Constructing preconditioner', 'Solving']):

            t1 = t0 + dt
            plt.axvspan(t0, t1, label=labels, color=next(cols), alpha=0.4)
            t0 = t1
        # plt.axvspan(timings['start_projector'] - timings['start_definitions'], timings['finished_projector'] - timings['start_definitions'], color=next(cols), label='Direct inverse for projection matrix', alpha=0.4)
        # plt.axvspan(timings['start_assembly'] - timings['start_definitions'], timings['finished_assembly'] - timings['start_definitions'], color=next(cols), label='Mass matrix assembly', alpha=0.4)
        # plt.axvspan(timings['start_math1'] - timings['start_definitions'], timings['finished_math1'] - timings['start_definitions'], color=next(cols), label='Math1', alpha=0.4)


    # mem_array = [m if m<t1 else None for m in mem]
    # time_array = time_array[0:len(mem_array)]
    plt.plot(time_array, mem, 'k', label='Memory usage')
    plt.legend()
    plt.xlabel('Time, [s]')
    plt.ylabel('Memory Usage, [MB]')
    plt.xlim([0, t1])



if __name__ == '__main__':

    # times = run(Solver='multigrid')
    # times = run(Solver='mg', block_dofs='vertices')
    # times = run(Solver='bddc')
    # times = run(Solver='local')

    # mem_profile(Solver='mg', block_dofs='vertices')
    # plt.figure(2); plt.title('Vertices')
    # mem_profile(Solver='mg', block_dofs='edges|faces|')
    # plt.figure(3); plt.title('Edges')
    # mem_profile(Solver='mg', block_dofs='faces')
    # plt.figure(4); plt.title('Faces')
    # mem_profile(Solver='mg', block_dofs='facets')
    # plt.figure(5); plt.title('Facets')
    mem_profile(Solver='bddc_mg', block_dofs='')
    plt.figure(5); plt.title('None')

    plt.figure(1); plt.legend(['Vertices', 'Edges', 'Faces', 'Facets', 'None'])

    # wave_propagation.Saving.save_figs.save_all_figures('Results', format='pdf', prefix='Sphere_p=2_mgprecond_comp_')


    # mem_profile()
    # plt.figure(2); plt.title('Multigrid')
    # mem_profile(Solver='bddc')
    # plt.figure(3); plt.title('BDDC')
    # mem_profile(Solver='local')
    # plt.figure(4); plt.title('Local')
    # mem_profile(Solver='mg')
    # plt.figure(5); plt.title('Multigrid using NGSolve API')
    #
    # plt.figure(1); plt.legend(['Multigrid', 'BDDC', 'Local', 'API Multigrid'])
    #
    # wave_propagation.Saving.save_figs.save_all_figures('Results', format='pdf', prefix='Sphere_p=2_precond_comp_')