# from netgen.gui import *

from ngsolve import *
from ngsolve.webgui import Draw
from netgen.occ import *
from netgen.webgui import Draw as Drawgeo
import matplotlib.pyplot as plt
import numpy as np

from scipy.special import jv as besselj
from scipy.special import yv as bessely
from scipy.special import hankel1, hankel2
from scipy.optimize import fmin, minimize

from ngsolve.krylovspace import CGSolver, GMResSolver

# try:
#     from ngsolve.ngscuda import *
# except:
#     print ("no CUDA library or device available, using replacement types on host")



def generate_scattered_E_field(R, theta, omega, max_n=30, TE=False):
    """Generates scattered E field from infinitely long unit cylinder.
    Follows Balanis. Note that we assume that wave is travelling in positive x: e^ikx.

    Args:
        R (Nd_Array): Meshgrid of radii from origin
        theta (Nd_Array): grid of angles at each coord.
        omega (float): Angular Frequency.
        max_n (int, optional): Max order of bessel and hankel functions to consider. Defaults to 30.
        TE (bool, optional): Consider TE(True) or TM(False) polarisations. Defaults to False.
    
    Returns:
        E (Nd_Array): complex 3xMxM grid of the scattered electric field.
    """
    
    # Computing area of xx and yy covered by cylinder.
    mask = np.ones(R.shape)
    for yind in range(R.shape[1]):
        for xind in range(R.shape[0]):
            if R[xind, yind] < 1:
                mask[xind, yind] = np.nan
    
    
    zz = np.zeros(R.shape)
    if TE is True:
        for n in range(max_n):
            dj=(-besselj(n-1,omega)+besselj(n+1,omega))/2
            dy=(-bessely(n-1,omega)+bessely(n+1,omega))/2
            dh=dj-1j*dy

            z=omega*R
            bj=besselj(n,z)
            by=bessely(n,z)
            h=bj-1j*by
            if n==0:
                zz=zz-(dj/dh)*h*np.cos(n*theta)*(1j**(-n)) * mask
            else:
                zz=zz-2*(dj/dh)*h*np.cos(theta*n)*(1j**(-n)) * mask
                
    else:
        for n in range(max_n):
            if n == 0:
                epsilon_n = 1
            else:
                epsilon_n = 2
            
            # for cylinder radius r=1
            
            jv_over_h2 = -besselj(n, omega*1)/hankel1(n, omega*1) # The minus sign here comes from the matlab implementation https://uk.mathworks.com/matlabcentral/fileexchange/30162-cylinder-scattering.
            zz = zz + (-1j)**n * epsilon_n * jv_over_h2 * hankel1(n, omega * R) * np.cos(theta*n) * mask
            zz = -1 * zz
    E = np.asarray([np.zeros(zz.shape), np.zeros(zz.shape), zz])
    return E  

def generate_exact_solution(N_samples, box_size=5, omega=2*np.pi, plot=True, use_exact_H=True):
    """ Generate 'exact' solution for scattered H field. This is done using an exact solution for the E field and computing its curl
    (numerically).
    The problem considers a 2d box containing a unit radius infinitely long cylinder.

    Args:
        N_samples (int): number of samples in each dimension to compute for the scattered E field. Bigger implies more accurate
        computation of curl(E).
        box_size (float, optional): Defaults to 5. size of the square domain to consider.
        omega (float, optional): Frequency. Defaults to 2*np.pi.
        plot (bool, optional): Generate figures showing the exact field. Defaults to True.
        
    Returns:
        Exact (Nd_array): 3xMxM complex scattered H field.
        mask (Nd_array): MxM mask containing nan inside cylinder (and one pixel surronding) else 1.
    """
    
    x_extent = np.linspace(-box_size/2, box_size/2, N_samples)
    y_extent = x_extent
    xx,yy = np.meshgrid(x_extent, y_extent, indexing='ij')

    R = np.sqrt(xx**2 + yy**2) # Cylinder is centered at (0,0)
    theta = np.arctan2(yy, xx)
    
    if use_exact_H is False:
        
        # Computing exact scattered electric field.
        E = generate_scattered_E_field(R, theta, omega)
        E_x = E[0,:,:]
        E_y = E[1,:,:]
        E_z = E[2,:,:]
        
        # using finite differences to compute curl(E) and by extension H.
        dist = 5/N_samples
        curlE = np.asarray([np.gradient(E_z,dist, axis=1, edge_order=2) , -np.gradient(E_z, dist, axis=0, edge_order=2) , np.zeros(R.shape)])
        H = (-1 / (1j*omega)) * curlE
        
        # Since we've taken a numerical gradient and some of the xx,yy region is inside the cylinder, we recheck for the nan values
        # corresponding to inside the cylinder (and a ring of one pixel outside due to gradient).
        mask = np.ones(R.shape)
        for xind in range(R.shape[0]):
            for yind in range(R.shape[1]):
                if np.isnan(curlE[0, xind,yind]):
                    mask[xind, yind] = np.nan
                elif np.isnan(curlE[1, xind,yind]):
                    mask[xind, yind] = np.nan
        
        # Exact 2d scattered H field.
        Exact = np.asarray([H[0,:,:], H[1,:,:]]) * mask

    else:
        
        # Computing area of xx and yy covered by cylinder.
        mask = np.ones(R.shape)
        for yind in range(R.shape[1]):
            for xind in range(R.shape[0]):
                if R[xind, yind] < 1:
                    mask[xind, yind] = np.nan
        
        
        #zz = np.zeros(xx.shape)
        E0 = 1
        tot_sum_rad = 0
        tot_sum_ang = 0
        for n in range(0,30):
            
            if n == 0:
                epsilon_n = 1
            else:
                epsilon_n = 2
            
            beta = omega / 3e8 # Converting omega to freespace wavenumber.
            mu = 4*np.pi * 1e-7 # Using permeability of freespace.
            
            # Derivative of Hankel_1 following balanis book.
            # Hankel fuction of form Y_n(ax) with derivative H'_n(ax) = aH_(n-1)(ax) - (n/x)H_n(ax). In this case a=1 and x=omega*R
            #hank_deriv = hankel1(n-1, omega*R) - (n/(omega*R))*hankel1(n, omega*R)
            hank_deriv = 0.5* (hankel1(n-1, beta*R) - hankel1(n+1, beta*R))
        
            jv_over_h2 = besselj(n, beta*1)/hankel1(n, beta*1)
            phasor = np.exp(1j*n*theta)
            #phasor = np.cos(n*theta) + 1j*np.sin(theta*n)
            phasor = epsilon_n * np.cos(n*theta)
            #phasor_deriv = n * 1j * phasor
            phasor_deriv = n * epsilon_n * -np.sin(theta * n)
            
            # Balanis book provides H in polar coords. Will convert to cartesian
            tot_sum_rad += (-1j)**-n * jv_over_h2 * hankel1(n, beta*R) * phasor_deriv
            tot_sum_ang += (-1j)**-n * jv_over_h2 * hank_deriv * phasor
        
        H_rad = tot_sum_rad * (E0/(1j*omega*mu)) * (1/R)
        H_ang = tot_sum_ang * -(E0/(1j*omega*mu)) * beta 
         
        H_cart = np.asarray([np.cos(theta)*H_rad + -np.sin(theta)*H_ang, np.sin(theta) * H_rad + np.cos(theta)*H_ang])
        H = H_cart * mask
        Exact = H
        

    if plot is True:
        plt.figure()
        plt.title('H_1 Field real')
        plt.imshow(H[0,:,:].real, cmap='jet', extent=[-box_size/2, box_size/2, -box_size/2, box_size/2])
        plt.colorbar()
        plt.xlabel('x')
        plt.xlabel('y')

        plt.figure()
        plt.title('H 1 Field imag')
        plt.imshow(H[0,:,:].imag, cmap='jet', extent=[-box_size/2, box_size/2, -box_size/2, box_size/2])
        plt.colorbar()
        plt.xlabel('x')
        plt.xlabel('y')

        plt.figure()
        plt.title('H 2 Field real')
        plt.imshow(H[1,:,:].real, cmap='jet', extent=[-box_size/2, box_size/2, -box_size/2, box_size/2])
        plt.colorbar()
        plt.xlabel('x')
        plt.xlabel('y')

        plt.figure()
        plt.title('H 2 Field imag')
        plt.imshow(H[1,:,:].imag, cmap='jet', extent=[-box_size/2, box_size/2, -box_size/2, box_size/2])
        plt.colorbar()
        plt.xlabel('x')
        plt.xlabel('y')
    
    return Exact, mask

def generate_exact_solution_dielectric(N_samples, mu, epsi, box_size=2, omega=2*np.pi, plot=True,):
    
    x_extent = np.linspace(-box_size/2, box_size/2, N_samples)
    y_extent = x_extent
    xx,yy = np.meshgrid(x_extent, y_extent, indexing='ij')

    R = np.sqrt(xx**2 + yy**2) # Cylinder is centered at (0,0)
    theta = np.arctan2(yy, xx)

    Z = np.sqrt(mu/ epsi) # intrinsic impedence inside dielectric
    Z_0 = 76.730313668 # impedence of free space.
    
    beta = omega * np.sqrt(mu * epsi) # wavenumber inside dielectric
    beta_0 = omega / 3e8 # Free space wavenumber

def generate_FEM_mesh(box_size=2, PML_size=7, h=0.25, hpml=0.15, use_PEC=True):
    """Generates 2d mesh box of size box_size + PML_size with a unit radius hole in center.
    boundaries are labeled as 'innerbnd' for the inner boundary with the PML, 'outerbnd' for the PML outer boundary, and
    'scabnd' for the boundary around the scatterer.
    The PML is given the 'pmlregion' material name.

    Args:
        box_size (float, optional): Defaults to 5. size of the square domain to consider.
        PML_size (int, optional): size of additional PML region. Defaults to 5.
        h (float, optional): max mesh element size. Defaults to 0.25.

    Returns:
        mesh (NgSolve FEM mesh): NGSolve FEM 2d mesh.
    """
    
    # if use_PEC is F0alse:
        # PML_size = 
    
    print('Generating Mesh')
    
    inner_rect=WorkPlane().RectangleC(box_size,box_size).Face()
    scatterer = WorkPlane().Circle(0,0,1).Face()

    inner_rect.edges.name = 'innerbnd'
    scatterer.edges.name = 'scabnd'

    inner = inner_rect - scatterer

    wp2=WorkPlane().RectangleC(box_size+PML_size,box_size+PML_size).RectangleC(box_size, box_size).Reverse()
    outer = wp2.Face()
    
    inner.maxh = h
    outer.maxh = hpml

    outer.edges.name = 'outerbnd'
    inner.faces.name ='inner'
    outer.faces.name = 'pmlregion'
    if use_PEC is False:
        scatterer.faces.name = 'scatterer'
        outer.edges.name = 'outerbnd'
        inner.faces.name ='inner'
        outer.faces.name = 'pmlregion'

    if use_PEC is True:
        geo = OCCGeometry(Glue([inner, outer]), dim=2)
        outer.edges.name = 'outerbnd'
        inner.faces.name ='inner'
        outer.faces.name = 'pmlregion'
    else:
        geo = OCCGeometry(Glue([scatterer, inner, outer]), dim=2)

    ngmesh = Mesh(geo.GenerateMesh())
    
    # ngmesh.BoundaryLayers
    
    ngmesh.Curve(5)
    # Draw(ngmesh)
    return ngmesh

def define_PML(d, omega, t, p, updated_PML=True):
    """Define PML functions.

    Args:
        d (float): box_size
    Returns:
        dz_x, dz_y (Ngsolve CF): NGsolve coefficient functions for PML derivatives.
    """
    def absval(x):
        return sqrt(x**2)
    
    def multilayer_pml(x, a, b,  d):
        # if (absval(x) - a <= d/2) and (absval(x) >= d/2): # in region a
        #     return (absval(x) - a) / (b - a)
        # elif (absval(x) - a > d/2) and (absval(x) >= d/2): # in region b
        #     return 1
        # else:
        #     return 1
        
        return IfPos(absval(x) - a - (d/2),(absval(x) - a - d) / (b - a) , 1)
        # return IfPos(absval(x) - d/2,(absval(x) - d), (absval(x) - d))
        
    #t = 0.5
    power = int(np.round(p))
    if updated_PML is True:
        print('Using New Method')
        # z_x = IfPos(absval(x) - d/2, (1j * t * 1/omega * (absval(x) - d)**power) * x, x) # returns z_j if |x| > d/2 else returns x
        # z_y = IfPos(absval(y) - d/2, (1j * t * 1/omega * (absval(y) - d)**power) * y, y) # returns z_j if |y| > d/2 else returns y
        power = 1
        a = 20
        b = 30

        
        # z_x = IfPos(absval(x) - d/2, (1j * t * multilayer_pml(x, a, b, d)**power) * x, x) # returns z_j if |x| > d/2 else returns x
        # z_y = IfPos(absval(y) - d/2, (1j * t * multilayer_pml(y, a, b, d)**power) * y, y) # returns z_j if |x| > d/2 else returns x

        dzx = IfPos(absval(x) - d/2, (1 + (1+1j) * t * multilayer_pml(x, a, b, d)), 1) # returns dz_j if |x| > d/2 else returns x
        dzy = IfPos(absval(y) - d/2, (1 + (1+1j) * t * multilayer_pml(y, a, b, d)), 1) # returns dz_j if |y| > d/2 else returns y
    
        # Draw(dzx, generate_FEM_mesh())
        return dzx, dzy
        
        
    
    elif updated_PML is False:
        print('Using Old Method')
        z_x = IfPos(absval(x) - d/2, (1j * t *(absval(x) - d)**power) * x, x) # returns z_j if |x| > d/2 else returns x
        z_y = IfPos(absval(y) - d/2, (1j * t *(absval(y) - d)**power) * y, y) # returns z_j if |y| > d/2 else returns y
        
    else:
        print('Failure')
    
    dzx = z_x.Diff(x)
    dzy = z_y.Diff(y)
    
    # Draw(dzx, generate_FEM_mesh())
    return dzx, dzy

def FEM_scattering(ngmesh, omega, t, p, box_size=5, order=4, updated_PML=False, use_PEC=True):
    """

    Args:
        mesh (NgSolve FEM mesh): NGSolve FEM 2d mesh
        omega (float): freqency of incident field
        box_size (float, optional): size of box to consider. Defaults to 5.
        order (int, optional): Order of FEM basis functions. Defaults to 4.

    Returns:
        scat (NGSolve grid function): FEM approximation of scattered field.
        Nd (int): number of degrees of freedom.
    """
    
    # Defining Neumann BC for plane wave travelling in dim 1:
    # Computing E_in for frequency omega and converting to H_in as NGsolve coefficient functions.
    
    # mur = ngmesh.MaterialCF({'scatterer': 10}, default=1)
    # epsi = 8.8541878128 * 1e-12
    # mu = 4*3.141592 * mur
    #if use_PEC is True:
    K = CF((omega / 3e8, 0))
    K_magnitude = omega / 3e8 # (2 \pi f) / c
    
    # for eddy current problem
    # K_magnitude = 0
    
    
    
    phasor = exp(1j * ((K[0] * x) + (K[1] * y)))
    ez = 1 * phasor
    E = CF((0, 0, ez))
    h = CF((E[2].Diff(y) - E[1].Diff(z), (E[2].Diff(x) - E[0].Diff(z)), 0)) * (-1/(1j*omega * 4*np.pi*1e-7))
    curlh = CF((0, 0, (h[1].Diff(x) - h[0].Diff(y))))
    ang = atan2(y,x)
    normal = CF((cos(ang), sin(ang), 0))
    n_cross_curlh = Cross(normal, curlh)
    n_cross_curlh_2d = CF((n_cross_curlh[0], n_cross_curlh[1])) # BC n x curl(H_in)
    
    # Defining PML function derivatives.
    dzx, dzy = define_PML(box_size, K_magnitude, t, p, updated_PML=updated_PML)
    dz_tot = dzx * dzy

    
    # Constucting finite element space.
    fes = HCurl(ngmesh, order=order, complex=True)#, dirichlet='outerbnd')
    u = fes.TrialFunction()
    v = fes.TestFunction()
    
    # PML tensors
    f = dzy / dzx
    Lambda = CF((f, 0, 0, 1/f), dims=(2,2))
    
    # Assembling bilinear and linear forms.
    a = BilinearForm(fes, symmetric=True)
    a += (1/(dzy * dzx)) *curl(u)*curl(v)*dx - K_magnitude**2*(Lambda *u) *v *dx

    # J = dzx * dzy
    # A = CF((1/dzy, 0, 0, 1/dzx), dims=(2,2))
    # B = CF((dzx, 0, 0, dzy), dims=(2,2))
    
    # a += (1 * J**-1 * curl(B*u)* J**-1 * curl(B*v))*dx - K_magnitude**2*(u) *v *dx
    
    if use_PEC is True:
        b = LinearForm(fes)
        b += n_cross_curlh_2d * v.Trace() * ds('scabnd')

    else:
        b = LinearForm(fes)
        b += n_cross_curlh_2d * v.Trace() * ds('scabnd')

    
    # c = Preconditioner(a, type="bddc", inverse="sparsecholesky")
    a.Assemble()  
    b.Assemble()
    # c.Update()
      
    # Using direct solve since 2d is quick.
    scat = GridFunction(fes) # FEM approx of scattered H field.
    scat.Set((0,0), BND)
    r = b.vec.CreateVector()
    r = b.vec - a.mat * scat.vec
    
    # inverse = GMResSolver(a.mat, c.mat, tol=1e-6, maxiter=500)
    
    scat.vec.data += a.mat.Inverse(freedofs=fes.FreeDofs()) * r
    # scat.vec.data += inverse * r
    
    # scat.vec.data += a.mat.Inverse(free) * rdev    
    return scat, fes.ndof

def compute_FEM_on_meshgrid(xx, yy, scat, mask, ngmesh, plot=False, box_size=5, use_PEC=True):
    """Generates Numpy array by evaluating FEM at coords defined by meshgrid.

    Args:
        xx (ND_array): x coords to evaluate
        yy (ND array): y coords to evaluate
        scat (Gridfunction): NGsolve fem approximation of scattered field
        mask (ND array): MxM mask containing nan inside cylinder else 1.
        ngmesh (Ngsolve mesh): mesh object used for FEM.
        plot (bool, optional): option to plot FEM approx. Default to False

    Returns:
        scat_numpy_x, scat_numpy_y: ND array containing the evaluated solution in dim 1 and dim 2.
    """
    
    
    scat_numpy_x = np.zeros(xx.ravel().shape, dtype=complex)
    scat_numpy_y = np.zeros(xx.ravel().shape, dtype=complex)

    for indx, xy in enumerate(zip(xx.ravel(), yy.ravel())):
        R = np.sqrt(xy[0]**2 + xy[1]**2)
        if use_PEC is True:
            if R >= 1:
                scat_numpy_x[indx] = scat(ngmesh(xy[0], xy[1]))[0]
                scat_numpy_y[indx] = scat(ngmesh(xy[0], xy[1]))[1]
        else:
            scat_numpy_x[indx] = scat(ngmesh(xy[0], xy[1]))[0]
            scat_numpy_y[indx] = scat(ngmesh(xy[0], xy[1]))[1]
                

    scat_numpy_x = scat_numpy_x.reshape(xx.shape) * mask
    scat_numpy_y = scat_numpy_y.reshape(xx.shape) * mask
    
    if plot is True:
        plt.figure()
        plt.imshow(scat_numpy_x.real, cmap='jet', extent=[-box_size/2,box_size/2,-box_size/2,box_size/2])
        plt.colorbar()
        plt.title('FEM Real dim 1')

        plt.figure()
        plt.imshow(scat_numpy_x.imag, cmap='jet', extent=[-box_size/2,box_size/2,-box_size/2,box_size/2])
        plt.colorbar()
        plt.title('FEM Imag dim 1')

        plt.figure()
        plt.imshow(scat_numpy_y.real, cmap='jet', extent=[-box_size/2,box_size/2,-box_size/2,box_size/2])
        plt.colorbar()
        plt.title('FEM Real dim 2')

        plt.figure()
        plt.imshow(scat_numpy_y.imag, cmap='jet', extent=[-box_size/2,box_size/2,-box_size/2,box_size/2])
        plt.colorbar()
        plt.title('FEM Imag dim 2')
    
    
    return scat_numpy_x, scat_numpy_y

def compare_FEM_exact():
    
    def compute_err(exact, approx):
        return np.linalg.norm((exact - approx)) / np.linalg.norm(exact)
    
    sigma_0 = 3
    print(f'Numpy ang = {np.angle(1 + 1j*sigma_0)}')
    
    om = 1e4
    Exact, mask = generate_exact_solution(200, omega=om, use_exact_H=True, plot=True, box_size=5)
    x_extent = np.linspace(-5/2, 5/2, 200)
    y_extent = x_extent
    xx,yy = np.meshgrid(x_extent, y_extent, indexing='ij')
    for h in [0.2]:
        error = []
        ndof = []
        ngmesh = generate_FEM_mesh(h=h, PML_size=50, hpml=0.2)
        for p in [4]:
            print(f'Solving for p={p}')
            scat, nd = FEM_scattering(ngmesh, om, sigma_0, 1, order=p, box_size=5, use_PEC=True, updated_PML=True)
            scat_pml_numpy_x, scat_pml_numpy_y = compute_FEM_on_meshgrid(xx, yy, scat, mask, ngmesh, plot=True)
            scat_pml_numpy = np.asarray([scat_pml_numpy_x, scat_pml_numpy_y])
            del scat_pml_numpy_x, scat_pml_numpy_y
            scat_pml_numpy_no_nans = scat_pml_numpy[~np.isnan(scat_pml_numpy)]
            del scat_pml_numpy
            exact_no_nans = Exact[~np.isnan(Exact)]
        
            
            err = compute_err((np.asarray(exact_no_nans)), np.asarray(scat_pml_numpy_no_nans))
            ndof += [nd]
            error += [err]
            # del _scat_pml_numpy_no_nans
            
        
        plt.figure(999)
        plt.loglog(ndof, error, label=f'h={h}')
        plt.ylabel('Relative Error')
        plt.xlabel('NDOF')
        plt.legend()
        plt.title(f'$\omega = {om}$')
        print('')
    plt.show()

def compare_diff_omega():
    def compute_err(exact, approx):
        return np.linalg.norm((exact - approx)) / np.linalg.norm(exact)
    
    
    for pml_meth in [True, False]:
        error = []
        ndof = []
        # omega_list = np.asarray([2*np.pi * 0.01 , 2*np.pi * 0.05,  2*np.pi * 0.1, 2*np.pi * 0.5, 2*np.pi]) * 3e8
        omega_list = [1e9]
        for om in omega_list:
            h = 0.25
            p = 5
            Exact, mask = generate_exact_solution(2000, omega=om, use_exact_H=True, plot=False)
            x_extent = np.linspace(-5/2, 5/2, 2000)
            y_extent = x_extent
            xx,yy = np.meshgrid(x_extent, y_extent, indexing='ij')
            ngmesh = generate_FEM_mesh(h=h, PML_size=100, hpml=0.3)
            print(f'Solving for p={p}')
            scat, nd = FEM_scattering(ngmesh, om, 0.5, 1, order=p, updated_PML=pml_meth)
            scat_pml_numpy_x, scat_pml_numpy_y = compute_FEM_on_meshgrid(xx, yy, scat, mask, ngmesh, plot=True)
            scat_pml_numpy = np.asarray([scat_pml_numpy_x, scat_pml_numpy_y])
            scat_pml_numpy_no_nans = scat_pml_numpy[~np.isnan(scat_pml_numpy)]
            exact_no_nans = Exact[~np.isnan(Exact)]

            err = compute_err((np.asarray(exact_no_nans)), np.asarray(scat_pml_numpy_no_nans))
            ndof += [nd]
            error += [err]
            
        plt.figure(999)
        plt.loglog(omega_list, error, label=f'h={h}, p={p}', marker='x')
        plt.ylabel('Relative Error')
        #plt.axvline(1/10, label='wavelength=PML size', linestyle='--')
        plt.xlabel('$\omega$, [rad/s]')
        plt.legend(['$\omega$ weighted', 'Not $\omega$ weigthed'])
        plt.title(f'Updated PML')
        plt.savefig('PML_Comparison.pdf')

    print('')
    plt.show()
  
def compute_error_at_frequency(x):
    
    print(x)
        
    def compute_err(exact, approx):
        return np.linalg.norm((exact - approx)) / np.linalg.norm(exact)
    
    
    om = 2*np.pi
    Exact, mask = generate_exact_solution(100, omega=om, use_exact_H=True, plot=False)
    x_extent = np.linspace(-5/2, 5/2, 100)
    y_extent = x_extent
    xx,yy = np.meshgrid(x_extent, y_extent, indexing='ij')
    ngmesh = generate_FEM_mesh(h=0.25, PML_size=5)

    scat, nd = FEM_scattering(ngmesh, om, x[0], x[1], order=4)
    scat_pml_numpy_x, scat_pml_numpy_y = compute_FEM_on_meshgrid(xx, yy, scat, mask, ngmesh, plot=False)
    scat_pml_numpy = np.asarray([scat_pml_numpy_x, scat_pml_numpy_y])
    scat_pml_numpy_no_nans = scat_pml_numpy[~np.isnan(scat_pml_numpy)]
    exact_no_nans = Exact[~np.isnan(Exact)]

    
    err = compute_err((np.asarray(exact_no_nans)), np.asarray(scat_pml_numpy_no_nans))
    return err


def main():
    
    use_PML = True
    box_size=10
    
    H, mask = generate_exact_solution(200, use_exact_H=True, plot=True, omega=1e9, box_size=box_size)
    # H, mask = generate_exact_solution(200, use_exact_H=False, plot=True)
    ngmesh = generate_FEM_mesh(h=0.25, PML_size=50, box_size=box_size, use_PEC=use_PML)
    scat, nd = FEM_scattering(ngmesh, 1e9, 5, 1, order=4, box_size=box_size, use_PEC=use_PML)
    
    # mask = np.ones((200,200))
    x_extent = np.linspace(-box_size/2,  box_size/2, 200)
    y_extent = x_extent
    xx,yy = np.meshgrid(x_extent, y_extent, indexing='ij')
    
    compute_FEM_on_meshgrid(xx, yy, scat, mask, ngmesh, plot=True, use_PEC=use_PML, box_size=box_size)
    plt.show()


if __name__ == '__main__':
    # main()
    compare_FEM_exact()
    # compare_diff_omega()
    
    # e = compute_error_at_frequency([0.5,2.6])
    # print(e)
    
    
    # cons = 0
    
    # minimum = fmin(compute_error_at_frequency, [0.5,1])
    # print(minimum)