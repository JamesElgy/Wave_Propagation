from ngsolve import *
from ngsolve.webgui import Draw
from netgen.occ import *
from netgen.webgui import Draw as Drawgeo
import matplotlib.pyplot as plt
import numpy as np

from scipy.special import jv as besselj
from scipy.special import yv as bessely
from scipy.special import hankel1, hankel2


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
            
            jv_over_h2 = besselj(n, omega*1)/hankel1(n, omega*1)
            zz = zz -(1j)**n * epsilon_n * jv_over_h2 * hankel1(n, omega * R) * np.cos(theta*n) * mask
            
    E = np.asarray([np.zeros(zz.shape), np.zeros(zz.shape), zz])
    return E  

def generate_exact_solution(N_samples, box_size=5, omega=2*np.pi, plot=True):
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
    
    # Computing exact scattered electric field.
    E = generate_scattered_E_field(R, theta, omega)
    E_x = E[0,:,:]
    E_y = E[1,:,:]
    E_z = E[2,:,:]
    
    # using finite differences to compute curl(E) and by extension H.
    dist = 5/N_samples
    curlE = np.asarray([np.gradient(E_z,dist, axis=1) , -np.gradient(E_z, dist, axis=0) , np.zeros(R.shape)])
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

def generate_FEM_mesh(box_size=5, PML_size=5, h=0.25):
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
        
    inner_rect=WorkPlane().RectangleC(box_size,box_size).Face().Extrude(box_size)
    scatterer = WorkPlane().Circle(0,0,1).Face().Extrude(box_size)

    inner_rect.edges.name = 'innerbnd'
    scatterer.edges.name = 'scabnd'

    inner = inner_rect - scatterer

    wp2=WorkPlane().RectangleC(box_size+PML_size,box_size+PML_size).RectangleC(box_size, box_size).Reverse()
    outer = wp2.Face()

    outer.edges.name = 'outerbnd'
    inner.faces.name ='inner'
    outer.faces.name = 'pmlregion'

    geo = OCCGeometry(Glue([inner, outer]), dim=2)
    ngmesh = Mesh(geo.GenerateMesh (maxh=h))
    ngmesh.Curve(5)
    return ngmesh

def define_PML(d):
    """Define PML functions.

    Args:
        d (float): box_size
    Returns:
        dz_x, dz_y (Ngsolve CF): NGsolve coefficient functions for PML derivatives.
    """
    def absval(x):
        return sqrt(x**2)
    z_x = IfPos(absval(x) - d/2, x + (1j * (( absval(x) - d)/0.5)**1) * x, x) # returns z_j if |x|>d/2 else returns x
    z_y = IfPos(absval(y) -d/2, y + (1j * (( absval(y) - d)/0.5)**1) * y, y) # returns z_j if |y|>d/2 else returns y
    
    dzx = z_x.Diff(x)
    dzy = z_y.Diff(y)
    
    return dzx, dzy

def FEM_scattering(ngmesh, omega, box_size=5, order=4):
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
    K = CF((omega, 0))
    phasor = exp(1j * ((K[0] * x) + (K[1] * y)))
    ez = 1 * phasor
    E = CF((0, 0, ez))
    h = CF((E[2].Diff(y) - E[1].Diff(z), (E[2].Diff(x) - E[0].Diff(z)), 0)) * (-1/(1j*omega))
    curlh = CF((0, 0, (h[1].Diff(x) - h[0].Diff(y))))
    ang = atan2(y,x)
    normal = CF((cos(ang), sin(ang), 0))
    n_cross_curlh = Cross(normal, curlh)
    n_cross_curlh_2d = CF((n_cross_curlh[0], n_cross_curlh[1])) # BC n x curl(H_in)
    
    # Defining PML function derivatives.
    dzx, dzy = define_PML(box_size)
    dz_tot = dzx * dzy

    
    # Constucting finite element space.
    fes = HCurl(ngmesh, order=order, complex=True)
    u = fes.TrialFunction()
    v = fes.TestFunction()
    
    # PML tensors
    f = dzy / dzx
    Lambda = CF((f, 0, 0, 1/f), dims=(2,2))
    
    # Assembling bilinear and linear forms.
    a = BilinearForm(fes, symmetric=True)
    a += (1/(dzy * dzx)) *curl(u)*curl(v)*dx - omega**2*(Lambda *u) *v *dx
    a.Assemble()
    
    b = LinearForm(fes)
    b += n_cross_curlh_2d * v.Trace() * ds('scabnd')
    b.Assemble()
    
    # Trying cuda implementation:

    
    
    
    # Using direct solve since 2d is quick.
    scat = GridFunction(fes) # FEM approx of scattered H field.
    r = b.vec.CreateVector()
    r = b.vec - a.mat * scat.vec
    scat.vec.data += a.mat.Inverse(freedofs=fes.FreeDofs()) * r
    
    return scat, fes.ndof

def compute_FEM_on_meshgrid(xx, yy, scat, mask, ngmesh, plot=False):
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
        if R >= 1:
            scat_numpy_x[indx] = scat(ngmesh(xy[0], xy[1]))[0]
            scat_numpy_y[indx] = scat(ngmesh(xy[0], xy[1]))[1]

    scat_numpy_x = scat_numpy_x.reshape(xx.shape) * mask
    scat_numpy_y = scat_numpy_y.reshape(xx.shape) * mask
    
    if plot is True:
        plt.figure()
        plt.imshow(scat_numpy_x.real, cmap='jet')
        plt.colorbar()
        plt.title('FEM Real dim 1')

        plt.figure()
        plt.imshow(scat_numpy_x.imag, cmap='jet')
        plt.colorbar()
        plt.title('FEM Imag dim 1')

        plt.figure()
        plt.imshow(scat_numpy_y.real, cmap='jet')
        plt.colorbar()
        plt.title('FEM Real dim 2')

        plt.figure()
        plt.imshow(scat_numpy_y.imag, cmap='jet')
        plt.colorbar()
        plt.title('FEM Imag dim 2')
    
    
    return scat_numpy_x, scat_numpy_y

def compare_FEM_exact():
    
    def compute_err(exact, approx):
        return np.linalg.norm((exact - approx)) / np.linalg.norm(exact)
    
    Exact, mask = generate_exact_solution(1000)
    x_extent = np.linspace(-5/2, 5/2, 1000)
    y_extent = x_extent
    xx,yy = np.meshgrid(x_extent, y_extent, indexing='ij')
    
    for h in [0.5, 0.25, 0.125, 0.065]:
        error = []
        ndof = []
        ngmesh = generate_FEM_mesh(h=h)
        for p in [0,1,2,3,4,5,6]:
            print(f'Solving for p={p}')
            scat, nd = FEM_scattering(ngmesh, 2*np.pi, order=p)
            scat_pml_numpy_x, scat_pml_numpy_y = compute_FEM_on_meshgrid(xx, yy, scat, mask, ngmesh)
            scat_pml_numpy = np.asarray([scat_pml_numpy_x, scat_pml_numpy_y])
            scat_pml_numpy_no_nans = scat_pml_numpy[~np.isnan(scat_pml_numpy)]
            exact_no_nans = Exact[~np.isnan(Exact)]
        
            
            err = compute_err((np.asarray(exact_no_nans)), np.asarray(scat_pml_numpy_no_nans))
            ndof += [nd]
            error += [err]
        
        plt.figure(999)
        plt.loglog(ndof, error, label=f'h={h}')
        plt.ylabel('Relative Error')
        plt.xlabel('NDOF')
        plt.legend()
        print('')
    plt.show()
    

def main():
    H, mask = generate_exact_solution(50)
    ngmesh = generate_FEM_mesh()
    scat, nd = FEM_scattering(ngmesh, 2*np.pi)
    
    x_extent = np.linspace(-5/2, 5/2, 50)
    y_extent = x_extent
    xx,yy = np.meshgrid(x_extent, y_extent, indexing='ij')
    
    compute_FEM_on_meshgrid(xx, yy, scat, mask, ngmesh, plot=True)
    plt.show()


if __name__ == '__main__':
    compare_FEM_exact()