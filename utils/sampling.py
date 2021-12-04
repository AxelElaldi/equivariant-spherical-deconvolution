import numpy as np
from scipy import special as sci
import math
from .laplacian import prepare_laplacian
from pygsp.graphs.nngraphs.spherehealpix import SphereHealpix
from .pooling import Healpix

class HealpixSampling:
    """Graph Spherical sampling class.
    """
    def __init__(self, n_side, depth, sh_degree=None, pooling_mode='average'):
        """Initialize the sampling class.
        Args:
            n_side (int): Healpix resolution
            depth (int): Depth of the encoder
            sh_degree (int, optional): Spherical harmonic degree of the sampling
            pooling_mode (str, optional): specify the mode for pooling/unpooling.
                                            Can be max or average. Defaults to 'average'.
        """
        assert math.log(n_side, 2).is_integer()
        assert n_side / (2**(depth-1)) >= 1

        G = SphereHealpix(n_side, nest=True, k=8) # Highest resolution sampling
        self.sampling = Sampling(G.coords, sh_degree)
        print(self.sampling.S2SH.shape[1], (sh_degree+1)*(sh_degree//2+1))
        assert self.sampling.S2SH.shape[1] == (sh_degree+1)*(sh_degree//2+1)
        
        self.laps = self.get_healpix_laplacians(n_side, depth, laplacian_type="normalized", neighbor=8)
        self.pooling = Healpix(mode=pooling_mode)
    
    def get_healpix_laplacians(self, starting_nside, depth, laplacian_type, neighbor=8):
        """Get the healpix laplacian list for a certain depth.
        Args:
            starting_nside (int): initial healpix grid resolution.
            depth (int): the depth of the UNet.
            laplacian_type ["combinatorial", "normalized"]: the type of the laplacian.
        Returns:
            laps (list): increasing list of laplacians from smallest to largest resolution
        """
        laps = []
        for i in range(depth):
            n_side = starting_nside//(2**i) # Get resolution of the grid at depth i
            G = SphereHealpix(n_side, nest=True, k=neighbor) # Construct Healpix Graph at resolution n_side
            G.compute_laplacian(laplacian_type) # Compute Healpix laplacian
            laplacian = prepare_laplacian(G.L) # Get Healpix laplacian
            laps.append(laplacian)
        return laps[::-1]


class ShellSampling:
    """Shell Spherical sampling class.
    """
    def __init__(self, vectors_path, shell_path, sh_degree=None, max_sh_degree=None):
        """Initialize the sampling class.
        Args:
            vectors_path (str): Path of the shell sampling vectors (bvecs)
            shell_path (str): Path of the shell sampling shells (bvals)
            sh_degree (int, optional): Spherical harmonic degree of the sampling
            max_sh_degree (int, optional): Max Spherical harmonic degree of the sampling if sh_degree is None
        """
        # Load sampling
        vectors = np.loadtxt(vectors_path)
        shell = np.loadtxt(shell_path)
        if vectors.shape[0] == 3:
            vectors = vectors.T
        
        assert shell.shape[0] == vectors.shape[0]
        assert vectors.shape[1] == 3
        vectors[:, 0] = -vectors[:, 0] # NIFTI FILE HAS STRIDE -1,2,3,4

        self.vectors = vectors # V x 3
        self.shell = shell # V

        # Separate shells
        shell_values, shell_inverse, shell_counts = np.unique(self.shell, return_inverse=True, return_counts=True)
        self.shell_values = shell_values # S
        self.shell_inverse = shell_inverse # V
        self.shell_counts = shell_counts # S

        # Save multi-shell sampling
        self.sampling = []
        for s in self.shell_values:
            vertice = self.vectors[self.shell == s] # V_s x 3
            s_sampling = Sampling(vertice, sh_degree, max_sh_degree, s==0)
            self.sampling.append(s_sampling)


class Sampling:
    """Spherical sampling class.
    """

    def __init__(self, vectors, sh_degree=None, max_sh_degree=None, constant=False):
        """Initialize symmetric sampling class.
        Args:
            vectors (np.array): [V x 3] Sampling position on the unit sphere (bvecs)
            sh_degree (int, optional): Spherical harmonic degree of the sampling
            max_sh_degree (int, optional): Max Spherical harmonic degree of the sampling if sh_degree is None
            constant (bool, optional): In the case of a shell==0
        """
        # Load sampling
        assert vectors.shape[1] == 3
        self.vectors = vectors # V x 3

        # Compute sh_degree
        if sh_degree is None:
            sh_degree = 2*int((np.sqrt(8*vectors.shape[0]-7) - 3) / 4) # We want the number of SHC to be at most the number of vectors
            if not max_sh_degree is None:
                sh_degree = min(sh_degree, max_sh_degree)
        if constant:
            self.S2SH = np.ones((vectors.shape[0], 1)) * math.sqrt(4*math.pi) # V x 1
            self.SH2S = np.zeros(((sh_degree+1)*(sh_degree//2+1), vectors.shape[0])) # (sh_degree+1)(sh_degree//2+1) x V 
            self.SH2S[0] = 1 / math.sqrt(4*math.pi)
        else:
            # Compute SH matrices
            _, self.SH2S = self.sh_matrix(sh_degree, vectors, with_order=1) # (sh_degree+1)(sh_degree//2+1) x V 
            
            # We can't recover more SHC than the number of vertices:
            sh_degree_s2sh = 2*int((np.sqrt(8*vectors.shape[0]-7) - 3) / 4)
            sh_degree_s2sh = min(sh_degree_s2sh, sh_degree)
            if not max_sh_degree is None:
                sh_degree_s2sh = min(sh_degree_s2sh, max_sh_degree)
            self.S2SH, _ = self.sh_matrix(sh_degree_s2sh, vectors, with_order=1) # V x (sh_degree_s2sh+1)(sh_degree_s2sh//2+1)

    def sh_matrix(self, sh_degree, vectors, with_order):
        return _sh_matrix(sh_degree, vectors, with_order)


def _sh_matrix(sh_degree, vector, with_order=1):
    """
    Create the matrices to transform the signal into and from the SH coefficients.

    A spherical signal S can be expressed in the SH basis:
    S(theta, phi) = SUM c_{i,j} Y_{i,j}(theta, phi)
    where theta, phi are the spherical coordinates of a point
    c_{i,j} is the spherical harmonic coefficient of the spherical harmonic Y_{i,j}
    Y_{i,j} is the spherical harmonic of order i and degree j

    We want to find the coefficients c from N known observation on the sphere:
    S = [S(theta_1, phi_1), ... , S(theta_N, phi_N)]

    For this, we use the matrix
    Y = [[Y_{0,0}(theta_1, phi_1)             , ..., Y_{0,0}(theta_N, phi_N)                ],
        ................................................................................... ,
        [Y_{sh_order,sh_order}(theta_1, phi_1), ... , Y_{sh_order,sh_order}(theta_N, phi_N)]]

    And:
    C = [c_{0,0}, ... , c_{sh_order,sh_order}}

    We can express S in the SH basis:
    S = C*Y


    Thus, if we know the signal SH coefficients C, we can find S with:
    S = C*Y --> This code creates the matrix Y

    If we known the signal Y, we can find C with:
    C = S * Y^T * (Y * Y^T)^-1  --> This code creates the matrix Y^T * (Y * Y^T)^-1

    Parameters
    ----------
    sh_degree : int
        Maximum spherical harmonic degree
    vector : np.array (N_grid x 3)
        Vertices of the grid
    with_order : int
        Compute with (1) or without order (0)
    Returns
    -------
    spatial2spectral : np.array (N_grid x N_coef)
        Matrix to go from the spatial signal to the spectral signal
    spectral2spatial : np.array (N_coef x N_grid)
        Matrix to go from the spectral signal to the spatial signal
    """
    if with_order not in [0, 1]:
        raise ValueError('with_order must be 0 or 1, got: {0}'.format(with_order))

    x, y, z = vector[:, 0], vector[:, 1], vector[:, 2]
    colats = np.arccos(z)
    lons = np.arctan2(y, x) % (2 * np.pi)
    grid = (colats, lons)
    gradients = np.array([grid[0].flatten(), grid[1].flatten()]).T

    num_gradients = gradients.shape[0]
    if with_order == 1:
        num_coefficients = int((sh_degree + 1) * (sh_degree/2 + 1))
    else:
        num_coefficients = sh_degree//2 + 1

    b = np.zeros((num_coefficients, num_gradients))
    for id_gradient in range(num_gradients):
        id_column = 0
        for id_degree in range(0, sh_degree + 1, 2):
            for id_order in range(-id_degree * with_order, id_degree * with_order + 1):
                gradients_phi, gradients_theta = gradients[id_gradient]
                y = sci.sph_harm(np.abs(id_order), id_degree, gradients_theta, gradients_phi)
                if id_order < 0:
                    b[id_column, id_gradient] = np.imag(y) * np.sqrt(2)
                elif id_order == 0:
                    b[id_column, id_gradient] = np.real(y)
                elif id_order > 0:
                    b[id_column, id_gradient] = np.real(y) * np.sqrt(2)
                id_column += 1

    b_inv = np.linalg.inv(np.matmul(b, b.transpose()))
    spatial2spectral = np.matmul(b.transpose(), b_inv)
    spectral2spatial = b
    return spatial2spectral, spectral2spatial
