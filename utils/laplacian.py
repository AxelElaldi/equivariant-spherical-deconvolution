import torch
from pygsp.graphs.nngraphs.spherehealpix import SphereHealpix
from scipy.sparse import coo_matrix
import numpy as np
from scipy import sparse
import math


def prepare_laplacian(laplacian):
    """Prepare a graph Laplacian to be fed to a graph convolutional layer.
    Args:
        laplacian :obj:'scipy.sparse.csr.csr_matrix': sparse numpy laplacian
    Returns:
        :obj:`torch.sparse.FloatTensor: Scaled, shifted and sparse torch laplacian
    """

    def estimate_lmax(laplacian, tol=5e-3):
        """Estimate the largest eigenvalue of an operator.
        """
        lmax = sparse.linalg.eigsh(laplacian, k=1, tol=tol, ncv=min(laplacian.shape[0], 10), return_eigenvectors=False)
        lmax = lmax[0]
        lmax *= 1 + 2 * tol  # Be robust to errors.
        return lmax

    def scale_operator(L, lmax, scale=1):
        """Scale the eigenvalues from [0, lmax] to [-scale, scale].
        """
        I = sparse.identity(L.shape[0], format=L.format, dtype=L.dtype)
        L *= 2 * scale / lmax
        L -= I
        return L

    lmax = estimate_lmax(laplacian)
    laplacian = scale_operator(laplacian, lmax)
    laplacian = scipy_csr_to_sparse_tensor(laplacian)
    return laplacian


def scipy_csr_to_sparse_tensor(csr_mat):
    """Convert scipy csr to sparse pytorch tensor.
    Args:
        csr_mat :obj:'scipy.sparse.csr.csr_matrix': The sparse scipy matrix.
    Returns:
        sparse_tensor :obj:`torch.sparse.FloatTensor`: The sparse torch matrix.
    """
    coo = coo_matrix(csr_mat)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    idx = torch.LongTensor(indices)
    vals = torch.FloatTensor(values)
    shape = coo.shape
    sparse_tensor = torch.sparse.FloatTensor(idx, vals, torch.Size(shape))
    sparse_tensor = sparse_tensor.coalesce()
    return sparse_tensor


def healpix_resolution_calculator(nodes):
    """Calculate the resolution of a healpix graph
    for a given number of nodes.
    Args:
        nodes (int): number of nodes in healpix sampling
    Returns:
        int: resolution for the matching healpix graph
    """
    resolution = int(math.sqrt(nodes / 12))
    return resolution
