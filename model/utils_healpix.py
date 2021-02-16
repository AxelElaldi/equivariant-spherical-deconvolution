#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Healpix laplacian construction functions.

original: https://github.com/deepsphere/deepsphere-cosmo-tf1/blob/ebcf162eaa6e23c1c92dbc84e0908695bb7245d7/deepsphere/utils.py
"""

import numpy as np
import healpy as hp
from scipy import sparse
import torch
from scipy.sparse import coo_matrix


def get_healpix_laplacians(starting_nside, depth):
    """Get the healpix laplacian list for a certain depth.
    Parameters
    ----------
    starting_nside: int
        The resolution of the healpix grid (nside)
    depth: int
        The number of Laplacian  (by increasing nside)
    Returns
    -------
    laps : list
        The list of Laplacian
    """
    laps = []
    for i in range(depth):
        nside = starting_nside//(2**i)
        laplacian = healpix_laplacian(nside=nside, indexes=None)
        laplacian = prepare_laplacian(laplacian)
        laps.append(laplacian)
    return laps[::-1]


def healpix_laplacian(nside=16,
                      nest=True,
                      lap_type='normalized',
                      indexes=None,
                      dtype=np.float32):
    """Build the healpix laplacian.
    Parameters
    ----------
    nside: int
        The resolution of the healpix grid (nside)
    nest: bool
        If True, use the nested healpix grid
    lap_type: str
        Type of laplacian to use (normalized or combinatorial)
    indexes:
        List of the indices to use (can be None to use the grid)
    dtype: type
        Data type of the laplacian matrix

    Returns
    -------
    lap : scipy.sparse
        The laplacian matrix
    """
    w = healpix_weightmatrix(nside=nside, nest=nest, indexes=indexes, dtype=dtype)
    lap = build_laplacian(w, lap_type=lap_type)
    return lap


def healpix_weightmatrix(nside=16, nest=True, indexes=None, dtype=np.float32):
    """Return an unnormalized weight matrix for a graph using the HEALPIX sampling.
    Parameters
    ----------
    nside : int
        The healpix nside parameter, must be a power of 2, less than 2**30.
    nest : bool, optional
        if True, assume NESTED pixel ordering, otherwise, RING pixel ordering
    indexes : list of int, optional
        List of indexes to use. This allows to build the graph from a part of
        the sphere only. If None, the default, the whole sphere is used.
    dtype : data-type, optional
        The desired data type of the weight matrix.
    Returns
    -------
    W : scipy.sparse
        The weight matrix of the healpix graph
    """
    if not nest:
        raise NotImplementedError()

    if indexes is None:
        indexes = range(nside**2 * 12)
    npix = len(indexes)  # Number of pixels.
    if npix >= (max(indexes) + 1):
        # If the user input is not consecutive nodes, we need to use a slower
        # method.
        usefast = True
        indexes = range(npix)
    else:
        usefast = False
        indexes = list(indexes)

    # Get the coordinates.
    x, y, z = hp.pix2vec(nside, indexes, nest=nest)
    coords = np.vstack([x, y, z]).transpose()
    coords = np.asarray(coords, dtype=dtype)
    # Get the 7-8 neighbors.
    neighbors = hp.pixelfunc.get_all_neighbours(nside, indexes, nest=nest)
    # Indices of non-zero values in the adjacency matrix.
    col_index = neighbors.T.reshape((npix * 8))
    row_index = np.repeat(indexes, 8)

    # Remove pixels that are out of our indexes of interest (part of sphere).
    if usefast:
        keep = (col_index < npix)
        # Remove fake neighbors (some pixels have less than 8).
        keep &= (col_index >= 0)
        col_index = col_index[keep]
        row_index = row_index[keep]
    else:
        col_index_set = set(indexes)
        keep = [c in col_index_set for c in col_index]
        inverse_map = [np.nan] * (nside**2 * 12)
        for i, index in enumerate(indexes):
            inverse_map[index] = i
        col_index = [inverse_map[el] for el in col_index[keep]]
        row_index = [inverse_map[el] for el in row_index[keep]]

    # Compute Euclidean distances between neighbors.
    distances = np.sum((coords[row_index] - coords[col_index])**2, axis=1)

    # Compute similarities / edge weights.
    kernel_width = np.mean(distances)
    weights = np.exp(-distances / (2 * kernel_width))

    # Similarity proposed by Renata & Pascal, ICCV 2017.
    # weights = 1 / distances

    # Build the sparse matrix.
    W = sparse.csr_matrix(
        (weights, (row_index, col_index)), shape=(npix, npix), dtype=dtype)
    return W


def build_laplacian(w, lap_type='normalized', dtype=np.float32):
    """"Return the laplacian matrix for a graph using the HEALPIX sampling.
    Parameters
    ----------
    w : scipy.sparse
        The weight matrix of the healpix graph
    lap_type: str
        Type of laplacian to use (normalized or combinatorial)
    dtype : data-type, optional
        The desired data type of the weight matrix.
    Returns
    -------
    scipy.sparse
        The laplacian matrix of the healpix graph
    """
    d = np.ravel(w.sum(1))
    if lap_type == 'combinatorial':
        D = sparse.diags(d, 0, dtype=dtype)
        return (D - w).tocsc()
    elif lap_type == 'normalized':
        d12 = np.power(d, -0.5)
        D12 = sparse.diags(np.ravel(d12), 0, dtype=dtype).tocsc()
        return sparse.identity(d.shape[0], dtype=dtype) - D12 * w * D12
    else:
        raise ValueError('Unknown Laplacian type {}'.format(lap_type))


def prepare_laplacian(laplacian):
    """"Return the scaled laplacian matrix.
    Parameters
    ----------
    laplacian : scipy.sparse
        The laplacian matrix
    Returns
    -------
    laplacian: torch.sparse.FloatTensor
        The scaled laplacian matrix into a pytorch Tensor object
    """

    def estimate_lmax(lap, tol=5e-3):
        """Estimate the largest eigenvalue of an operator.
        """
        l_max = sparse.linalg.eigsh(lap, k=1, tol=tol, ncv=min(lap.shape[0], 10), return_eigenvectors=False)
        l_max = l_max[0]
        l_max *= 1 + 2 * tol  # Be robust to errors.
        return l_max

    def scale_operator(lap, l_max, scale=1):
        """Scale the eigenvalues from [0, lmax] to [-scale, scale].
        """
        identity = sparse.identity(lap.shape[0], format=lap.format, dtype=lap.dtype)
        lap *= 2 * scale / l_max
        lap -= identity
        return lap

    lmax = estimate_lmax(laplacian)
    laplacian = scale_operator(laplacian, lmax)
    laplacian = scipy_csr_to_sparse_tensor(laplacian)
    return laplacian


def scipy_csr_to_sparse_tensor(csr_mat):
    """Convert scipy csr to sparse pytorch tensor.
    Args:
        csr_mat (csr_matrix): The sparse scipy matrix.
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
