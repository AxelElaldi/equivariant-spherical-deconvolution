#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains utils functions to generate a synthetic DWI dataset.

Contents
--------
    create_grid() : compute spherical coordinate of an equiangular spherical grid.
    get_cartesian() : compute euclidean coordinates from spherical coordinates.
    get_spherical() : compute spherical coordinate from cartesian coordinates.
    get_healpix() : compute euclidean coordinates of an healpix grid.
    create_spherical_edges_from_equiangular() : compute the edges of an equiangular grid.
    create_spherical_edges_from_healpix() : compute the edges of an healpix grid
    save() : save files
    random_diffusion_direction() : draw random properties of tissues and fibers in a voxel.
    load_path() : load path in a directory
    load_data() : load data from list of path
    create_hemisphere() :  compute euclidean coordinates of evenly distributed points on an hemisphere
    sh_matrix() : compute the matrices to go from a signal to its spherical harmonics, and the opposite
    simulate() : simulate a multi-tissue signal

If you find bugs and/or limitations, please email axel DOT elaldi AT nyu DOT edu.
"""

import numpy as np
import math
import healpy as hp
import os
import pickle as pkl
from dipy.core.sphere import disperse_charges, HemiSphere
from dipy.sims.voxel import add_noise
from scipy import special as sci


def create_grid(res):
    """Compute the latitude and the longitude of a spherical grid of resolution res*res

    A spherical grid is a discretization of a sphere. It is defined as follow:
    a point in a sphere can be localized thanks to its longitude (between 0 and 2*pi) and
    its colatitude (between 0 and pi). Thus, we can discretize the longitude and the
    latitude in res*res points. Thus, the discretization of the sphere gives two matrice,
    both of size (res*res). The first matrix is the latitude of the res*res points and the
    second matrix is the longitude of the res*res points.
    Because of the non-unicity of the North and South pole latitude, the minimum and maximum
    latitudes are (pi/(2*res)) and (pi - pi/(2*res)).

    Parameters
    ----------
    res : integer
        Resolution of the grid.

    Returns
    -------
    colats : np.array (res x res, float)
        The colatitude coordinates of the grid points, between (pi/(2*res)) and (pi - pi/(2*res)).
    lons : np.array (res x res, float)
        The longitude coordinates of the grid points, between (0) and (2*pi-2*pi/res).
    """
    uv = np.mgrid[0:res, 0:res].astype(np.float32)
    colats = (np.pi / res) * (uv[0, :, :] + 1/2)
    lons = (2 * math.pi / res) * (uv[1, :, :])
    return colats, lons


def get_cartesian(colats, lons):
    """Compute the cartesian coordinates from a list of spherical coordinates.
    The points must live on the sphere of radius 1.

    Parameters
    ----------
    colats : np.array (float)
        The colatitude coordinates (between 0 and pi)
    lons : np.array (float)
        The longitude coordinates (between 0 and 2*pi)

    Returns
    -------
    vertices : np.array (colats.size x 3, float)
        The euclidean coordinates, [[x1,y1,z1], ..., [xn,yn,zn]]
    """
    x = np.cos(lons) * np.sin(colats)
    y = np.sin(lons) * np.sin(colats)
    z = np.cos(colats)
    vertices = np.ones((colats.size, 3))
    vertices[:, 0], vertices[:, 1], vertices[:, 2] = x.flatten(), y.flatten(), z.flatten()
    return vertices


def get_spherical(vertices):
    """Compute the spherical coordinates from a list of cartesian coordinates.
    The points must live on the sphere of radius 1.

    Parameters
    ----------
    vertices : matrix (n x 3, float)
        The euclidean coordinates of n points, [[x1,y1,z1], ..., [xn,yn,zn]]

    Returns
    -------
    colats : vector (n , float)
        The colatitude coordinates (between 0 and pi)
    lons : vector (n, float)
        The longitude coordinates (between 0 and 2*pi)
    """
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]  # Get their cartesian coordinates
    colats = np.arccos(z)
    lons = np.arctan2(y, x) % (2*np.pi)
    return colats, lons


def get_healpix(res):
    """Compute the cartesian coordinates of the healpix sampling.
    The points must live on the sphere of radius 1.

    Parameters
    ----------
    res : int
        Resolution of the grid.

    Returns
    -------
    vertices : np.array (((res ** 2) * 12) x 3, float)
        The euclidean coordinates [[x1,y1,z1], ..., [xn,yn,zn]]
    """
    x, y, z = hp.pix2vec(res, range((res ** 2) * 12), nest=True)
    vertices = np.ones((x.size, 3))
    vertices[:, 0], vertices[:, 1], vertices[:, 2] = x.flatten(), y.flatten(), z.flatten()
    return vertices


def create_spherical_edges_from_equiangular(res):
    """Compute the N edges from an equiangular grid.

    Parameters
    ----------
    res: int
        The resolution of the equiangular grid.

    Returns
    -------
    edges: np.array (N x 2)
        The edges of the equiangular spherical grid, with N the number of edges.
    """
    indice = np.arange(res * res)
    edges = []
    for u in range(len(indice)):
        if (u + 1) % res != 0:
            edges.append([indice[u], indice[u + 1]])
        else:
            edges.append([indice[u + 1 - res], indice[u]])
        if u >= res:
            edges.append([indice[u - res], indice[u]])
        elif u < (res // 2):
            edges.append([indice[u], indice[u + (res // 2)]])
        if u >= res * (res - 1) and u < res * res - (res // 2):
            edges.append([indice[u], indice[u + (res // 2)]])
    edges = np.array(edges, dtype='uint16')
    return edges


def create_spherical_edges_from_healpix(res):
    """Compute the N edges from an healpix grid.

    Parameters
    ----------
    res: int
        The resolution of the healpix grid.

    Returns
    -------
    edges: np.array (N x 2)
        The edges of the healpix grid, with N the number of edges.
    """
    neighbours = hp.get_all_neighbours(res, range((res ** 2) * 12), nest=True).T
    edges = []
    for i in range(neighbours.shape[0]):
        for j in range(neighbours.shape[1]):
            if i < neighbours[i][j]:
                edges.append([i, neighbours[i][j]])
    edges = np.array(edges, dtype='uint16')
    return edges


def save(index, data_to_save, path_to_save, name_to_save):
    """Save the simulations.

    Parameters
    ----------
    index : int
        Index of the files to save
    data_to_save : list (k x n_save, list)
        The data to save, in a list, where k is the number of file and n_save is the number of voxels
    path_to_save : list (k, str)
        The list of the path to save the data.
    name_to_save : list (k, str)
        The list of the name of the saved data files
    """
    for i in range(len(data_to_save)):
        with open(os.path.join(path_to_save[i], name_to_save[i] + str(index) + '.pkl'), 'wb') as f:
            pkl.dump(data_to_save[i], f)


def random_diffusion_direction(rand_n_tensor=True, max_tensor=1,
                               rand_angles=True, rand_vf_fiber=False, rand_vf_tissue=True,
                               fix_angles=None, fix_vf_fiber=None, fix_vf_tissue=None):
    """Randomly compute the fiber properties in a voxel, i.e. the number of fibers,
    the direction of each fiber, the diffusion fraction for each fiber and tissue.
    These parameters are the ones use in the Multi Tissue model.

    Parameters
    ----------
    rand_n_tensor : bool
        True to have random number of fibers, False to have fix number of fibers
    max_tensor : int
        If rand_n_tensor is True, this is the maximum number of fibers in the voxel. Otherwise,
        this is the fix number of fibers in the voxel.
    rand_angles : bool
        True to have random angles, False to have fix default angles.
    fix_angles: list
        If rand_angle and rand_n_tensor are False, use this default angle.
         Must be a 2D list of shape (max_tensor, 3).
    rand_vf_fiber: bool
        True to have random fiber volume fraction, False to have fix default volume fraction.
    fix_vf_fiber: list
        If rand_vf_fiber and rand_n_tensor are False, use this default fractions.
        Must be a 2D list of shape (n_max_tensor, 3). If None, use an equidistribution.
    rand_vf_tissue: bool
        True to have random tissue volume fraction, False to have fix default volume fraction.
    fix_vf_tissue: list
        If rand_vf_tissue is False, use this default fractions.
        Must be a 1D list of shape 3. If None, use an equidistribution.

    Returns
    -------
    angles : list of tuple (n_tensor * 2, int)
        The spherical coordinates of each fiber in degree.
        Theta between 0 and 90, with 0 the North pole and 90 the equator
        Phi between 0 and 360.
    fractions : list (n_tensor, float)
        Volume fractions of each fiber.
    fraction_tissues: list (3, float)
        Volume fractions of each tissue.
    """
    # Draw number of tissues and tissue volume fraction
    if not rand_vf_tissue and fix_vf_tissue:
        wm, gm, csf = int(bool(fix_vf_tissue[0])), int(bool(fix_vf_tissue[1])), int(bool(fix_vf_tissue[2]))
        fraction_tissues = fix_vf_tissue
    else:
        wm, gm, csf = 0, 0, 0
        while wm + gm + csf == 0:
            wm = np.random.randint(0, 2)
            gm = np.random.randint(0, 2)
            csf = np.random.randint(0, 2)
        if rand_vf_tissue:
            fraction_tissues = [0, 0, 0]
            fraction_tissues[0] = np.random.uniform(0, 1) * wm
            fraction_tissues[1] = np.random.uniform(0, 1) * gm
            fraction_tissues[2] = np.random.uniform(0, 1) * csf
        else:
            fraction_tissues = [wm, gm, csf]
    s_f = sum(fraction_tissues)
    fraction_tissues = [f / s_f for f in fraction_tissues]

    # Draw white matter properties
    if wm == 1:
        # Draw number of cross-fiber
        if rand_n_tensor:
            n_tensor = np.random.randint(1, max_tensor + 1)
        else:
            n_tensor = max_tensor

        # Draw fiber spherical coordinates in degree
        if rand_angles:
            angles = [(np.random.randint(0, 90), np.random.randint(0, 360)) for _ in range(n_tensor)]
        elif np.shape(fix_angles) == (n_tensor, 2) and not rand_n_tensor:
            angles = fix_angles
        else:
            print("Can't draw the fiber spherical coordinates")
            raise NotImplementedError

        # Draw fiber volume fractions
        if rand_vf_fiber:
            fractions = [np.random.uniform(0, 1) for _ in range(n_tensor)]
            s_f = sum(fractions)
            fractions = [f / s_f for f in fractions]
        elif fix_vf_fiber is None:
            fractions = [1 / n_tensor for _ in range(n_tensor)]
        elif np.shape(fix_vf_fiber) == (n_tensor, 3) and not rand_n_tensor:
            s_f = sum(fix_vf_fiber)
            fractions = [f / s_f for f in fix_vf_fiber]
        else:
            print("Can't draw the fiber volume fractions")
            raise NotImplementedError
    else:
        angles = []
        fractions = []

    return angles, fractions, fraction_tissues


def load_path(path, data_prefix):
    """Load the path of the data

    Parameters
    ----------
    path : str
        Root path of the data
    data_prefix : str
        Prefix of the data

    Returns
    -------
    data_path : list (str)
        The list of the paths
    """
    n = len(os.listdir(path))
    data_path = []
    for i in range(n):
        data_path.append(data_prefix + str(i + 1) + '.pkl')
    return data_path


def load_data(path, list_data_path, split=None):
    """Load the data

    Parameters
    ----------
    path : str
        Root path of the data
    list_data_path : list (str)
        The list of the paths of the data
    split: list (int)
        Select specific index

    Returns
    -------
    data : list
        The data
    """
    data = []
    u = 0
    for j in range(len(list_data_path)):
        with open(os.path.join(path, list_data_path[j]), 'rb') as f:
            data_ = pkl.load(f)
        for i in range(len(data_)):
            if split is not None:
                if u in split:
                    data.append(data_[i])
            else:
                data.append(data_[i])
            u += 1
    print('Set size: {0}'.format(len(data)))
    return data


def create_hemisphere(n_pts):
    """Compute the Euclidean coordinates of n_points evenly distributed on an hemisphere

    In a real scenario, the DWI signal is measured along a fix number of "gradients".
    The measure can be seen as a point on a sphere of radius 1 and the gradient is
    the direction from the sphere center to the point. Thus, for each voxel, the signal
    is measured on a sphere of radius 1 and on a fix number of points on that sphere.
    Moreover, the DWI signal is symmetric. we can compute it only for points on an hemisphere.
    This function creates the Euclidean coordinates of the gradients.

    Parameters
    ----------
    n_pts : int
        Number of points on the hemisphere.

    Returns
    -------
    vertices : np.array (n_pts x 3, float)
        The Euclidean coordinates of the n_pts evenly distributed on an hemisphere.
    """
    # Random inclination of the gradients (between 0 and pi)
    theta = np.pi * np.random.rand(n_pts)
    # Random azimuth of the gradients (between 0 and 2*pi)
    phi = 2 * np.pi * np.random.rand(n_pts)
    # Initial hemisphere with the previous random points
    h_initial = HemiSphere(theta=theta, phi=phi)
    # Evenly distribute the points on the hemisphere
    h_updated, potential = disperse_charges(h_initial, 500)
    # Get the cartesian coordinate of the points
    vertices = h_updated.vertices
    return vertices


def sh_matrix(sh_order, vector, with_order=1):
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
    sh_order : int
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
        num_coefficients = int((sh_order + 1) * (sh_order/2 + 1))
    else:
        num_coefficients = sh_order//2 + 1

    b = np.zeros((num_coefficients, num_gradients))
    for id_gradient in range(num_gradients):
        id_column = 0
        for id_order in range(0, sh_order + 1, 2):
            for id_degree in range(-id_order * with_order, id_order * with_order + 1):
                gradients_phi, gradients_theta = gradients[id_gradient]
                y = sci.sph_harm(np.abs(id_degree), id_order, gradients_theta, gradients_phi)
                if id_degree < 0:
                    b[id_column, id_gradient] = np.imag(y) * np.sqrt(2)
                elif id_degree == 0:
                    b[id_column, id_gradient] = np.real(y)
                elif id_degree > 0:
                    b[id_column, id_gradient] = np.real(y) * np.sqrt(2)
                id_column += 1

    b_inv = np.linalg.inv(np.matmul(b, b.transpose()))
    spatial2spectral = np.matmul(b.transpose(), b_inv)
    spectral2spatial = b
    return spatial2spectral, spectral2spatial


def simulate(csf, gm, wm_dense, sh2grad_list, ind_list,
             grid, angles, fractions, fraction_tissue, snr=20):
    """Randomly compute the DWI response for one voxel and one shell with
    the multi tensor model.

    Parameters
    ----------
    csf : np.array (n_grad, float)
        CSF signal for pur compartment, for each gradients.
    gm : np.array (n_grad, float)
        GM signal for pur compartment, for each gradients.
    wm_dense : np.array (n_shell x n_grid, float)
        WM signal for pur compartment, for each shell.
    sh2grad_list : list (n_shell, np.array)
        The SH2S matrix from SH to the gradients of each shell.
    ind_list : np.array (n_shell x n_grad, bool)
        Index of the gradients belonging to a particular shell.
    grid : np.array (N x 3, float)
        The euclidean coordinate of the grid to rotate the WM response function.
    angles : list of tuple (n_fiber, tuple(int, int))
        The spherical coordinate of the fibers in DEGREE
    fractions : list (n_fiber, float)
        The fiber volume fractions.
    fraction_tissue : list (3, float)
        The tissue volume fractions.
    snr : int
        Noise of the model

    Returns
    -------
    signal : np.array (n_pts, float)
        The simulated signal for the n gradient directions.
    """
    # Simulation of the signal for each gradient. Noise can be added with 'snr'
    s_wm = 0
    if fraction_tissue[0] != 0:
        for i, angle in enumerate(angles):
            theta = angle[0] * np.pi / 180
            phi = angle[1] * np.pi / 180
            ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                           [0, 1, 0],
                           [-np.sin(theta), 0, np.cos(theta)]])
            rz = np.array([[np.cos(phi), -np.sin(phi), 0],
                           [np.sin(phi), np.cos(phi), 0],
                           [0, 0, 1]])
            rot = rz.dot(ry)
            dense_grid_rot = rot.dot(grid.T)
            dense2sh_rot, sh2dense_rot = sh_matrix(10, dense_grid_rot.T, with_order=1)
            wm_sh_rot = wm_dense.dot(dense2sh_rot)
            wm_rot = np.zeros(ind_list.shape[1])
            for j, m in enumerate(sh2grad_list):
                if type(m) == str:
                    wm_rot[ind_list[j]] = wm_dense[0][0]
                else:
                    wm_rot[ind_list[j]] = wm_sh_rot[j].dot(sh2grad_list[j])
            s_wm += wm_rot * fractions[i]
    signal = fraction_tissue[0] * s_wm + fraction_tissue[1] * gm + fraction_tissue[2] * csf
    signal = add_noise(signal, snr, wm_dense[0][0], noise_type='rician')
    return signal
