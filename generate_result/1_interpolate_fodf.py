#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file interpolate the fODF into a dense grid.

If you find bugs and/or limitations, please email axel DOT elaldi AT nyu DOT edu.
"""

import pickle
import os
import argparse
from dipy.io.image import load_nifti
import numpy as np
import torch
from utils_result import compute_D


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        default='data',
        help='The fODF root path (default: data)',
        type=str
    )
    parser.add_argument(
        '--n_save',
        default=1000,
        help='Number of voxel per saved file (default: 1000)',
        type=int
    )
    parser.add_argument(
        '--name_grid',
        help='Name of interpolation grid',
        type=str
    )
    parser.add_argument(
        '--path_grid',
        help='Path of the grid',
        type=str
    )
    # Parse the arguments
    args = parser.parse_args()
    path = args.path
    n_save = args.n_save
    name_grid = args.name_grid
    path_grid = args.path_grid

    # Check if directory exist
    fodf_cat_path = os.path.join(path, 'fodf_cat', 'fodf.nii')
    path_grid_vec = os.path.join(path_grid, 'gradients', 'gradients.pkl')
    if not os.path.exists(fodf_cat_path):
        print("Path doesn't exist: {0}".format(fodf_cat_path))
        raise NotADirectoryError
    if not os.path.exists(path_grid_vec):
        print("Path doesn't exist: {0}".format(path_grid_vec))
        raise NotADirectoryError

    # Create the data directory
    fodf_grid_path = os.path.join(path, 'fodf_{0}'.format(name_grid))
    if not os.path.exists(fodf_grid_path):
        print('Create new directory: {0}'.format(fodf_grid_path))
        os.makedirs(fodf_grid_path)
    print('Save interpolated fodf under: {0}'.format(fodf_grid_path))

    # Load the data
    print('Load fodf from: {0}'.format(fodf_cat_path))
    fodf, affine = load_nifti(fodf_cat_path)
    fodf = fodf.reshape(-1, 1, fodf.shape[-1])
    print('fodf loaded: {0}'.format(fodf.shape))
    print('Affine matrix: {0}'.format(affine))

    print('Load grid vertices from: {0}'.format(path_grid_vec))
    with open(path_grid_vec, 'rb') as f:
        grid = pickle.load(f)
    print('Grid loaded: {0}'.format(grid.shape))

    print('Compute interpolation matrix')
    print('Number of spherical harmonic coefficients: {0}'.format(fodf.shape[-1]))
    order = int(1/2 * (np.sqrt(1 + 8 * fodf.shape[-1]) - 3))
    print('Spherical harmonic order: {0}'.format(order))
    D = compute_D(grid.dot(np.linalg.inv(affine)[:3, :3]), order)
    print('Interpolation matrix: {0}'.format(D.shape))
    D = torch.Tensor(D).to(DEVICE)

    n = len(fodf)
    print('nb created files: {0}'.format(len(fodf) // n_save))
    d = 0
    if (len(fodf) // n_save) != (len(fodf) / n_save):
        d = 1
    for k in range(len(fodf) // n_save + d):
        print(k / (len(fodf) // n_save + d)*100, end='\r')
        signal = fodf[n_save * k:min(n_save * (k + 1), len(fodf))]
        signal = torch.Tensor(signal).to(DEVICE)
        values = signal.matmul(D).detach().cpu().numpy()
        values = values.astype('float64')

        with open('{0}/fodf{1}.pkl'.format(fodf_grid_path, str(k + 1)), 'wb') as f:
            pickle.dump(values, f)

