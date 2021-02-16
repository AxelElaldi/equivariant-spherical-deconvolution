#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file generates and saves the generated signal of a new synthetic dataset.

If you find bugs and/or limitations, please email axel DOT elaldi AT nyu DOT edu.
"""

import argparse
import os
import json
import numpy as np
from utils import load_path, load_data, create_hemisphere, sh_matrix, save, simulate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--snr',
        help='SNR of the simulated signal (default: None)',
        type=int
    )
    parser.add_argument(
        '--n_save',
        default=1000,
        help='Number of voxel per saved file (default: 1000)',
        type=int
    )
    parser.add_argument(
        '--base_path',
        default='data',
        help='The base directory (default: data)',
        type=str
    )
    parser.add_argument(
        '--name_scheme',
        help='The name of the scheme (default: None)',
        type=str
    )
    parser.add_argument(
        '--rf_path',
        help='The path to the response functions (default: None)',
        type=str
    )
    # Parse the arguments
    args = parser.parse_args()
    snr = args.snr
    path = args.base_path
    scheme = args.name_scheme
    n_save = args.n_save
    rf_path = args.rf_path

    # Check if directory exist
    if not os.path.exists(path):
        print("Path doesn't exist: {0}".format(path))
        raise NotADirectoryError

    path_ground_truth = os.path.join(path, 'ground_truth')
    path_ground_truth_angles = os.path.join(path_ground_truth, 'angles')
    path_ground_truth_fiber_vf = os.path.join(path_ground_truth, 'fiber_fractions')
    path_ground_truth_tissue_vf = os.path.join(path_ground_truth, 'tissue_fractions')
    if not os.path.exists(path_ground_truth):
        print("Path doesn't exist: {0}".format(path_ground_truth))
        raise NotADirectoryError
    if not os.path.exists(path_ground_truth_angles):
        print("Path doesn't exist: {0}".format(path_ground_truth_angles))
        raise NotADirectoryError
    if not os.path.exists(path_ground_truth_fiber_vf):
        print("Path doesn't exist: {0}".format(path_ground_truth_fiber_vf))
        raise NotADirectoryError
    if not os.path.exists(path_ground_truth_tissue_vf):
        print("Path doesn't exist: {0}".format(path_ground_truth_tissue_vf))
        raise NotADirectoryError

    # Create the scheme bvals and bvecs directory
    path_scheme_original = os.path.join('scheme_example', scheme)
    path_bvecs_original = os.path.join(path_scheme_original, 'bvecs.bvecs')
    path_bvals_original = os.path.join(path_scheme_original, 'bvals.bvals')
    if not os.path.exists(path_scheme_original):
        print("Path doesn't exist: {0}".format(path_scheme_original))
        raise NotADirectoryError
    if not os.path.exists(path_bvecs_original):
        print("Path doesn't exist: {0}".format(path_bvecs_original))
        raise NotADirectoryError
    if not os.path.exists(path_bvals_original):
        print("Path doesn't exist: {0}".format(path_bvals_original))
        raise NotADirectoryError
    root_scheme = os.path.join(path, scheme)
    if not os.path.exists(root_scheme):
        print('Create new directory: {0}'.format(root_scheme))
        os.makedirs(root_scheme)
    path_scheme = os.path.join(root_scheme, 'scheme')
    if not os.path.exists(path_scheme):
        print('Create new directory: {0}'.format(path_scheme))
        os.makedirs(path_scheme)
    path_bvecs = os.path.join(path_scheme, 'bvecs.bvecs')
    path_bvals = os.path.join(path_scheme, 'bvals.bvals')
    os.system('cp {0} {1}'.format(path_bvecs_original, path_bvecs))
    os.system('cp {0} {1}'.format(path_bvals_original, path_bvals))

    # Create the data directory
    root_gradients = os.path.join(root_scheme, '{0}_snr'.format(snr))
    path_gradients = os.path.join(root_gradients, 'gradients')
    path_data = os.path.join(path_gradients, 'data')
    name_to_save = ['features']
    if not os.path.exists(root_gradients):
        print('Create new directory: {0}'.format(root_gradients))
        os.makedirs(root_gradients)
    if not os.path.exists(path_gradients):
        print('Create new directory: {0}'.format(path_gradients))
        os.makedirs(path_gradients)
    if not os.path.exists(path_data):
        print('Create new directory: {0}'.format(path_data))
        os.makedirs(path_data)

    # Save parameters
    with open(root_gradients + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Load the data
    list_angles = load_path(path_ground_truth_angles, 'angles')
    list_fiber_fractions = load_path(path_ground_truth_fiber_vf, 'ff')
    list_tissue_fractions = load_path(path_ground_truth_tissue_vf, 'tf')

    orientation = load_data(path_ground_truth_angles, list_angles)
    frac_fib = load_data(path_ground_truth_fiber_vf, list_fiber_fractions)
    frac_tis = load_data(path_ground_truth_tissue_vf, list_tissue_fractions)
    n_data = len(orientation)
    print("Load data done: {0}".format(n_data))

    # Load the DWI scheme
    bvecs = np.loadtxt(path_bvecs)
    bvals = np.loadtxt(path_bvals)
    n_points = len(bvals)
    shells = np.unique(bvals)
    if bvecs.shape[0] != n_points:
        print("bvec and bval don't match.")
        raise NotImplementedError
    if bvecs.shape[1] != 3:
        print("bvec doesn't have a good shape.")
        raise NotImplementedError
    print("Load DWI scheme done: {0}".format(n_points))
    print("Shells: {0}".format(shells))

    # Load RF
    csf_sh = np.loadtxt(os.path.join(rf_path, 'csf_response.txt'))
    gm_sh = np.loadtxt(os.path.join(rf_path, 'gm_response.txt'))
    wm_sh = np.loadtxt(os.path.join(rf_path, 'wm_response.txt'))
    print("CSF Rf: {0}".format(csf_sh))
    print("GM Rf: {0}".format(gm_sh))
    print("WM Rf: {0}".format(wm_sh))

    # Create SH matrices
    csf = np.zeros(n_points)
    gm = np.zeros(n_points)
    sh2grad_list = []
    ind_list = np.zeros((len(shells), n_points), dtype=bool)
    dense_grid = create_hemisphere(122)
    print("Grid size to rotate the WM RF: {0}".format(dense_grid.shape))
    dense2sh_iso, sh2dense_iso = sh_matrix(10, dense_grid, with_order=0)
    wm_dense = wm_sh.dot(sh2dense_iso)
    print("S0: {0}".format(wm_dense[0][0]))
    for i, shell in enumerate(shells):
        ind = bvals == shell
        ind_list[i] = ind
        vector = bvecs[ind]
        _, sh2grad_iso = sh_matrix(0, vector, with_order=0)
        if shell != 0:
            _, sh2grad = sh_matrix(10, vector, with_order=1)
            sh2grad_list.append(sh2grad)
        else:
            sh2grad_list.append('b0')
        csf[ind] = csf_sh[i, None].dot(sh2grad_iso)
        gm[ind] = gm_sh[i, None].dot(sh2grad_iso)

    # Initialize the output objects
    simulated_input_gradient = np.ones((min(n_data, n_save), n_points))  # The signal on the gradients

    # We iterate over the number of voxel we want to simulate.
    for i in range(n_data):
        print(str(i * 100 / n_data) + " %", end='\r')
        # Compute the simulated signal for each gradients
        signal = simulate(csf, gm, wm_dense, sh2grad_list, ind_list, dense_grid,
                          orientation[i], frac_fib[i], frac_tis[i],
                          snr=snr)
        # Save the simulated signal
        simulated_input_gradient[i - ((i + 1) // n_save) * n_save] = signal

        # Save the files every n_save voxel and initialize the output objects
        if (i + 1) % n_save == 0:
            data_to_save = [simulated_input_gradient]
            index = (i + 1) // n_save
            save(index, data_to_save, [path_data], name_to_save)
            simulated_input_gradient = np.ones((min(n_data - ((i + 1) // n_save) * n_save, n_save), n_points))

    # Last save
    if simulated_input_gradient.shape[0] != 0:
        data_to_save = [simulated_input_gradient]
        index = n_data // n_save + 1
        save(index, data_to_save, [path_data], name_to_save)
