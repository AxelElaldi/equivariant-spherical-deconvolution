#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file compute the response functions from an nii file.

If you find bugs and/or limitations, please email axel DOT elaldi AT nyu DOT edu.
"""

import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        default='data',
        help='The root path to the gradients directory (default: data)',
        type=str
    )
    parser.add_argument(
        '--method_name',
        default='tournier',
        help='The mrtrix method name, tournier or dhollander (default: tournier)',
        type=str
    )
    parser.add_argument(
        '--lmax',
        default=8,
        help='The max order of the Spherical Harmonic coefficients (default: 8)',
        type=int
    )
    parser.add_argument(
        '--no_erode',
        action='store_true',
        help='Erode for dhollander (default: False)'
    )
    parser.add_argument(
        '--shell_list',
        help='List of shell indices in the data, sep by coma (default: None)',
        type=str
    )
    parser.add_argument(
        '--fa',
        help='FA for gm and csf selection (default: None)',
        type=float
    )
    parser.add_argument(
        '--mask',
        help='Mask path (default: None)',
        type=str
    )
    # Parse the arguments
    args = parser.parse_args()
    path = args.path
    method_name = args.method_name
    lmax = args.lmax
    no_erode = args.no_erode
    shell_list = args.shell_list
    fa = args.fa
    mask = args.mask

    # Check if directory exist
    gradient_mif_path = os.path.join(path, 'data_cat', 'features.mif')
    if not os.path.exists(gradient_mif_path):
        print("Path doesn't exist: {0}".format(gradient_mif_path))
        raise NotADirectoryError
    print('Load mif file from: {0}'.format(gradient_mif_path))

    # Create the data directory
    root_rf = os.path.join(path, 'response_functions')
    path_rf = os.path.join(root_rf, '{0}_{1}_{2}'.format(method_name, lmax, shell_list))
    if not os.path.exists(root_rf):
        print('Create new directory: {0}'.format(root_rf))
        os.makedirs(root_rf)
    if not os.path.exists(path_rf):
        print('Create new directory: {0}'.format(root_rf))
        os.makedirs(path_rf)

    wm_response_path = os.path.join(path_rf, 'wm_response.txt')
    print('Write WM response function to: {0}'.format(wm_response_path))

    cmd = 'dwi2response {0} {1} {2}'.format(method_name, gradient_mif_path, wm_response_path)
    if method_name == 'dhollander':
        gm_response_path = os.path.join(path_rf, 'gm_response.txt')
        csfm_response_path = os.path.join(path_rf, 'csf_response.txt')
        print('Write GM response function to: {0}'.format(gm_response_path))
        print('Write CSF response function to: {0}'.format(csfm_response_path))
        cmd += ' {0} {1}'.format(gm_response_path, csfm_response_path)
        if no_erode:
            cmd += ' -erode 0'
    else:
        cmd += ' -lmax {0}'.format(lmax)
    if shell_list is not None:
        cmd += ' -shells {0}'.format(shell_list)
    if fa is not None:
        cmd += ' -fa {0}'.format(fa)
    if mask is not None:
        cmd += ' -mask {0}'.format(mask)
    print('Compute RF with method: {0}'.format(method_name))
    print('Compute RF with max order: {0}'.format(lmax))
    print('Run command: {0}'.format(cmd))
    os.system(cmd)
