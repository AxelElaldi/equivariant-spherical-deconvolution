#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file convert a nii file into a mif file.

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
        '--path_bvals_bvecs',
        help='The root path to the bvals and bvecs directory (default: None)',
        type=str
    )
    parser.add_argument(
        '--mask',
        action='store_true',
        help='Transform the max (default: False)'
    )
    # Parse the arguments
    args = parser.parse_args()
    path = args.path
    path_bvals_bvecs = args.path_bvals_bvecs
    mask = args.mask

    if not os.path.exists(path):
        print("Path doesn't exist: {0}".format(path))
        raise NotADirectoryError

    # Check if directory exist
    if mask:
        gradient_nii_path = os.path.join(path, 'mask.nii')
        gradient_mif_path = os.path.join(path, 'mask.mif')
        cmd = "mrconvert {0} {1}".format(gradient_nii_path, gradient_mif_path)
    else:
        gradient_nii_path = os.path.join(path, 'features.nii')
        gradient_mif_path = os.path.join(path, 'features.mif')
        bvecs_path = os.path.join(path_bvals_bvecs, 'bvecs.bvecs')
        bvals_path = os.path.join(path_bvals_bvecs, 'bvals.bvals')
        cmd = "mrconvert {0} {1} -fslgrad {2} {3}".format(gradient_nii_path, gradient_mif_path, bvecs_path, bvals_path)
        if not os.path.exists(bvecs_path):
            print("Path doesn't exist: {0}".format(bvecs_path))
            raise NotADirectoryError
        if not os.path.exists(bvals_path):
            print("Path doesn't exist: {0}".format(bvals_path))
            raise NotADirectoryError
        print('Load bvals file from: {0}'.format(bvals_path))
        print('Load bvecs file from: {0}'.format(bvecs_path))
    if not os.path.exists(gradient_nii_path):
        print("Path doesn't exist: {0}".format(gradient_nii_path))
        raise NotADirectoryError

    print('Load nii file from: {0}'.format(gradient_nii_path))
    print('Write mif file to: {0}'.format(gradient_mif_path))
    print('Run command: {0}'.format(cmd))
    os.system(cmd)
