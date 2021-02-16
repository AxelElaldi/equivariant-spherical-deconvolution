#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file convert the pkl files into a nii file.

If you find bugs and/or limitations, please email axel DOT elaldi AT nyu DOT edu.
"""

import numpy as np
import argparse
import os
import nibabel as nib
from utils import load_path, load_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        default='data',
        help='The root path to the gradients directory (default: data)',
        type=str
    )
    parser.add_argument(
        '--dim',
        action='store_true',
        help='Save the voxel into a 3D volume (default: False)'
    )
    # Parse the arguments
    args = parser.parse_args()
    path = args.path
    threeD = args.dim

    # Check if directory exist
    gradient_path = os.path.join(path, 'data')
    if not os.path.exists(path):
        print("Path doesn't exist: {0}".format(path))
        raise NotADirectoryError
    if not os.path.exists(gradient_path):
        print("Path doesn't exist: {0}".format(gradient_path))
        raise NotADirectoryError

    # Create the data directory
    nii_path = os.path.join(path, 'data_cat')
    if not os.path.exists(nii_path):
        print('Create new directory: {0}'.format(nii_path))
        os.makedirs(nii_path)

    # Load the data
    gradients_list = load_path(gradient_path, 'features')
    gradients = load_data(gradient_path, gradients_list)
    data = np.array(gradients).astype(np.float32)
    if threeD:
        data = data.reshape((data.shape[0]//4, 2, 2, data.shape[1]))
    else:
        data = data.reshape((data.shape[0], 1, 1, data.shape[1]))
    n_data = data.shape
    print("Load data done: {0}".format(n_data))

    # Save the data
    ide = np.eye(4)
    ide[0, 0] = -1
    img = nib.Nifti2Image(data, ide)
    nib.save(img, os.path.join(nii_path, 'features.nii'))
