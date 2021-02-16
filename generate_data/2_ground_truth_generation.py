#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file generates and saves the ground truth of a new synthetic dataset.

If you find bugs and/or limitations, please email axel DOT elaldi AT nyu DOT edu.
"""

import argparse
import os
import json
from utils import random_diffusion_direction, save
import numpy as np
import pickle as pkl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rand_tensors',
        action='store_true',
        help='Use a random number of fiber per voxel (default: False)'
    )
    parser.add_argument(
        '--max_tensors',
        default=3,
        help='Maximum number of fibers (default: 3)',
        type=int
    )
    parser.add_argument(
        '--rand_angles',
        action='store_true',
        help='Use a random fiber angles(default: False)'
    )
    parser.add_argument(
        '--rand_vf_tissue',
        action='store_true',
        help='Use a random tissue volume fraction (default: False)'
    )
    parser.add_argument(
        '--n_data',
        default=10000,
        help='Number of simulated voxels (default: 10000)',
        type=int
    )
    parser.add_argument(
        '--n_save',
        default=1000,
        help='Number of voxel per saved file (default: 1000)',
        type=int
    )
    parser.add_argument(
        '--start',
        default=1,
        help='Index of the first saved file (default: 1)',
        type=int
    )
    parser.add_argument(
        '--path',
        default='data',
        help='The directory to save the data (default: data)',
        type=str
    )
    parser.add_argument(
        '--rand_vf_fiber',
        action='store_true',
        help='Use a random fiber volume fraction (default: False)'
    )
    # Parse the arguments
    args = parser.parse_args()

    # WM tissue properties
    rand_tensors = args.rand_tensors
    rand_angles = args.rand_angles
    rand_vf_fiber = args.rand_vf_fiber
    rand_vf_tissue = args.rand_vf_tissue
    max_tensors = args.max_tensors

    # Data arguments
    n_data = args.n_data
    n_save = args.n_save
    start = args.start
    path = args.path

    # Initialize the output objects
    angles_gt = []  # The angles
    ff_gt = []  # The fiber volume fractions
    tf_gt = []  # The tissue volume fractions

    # Create the data directory
    print('Save data ground truth to: {0}'.format(path))
    if not os.path.exists(path):
        print('Create new directory: {0}'.format(path))
        os.makedirs(path)

    path_ground_truth = os.path.join(path, 'ground_truth')
    if not os.path.exists(path_ground_truth):
        print('Create new directory: {0}'.format(path_ground_truth))
        os.makedirs(path_ground_truth)

    path_ground_truth_angles = os.path.join(path_ground_truth, 'angles')
    path_ground_truth_fiber_vf = os.path.join(path_ground_truth, 'fiber_fractions')
    path_ground_truth_tissue_vf = os.path.join(path_ground_truth, 'tissue_fractions')
    path_to_save = [path_ground_truth_angles, path_ground_truth_fiber_vf, path_ground_truth_tissue_vf]
    for p in path_to_save:
        if not os.path.exists(p):
            print('Create new directory: {0}'.format(p))
            os.makedirs(p)

    name_to_save = ['angles', 'ff', 'tf']

    # Save parameters
    with open(os.path.join(path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    print('Use random number of tensor: {0}'.format(rand_tensors))
    print('Use random angle directions: {0}'.format(rand_angles))
    print('Use random fiber volume fractions: {0}'.format(rand_vf_fiber))
    print('Use random tissue volume fractions: {0}'.format(rand_vf_tissue))
    print('Maximum number of tensor: {0}'.format(max_tensors))

    # We iterate over the number of voxel we want to simulate.
    for i in range(n_data):
        print(str(i * 100 / n_data) + " %", end='\r')
        # Get the random properties of a random number of WM fibers for that voxel.
        angles, fiber_vf, tissue_vf = random_diffusion_direction(rand_n_tensor=rand_tensors,
                                                                 max_tensor=max_tensors,
                                                                 rand_angles=rand_angles,
                                                                 rand_vf_fiber=rand_vf_fiber,
                                                                 rand_vf_tissue=rand_vf_tissue)
        angles_gt.append(angles)
        ff_gt.append(fiber_vf)
        tf_gt.append(tissue_vf)

        # Save the files every n_save voxel and initialize the output objects
        if (i + 1) % n_save == 0:
            data_to_save = [angles_gt, ff_gt, tf_gt]
            index = (i + 1) // n_save + (start - 1)
            save(index, data_to_save, path_to_save, name_to_save)
            angles_gt = []
            ff_gt = []
            tf_gt = []

    # Last save
    if len(angles_gt) != 0:
        data_to_save = [angles_gt, ff_gt, tf_gt]
        index = n_data // n_save + start
        save(index, data_to_save, path_to_save, name_to_save)

    train = np.sort(np.random.choice(np.arange(n_data), size=int(n_data*0.7), replace=False))
    train_bar = np.array(list((set(np.arange(n_data)) - set(train))))
    val = np.sort(np.random.choice(train_bar, size=int(n_data*0.1), replace=False))
    test = np.sort(np.array(list(set(train_bar) - set(train))))
    split = [train, val, test]
    path_split = os.path.join(path_ground_truth, 'split')
    if not os.path.exists(path_split):
        print('Create new directory: {0}'.format(path_split))
        os.makedirs(path_split)
    pkl.dump(split, open(os.path.join(path_split, 'split_1.pkl'), 'wb'))

