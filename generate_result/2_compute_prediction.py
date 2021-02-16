#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file extract the peak from the fODF.

If you find bugs and/or limitations, please email axel DOT elaldi AT nyu DOT edu.
"""
import sys
sys.path.append("./generate_data")
import pickle
import numpy as np
import os
import argparse
from utils import load_path, load_data, save
from utils_result import peak_detection_


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result_path',
        default='data',
        help='The result root path (default: data)',
        type=str
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
    parser.add_argument(
        '--threshold',
        default=0.5,
        help='The relative peak threshold for the peak detection (default: 0.5)',
        type=float
    )
    parser.add_argument(
        '--sep_min',
        default=15,
        help='The minimum angle between two predicted fiber (default: 15)',
        type=float
    )
    parser.add_argument(
        '--max_fiber',
        default=3,
        help='The maximum number of fiber (default: 3)',
        type=int
    )
    parser.add_argument(
        '--n_save',
        default=1000,
        help='Number of voxel per saved file (default: 1000)',
        type=int
    )
    # Parse the arguments
    args = parser.parse_args()
    result_path = args.result_path
    path_grid = args.path_grid
    name_grid = args.name_grid
    threshold = args.threshold
    sep_min = args.sep_min
    max_fiber = args.max_fiber
    n_save = args.n_save

    # Check if directory exist
    fodf_path = os.path.join(result_path, 'fodf', 'fodf_{0}'.format(name_grid))
    grid_edges_path = os.path.join(path_grid, 'edges', 'edges.pkl')
    grid_spherical_path = os.path.join(path_grid, 'spherical', 'spherical.pkl')
    if not os.path.exists(fodf_path):
        print("Path doesn't exist: {0}".format(fodf_path))
        raise NotADirectoryError
    if not os.path.exists(grid_edges_path):
        print("Path doesn't exist: {0}".format(grid_edges_path))
        raise NotADirectoryError
    if not os.path.exists(grid_spherical_path):
        print("Path doesn't exist: {0}".format(grid_spherical_path))
        raise NotADirectoryError

    # Create the data directory
    prediction_root = os.path.join(result_path, 'angles_predicted')
    prediction_root_2 = os.path.join(prediction_root, 'fodf_{0}'.format(name_grid))
    prediction_root_3 = os.path.join(prediction_root_2, '{0}_threshold_{1}_min_{2}_fiber'.format(threshold, sep_min, max_fiber))
    prediction_path = os.path.join(prediction_root_3, 'angles')
    path_to_save = [prediction_path]
    if not os.path.exists(prediction_root):
        print('Create new directory: {0}'.format(prediction_root))
        os.makedirs(prediction_root)
    if not os.path.exists(prediction_root_2):
        print('Create new directory: {0}'.format(prediction_root_2))
        os.makedirs(prediction_root_2)
    if not os.path.exists(prediction_root_3):
        print('Create new directory: {0}'.format(prediction_root_3))
        os.makedirs(prediction_root_3)
    if not os.path.exists(prediction_path):
        print('Create new directory: {0}'.format(prediction_path))
        os.makedirs(prediction_path)

    # Load the data
    print('Load fodf from: {0}'.format(fodf_path))
    fodf_l = load_path(fodf_path, 'fodf')
    fodf = np.array(load_data(fodf_path, fodf_l))
    print('fodf loaded: {0}'.format(fodf.shape))

    print('Load grid edges from: {0}'.format(grid_edges_path))
    with open(grid_edges_path, 'rb') as f:
        edges = pickle.load(f)
    print('Edges loaded: {0}'.format(len(edges)))

    print('Load grid spherical coordinates from: {0}'.format(grid_spherical_path))
    with open(grid_spherical_path, 'rb') as f:
        spherical_coordinate = pickle.load(f)

    print('Save result in: {0}'.format(prediction_path))
    n = len(fodf)
    angle_predicted = []
    for i in range(n):
        print(100*i/n, end='\r')
        # The predicted fiber direction
        direction = peak_detection_(fodf[i, 0], edges, spherical_coordinate, threshold, sep_min, max_fiber)
        angle_predicted.append(direction)
        # Save the files every n_save voxel and initialize the output objects
        if (i + 1) % n_save == 0:
            data_to_save = [angle_predicted]
            index = (i + 1) // n_save
            save(index, data_to_save, path_to_save, ['angles_predicted'])
            angle_predicted = []

        # Last save
    if len(angle_predicted) != 0:
        data_to_save = [angle_predicted]
        index = n // n_save + 1
        save(index, data_to_save, path_to_save, ['angles_predicted'])
