#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file generates and saves the spherical grid used by the ESD model.

If you find bugs and/or limitations, please email axel DOT elaldi AT nyu DOT edu.
"""

import argparse
import os
import pickle
import json
from utils import create_grid, get_cartesian, create_spherical_edges_from_equiangular, \
    get_healpix, create_spherical_edges_from_healpix, get_spherical


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--grid_res',
        help='Resolution of the grid (N if healpy, res if equiangular)',
        type=int
    )
    parser.add_argument(
        '--grid_name',
        help='The name of the grid (equiangular or healpix)',
        type=str
    )
    parser.add_argument(
        '--path',
        default='grid',
        help='The root directory to save the grid (default: grid)',
        type=str
    )
    # Parse the arguments
    args = parser.parse_args()
    grid_res = args.grid_res
    grid_name = args.grid_name
    root_path = args.path

    # Create the grid directory
    if not os.path.exists(root_path):
        print('Create new directory: {0}'.format(root_path))
        os.makedirs(root_path)
    output_path = root_path+'/{0}_{1}'.format(grid_name, grid_res)
    if not os.path.exists(output_path):
        print('Create new directory: {0}'.format(output_path))
        os.makedirs(output_path)
    output_path_gradients = os.path.join(output_path, 'gradients')
    if not os.path.exists(output_path_gradients):
        print('Create new directory: {0}'.format(output_path_gradients))
        os.makedirs(output_path_gradients)
    output_path_edges = os.path.join(output_path, 'edges')
    if not os.path.exists(output_path_edges):
        print('Create new directory: {0}'.format(output_path_edges))
        os.makedirs(output_path_edges)
    output_path_spherical = os.path.join(output_path, 'spherical')
    if not os.path.exists(output_path_spherical):
        print('Create new directory: {0}'.format(output_path_spherical))
        os.makedirs(output_path_spherical)

    # Save parameters
    with open(os.path.join(output_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Compute the vertices of the grid
    if grid_name == 'equiangular':
        print('Create equiangular grid')
        lats, lons = create_grid(grid_res)
        vertices_grid = get_cartesian(lats, lons)
        edges = create_spherical_edges_from_equiangular(grid_res)
    elif grid_name == 'healpix':
        print('Create healpix grid')
        vertices_grid = get_healpix(grid_res)
        edges = create_spherical_edges_from_healpix(grid_res)
    else:
        raise NotImplementedError("name_grid must be 'healpix' or 'equiangular': {0}".format(grid_name))
    spherical = get_spherical(vertices_grid)
    print('Grid gradients: {0}'.format(vertices_grid.shape))
    print('Grid spherical coordinates: {0} - {1} - {2}'.format(len(spherical), spherical[0].shape, spherical[1].shape))
    print('Grid edges: {0}'.format(len(edges)))

    # Save the result
    gradient_file = os.path.join(output_path_gradients, 'gradients.pkl')
    print('Grid gradients saved in: {0}'.format(gradient_file))
    with open(gradient_file, 'wb') as f:
        pickle.dump(vertices_grid, f)

    spherical_file = os.path.join(output_path_spherical, 'spherical.pkl')
    print('Grid spherical coordinates saved in: {0}'.format(spherical_file))
    with open(spherical_file, 'wb') as f:
        pickle.dump(spherical, f)

    edge_file = os.path.join(output_path_edges, 'edges.pkl')
    print('Grid edges saved in: {0}'.format(edge_file))
    with open(edge_file, 'wb') as f:
        pickle.dump(edges, f)
