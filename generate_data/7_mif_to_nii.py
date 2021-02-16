#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file convert a mif file into a nii file.

If you find bugs and/or limitations, please email axel DOT elaldi AT nyu DOT edu.
"""

import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        help='The path to the mif file (default: None)',
        type=str
    )
    # Parse the arguments
    args = parser.parse_args()
    path = args.path

    # Check if directory exist
    if not os.path.exists(path):
        print("Path doesn't exist: {0}".format(path))
        raise NotADirectoryError
    print('Load mif file from: {0}'.format(path))

    nii_file = path[:-3] + 'nii'
    print('Write nii file to: {0}'.format(nii_file))
    cmd = 'mrconvert {0} {1}'.format(path, nii_file)
    print('Run command: {0}'.format(cmd))
    os.system(cmd)
