#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file compute the CSD algorithms from mrtrix.

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
        '--path_rf',
        help='The rf path (default: None)',
        type=str
    )
    parser.add_argument(
        '--model',
        help='The model to use, csd, ssmt, msmt (default: None)',
        type=str
    )
    parser.add_argument(
        '--shell_list',
        help='List of shell, sep by coma (default: None)',
        type=str
    )
    parser.add_argument(
        '--mask',
        action='store_true',
        help='Use a mask (default: False)'
    )
    parser.add_argument(
        '--gm',
        action='store_true',
        help='Multi-tissue with grey matter (default: False)'
    )
    parser.add_argument(
        '--csf',
        action='store_true',
        help='Multi-tissue with csf (default: False)'
    )
    # Parse the arguments
    args = parser.parse_args()
    path = args.path
    path_rf = args.path_rf
    name_rf = path_rf.split('/')[-1]
    model = args.model
    mask = args.mask
    shell_list = args.shell_list
    gm = args.gm
    csf = args.csf

    cmd = 'dwi2fod '
    # Check if directory exist
    gradient_mif_path = os.path.join(path, 'data_cat', 'features.mif')
    if not os.path.exists(gradient_mif_path):
        print("Path doesn't exist: {0}".format(gradient_mif_path))
        raise NotADirectoryError
    gradient_mif_path_cmd = '{0} '.format(gradient_mif_path)
    print('Load mif file from: {0}'.format(gradient_mif_path))

    wm_response_path = os.path.join(path_rf, 'wm_response.txt')
    if not os.path.exists(wm_response_path):
        print("Path doesn't exist: {0}".format(wm_response_path))
        raise NotADirectoryError
    print('Load WM rf file from: {0}'.format(wm_response_path))

    if model == 'ssmt' or model == 'msmt':
        if gm:
            gm_response_path = os.path.join(path_rf, 'gm_response.txt')
            if not os.path.exists(gm_response_path):
                print("Path doesn't exist: {0}".format(gm_response_path))
                raise NotADirectoryError
            print('Load GM rf file from: {0}'.format(gm_response_path))
        if csf:
            csf_response_path = os.path.join(path_rf, 'csf_response.txt')
            if not os.path.exists(csf_response_path):
                print("Path doesn't exist: {0}".format(csf_response_path))
                raise NotADirectoryError
            print('Load CSF rf file from: {0}'.format(csf_response_path))
        if model == 'msmt':
            cmd += 'msmt_csd '
        else:
            cmd = 'ss3t_csd_beta1 '
    else:
        cmd += 'csd '

    # Create the data directory
    root_result = os.path.join(path, 'result')
    path_model = os.path.join(root_result, '{0}_{1}_{2}'.format(model, name_rf, shell_list))
    path_model_rf = os.path.join(path_model, 'rf')
    path_model_fodf = os.path.join(path_model, 'fodf')
    path_model_fodf_cat = os.path.join(path_model_fodf, 'fodf_cat')
    if not os.path.exists(root_result):
        print('Create new directory: {0}'.format(root_result))
        os.makedirs(root_result)
    if not os.path.exists(path_model):
        print('Create new directory: {0}'.format(path_model))
        os.makedirs(path_model)
    if not os.path.exists(path_model_rf):
        print('Create new directory: {0}'.format(path_model_rf))
        os.makedirs(path_model_rf)
    if not os.path.exists(path_model_fodf):
        print('Create new directory: {0}'.format(path_model_fodf))
        os.makedirs(path_model_fodf)
    if not os.path.exists(path_model_fodf_cat):
        print('Create new directory: {0}'.format(path_model_fodf_cat))
        os.makedirs(path_model_fodf_cat)

    fodf_mif_path_wm = os.path.join(path_model_fodf_cat, 'fodf.mif')
    print('Write wm fodf mif file to: {0}'.format(fodf_mif_path_wm))
    wm_cmd = '{0} {1}'.format(wm_response_path, fodf_mif_path_wm)
    cmd += gradient_mif_path_cmd + wm_cmd
    if model == 'ssmt' or model == 'msmt':
        if gm:
            fodf_mif_path_gm = os.path.join(path_model_fodf_cat, 'fodf_gm.mif')
            print('Write gm fodf mif file to: {0}'.format(fodf_mif_path_gm))
            gm_cmd = ' {0} {1}'.format(gm_response_path, fodf_mif_path_gm)
            cmd += gm_cmd
        if csf:
            fodf_mif_path_csf = os.path.join(path_model_fodf_cat, 'fodf_csf.mif')
            print('Write csf fodf mif file to:{0}'.format(fodf_mif_path_csf))
            csf_cmd = ' {0} {1}'.format(csf_response_path, fodf_mif_path_csf)
            cmd += csf_cmd
    if mask:
        mask = os.path.join(path, 'data_cat', 'mask.mif')
        if not os.path.exists(mask):
            print("Path doesn't exist: {0}".format(mask))
            raise NotADirectoryError
        cmd += ' -mask {0}'.format(mask)
    if shell_list is not None:
        cmd += ' -shells {0}'.format(shell_list)
    print('Run command: {0}'.format(cmd))
    os.system(cmd)

    wm_copy = os.path.join(path_model_rf, 'wm_response.txt')
    print('Write wm fodf rf file to: {0}'.format(wm_copy))
    os.system('cp {0} {1}'.format(wm_response_path, wm_copy))
    if model == 'ssmt' or model == 'msmt':
        if gm:
            gm_copy = os.path.join(path_model_rf, 'gm_response.txt')
            print('Write gm fodf rf file to: {0}'.format(gm_copy))
            os.system('cp {0} {1}'.format(gm_response_path, gm_copy))
        if csf:
            csf_copy = os.path.join(path_model_rf, 'csf_response.txt')
            print('Write csf fodf rf file to: {0}'.format(csf_copy))
            os.system('cp {0} {1}'.format(csf_response_path, csf_copy))
