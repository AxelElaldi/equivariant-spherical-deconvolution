#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the code to test a model.

Contents
--------
    main() : Run testing.

If you find bugs and/or limitations, please email axel DOT elaldi AT nyu DOT edu.
"""

import sys
sys.path.append("./generate_result")
sys.path.append("./generate_data")
import os
import pickle
import torch
import argparse
import numpy as np
import math
import nibabel as nib
from utils_dataset import dataset_test, dataset_test_augmented, load_pretrained, extract_rf, load_fodf_input
from utils import load_data, load_path
from model import DeepCSD


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(data_path, epoch,
         shell_list, bvals, bvecs, grid_vec,
         activation, normalization, pooling, filter_start, max_order, interpolation,
         model_path, gm, csf, nii_path,
         split_path, split_nb,
         fodf_in, fodf_path, wm_in, gm_in, csf_in, mrtrix_input, ind_path):
    """
    Test the model

    Parameters
    ----------
    data_path: str
        Root path of the data.
    epoch : int
        Epoch to test
    shell_list: list
        Used shells. These shells should match the shells used to compute the RF initialization.
    bvals: numpy.array
        Bvals file
    bvecs: numpy.array
        Bvecs file
    grid_vec: numpy.array
        Vertices of the grid
    activation: str
        Activation layer
    normalization: str
        Normalization layer
    pooling: str
        Pooling layer
    filter_start: int
        The number of filter in the first layer.
    max_order: int
        Maximum spherical harmonic order
    interpolation: str
        Interpolation layer
    model_path : str
        Path of the model to test
    gm: bool
        If specified, model with gm.
    csf: bool
        If specified, model with csf.
    nii_path: str
        Path to save the fODF
    split_path:
        Name of the dataset for the split
    split_nb:
        Split number
    """
    with torch.no_grad():
        if split_nb:
            with open(os.path.join(split_path, 'split', 'split_' + str(split_nb) + '.pkl'), 'rb') as f:
                split = pickle.load(f)
        else:
            split = None
        # Load path of the files then the data
        gradients_path = load_path(data_path, 'features')
        gradients = np.array(load_data(data_path, gradients_path))
        if fodf_in:
            if ind_path:
                ind_path = pickle.load(open(ind_path, 'rb'))
            print('ind path: ', ind_path)
            fodf = load_fodf_input(fodf_path, wm_in, gm_in, csf_in, ind_path)
            grad_all, distance_all, bvec_all, fodf_feature, normalize_all = dataset_test_augmented(gradients, fodf,
                                                                                                   bvals, bvecs,
                                                                                                   grid_vec, shell_list,
                                                                                                   interpolation, split)
        else:
            grad_all, distance_all, bvec_all, normalize_all = dataset_test(gradients, bvals, bvecs, grid_vec,
                                                                           shell_list, interpolation, split)

        n_grid = len(grid_vec)
        n_side = int(math.sqrt(int(n_grid/12)))
        n_data = len(grad_all[0])

        # Model
        model = DeepCSD(shell_list, max_order, grid_vec, bvec_all, n_side,
                        activation, normalization, pooling, filter_start,
                        csf, gm, interpolation,
                        fodf_in, wm_in, gm_in, csf_in, mrtrix_input)
        # Load a pre train network
        load_state = os.path.join(model_path, 'history', 'epoch_{0}.pth'.format(epoch))
        model = load_pretrained(model, load_state)

        # Load model in GPU
        model = model.to(DEVICE)
        model.eval()

        # Save Response Function
        extract_rf(model_path, model, normalize_all)
        for i in range(len(distance_all)):
            if shell_list[i] != 0:
                if interpolation == 'sh':
                    w = model.int_weight[i]
                else:
                    w = model.interpolation.get_weight(distance_all[i]).cpu().detach().numpy()
                outfile = open('{0}/weight{1}.pkl'.format(nii_path, str(shell_list[i])), 'wb')
                pickle.dump(w, outfile)
                outfile.close()

        # Compute the results
        nb_coef = int((model.max_sh_order + 1) * (model.max_sh_order / 2 + 1))
        fodf_shc_wm_list = np.zeros((n_data, nb_coef))
        if gm:
            fodf_shc_gm_list = np.zeros((n_data, 1))
        if csf:
            fodf_shc_csf_list = np.zeros((n_data, 1))

        batch = 32
        # This is really slow and memory intensive, can be improved:
        print(n_data // batch)
        for i in range(n_data // batch):
            print(str(i * 100 / (n_data // batch)) + " %", end='\r')
            dwi_list = [None] * len(grad_all)
            for j in range(len(grad_all)):
                dwi_list[j] = grad_all[j][i * batch:(i + 1) * batch].to(DEVICE)
            if fodf_in:
                fodf_t = fodf_feature[i * batch:(i + 1) * batch].to(DEVICE)
                _, _, _, _, fodf_shc_wm, fodf_shc_gm, fodf_shc_csf = model(dwi_list, distance_all, fodf_t)
            else:
                _, _, _, _, fodf_shc_wm, fodf_shc_gm, fodf_shc_csf = model(dwi_list, distance_all)
            fodf_shc_wm_list[i * batch:(i + 1) * batch] = fodf_shc_wm.cpu().detach().numpy()
            if gm:
                fodf_shc_gm_list[i * batch:(i + 1) * batch] = fodf_shc_gm.cpu().detach().numpy()
            if csf:
                fodf_shc_csf_list[i * batch:(i + 1) * batch] = fodf_shc_csf.cpu().detach().numpy()

        if n_data % batch != 0:
            print(str(100) + " %", end='\r')
            dwi_list = [None] * len(grad_all)
            for j in range(len(grad_all)):
                dwi_list[j] = grad_all[j][(n_data // batch) * batch:].to(DEVICE)
            if fodf_in:
                fodf_t = fodf_feature[(n_data // batch) * batch:].to(DEVICE)
                _, _, _, fodf_shc_wm, fodf_shc_gm, fodf_shc_csf = model(dwi_list, distance_all, fodf_t)
            else:
                _, _, _, fodf_shc_wm, fodf_shc_gm, fodf_shc_csf = model(dwi_list, distance_all)
            fodf_shc_wm_list[(n_data // batch) * batch:] = fodf_shc_wm.cpu().detach().numpy()
            if gm:
                fodf_shc_gm_list[(n_data // batch) * batch:] = fodf_shc_gm.cpu().detach().numpy()
            if csf:
                fodf_shc_csf_list[(n_data // batch) * batch:] = fodf_shc_csf.cpu().detach().numpy()

        # Save the results
        fodf_shc_wm_nii = np.array(fodf_shc_wm_list).astype(np.float32)
        fodf_shc_wm_nii = fodf_shc_wm_nii.reshape((n_data, 1, 1, nb_coef))
        img = nib.Nifti2Image(fodf_shc_wm_nii, np.eye(4))
        nib.save(img, nii_path + '/fodf.nii')
        if gm:
            fodf_shc_gm_nii = np.array(fodf_shc_gm_list).astype(np.float32)
            fodf_shc_gm_nii = fodf_shc_gm_nii.reshape((n_data, 1, 1, 1))
            img = nib.Nifti2Image(fodf_shc_gm_nii, np.eye(4))
            nib.save(img, nii_path + '/fodf_gm.nii')
        if csf:
            fodf_shc_csf_nii = np.array(fodf_shc_csf_list).astype(np.float32)
            fodf_shc_csf_nii = fodf_shc_csf_nii.reshape((n_data, 1, 1, 1))
            img = nib.Nifti2Image(fodf_shc_csf_nii, np.eye(4))
            nib.save(img, nii_path + '/fodf_csf.nii')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path',
        required=True,
        help='Root path of the data (default: None)',
        type=str
    )
    parser.add_argument(
        '--scheme',
        required=True,
        help='Data scheme name (default: None)',
        type=str
    )
    parser.add_argument(
        '--snr',
        required=True,
        help='Data SNR (default: None)',
        type=str
    )
    parser.add_argument(
        '--epoch',
        required=True,
        help='Epoch (default: None)',
        type=int
    )
    parser.add_argument(
        '--activation',
        default='relu',
        choices=('relu', 'softplus'),
        help='Activation class (default: relu)',
        type=str
    )
    parser.add_argument(
        '--normalization',
        choices=(None, 'batch', 'instance'),
        help='Normalization class (default: None)',
        type=str
    )
    parser.add_argument(
        '--pooling',
        choices=('max', 'mean'),
        help="Pooling class (default: 'max')",
        default='max',
        type=str
    )
    parser.add_argument(
        '--filter_start',
        help='Number of filters for the first convolution (default: 8)',
        default=8,
        type=int
    )
    parser.add_argument(
        '--max_order',
        help='Max spherical harmonic order (default: 20)',
        default=20,
        type=int
    )
    parser.add_argument(
        '--interpolation',
        choices=('sh', 'linear', 'network'),
        default='sh',
        help='What interpolation (default: sh)',
        type=str
    )
    parser.add_argument(
        '--shell_list',
        required=True,
        help='List of shell in the data, sep by coma (default: None)',
        type=str
    )
    parser.add_argument(
        '--model_name',
        required=True,
        help='Name of the tested model (default: None)',
        type=str
    )
    parser.add_argument(
        '--gm',
        help='Use gm rf (default: False)',
        action='store_true'
    )
    parser.add_argument(
        '--csf',
        help='Use csf rf (default: False)',
        action='store_true'
    )
    parser.add_argument(
        '--split_nb',
        help='VAL: Split to compute only val/test. Otherwise, compute everything (default: None)',
        type=int
    )
    parser.add_argument(
        '--fodf_name',
        help='AUGMENTATION: fODF name as model input (default: None)',
        type=str
    )
    parser.add_argument(
        '--wm_in',
        action='store_true',
        help='AUGMENTATION: Use wm as input (default: False)'
    )
    parser.add_argument(
        '--gm_in',
        action='store_true',
        help='AUGMENTATION: Use gm as input (default: False)'
    )
    parser.add_argument(
        '--csf_in',
        action='store_true',
        help='AUGMENTATION: Use csf as input (default: False)'
    )
    parser.add_argument(
        '--mrtrix_input',
        action='store_true',
        help='AUGMENTATION: Use mrtrix output as input (default: False)'
    )
    parser.add_argument(
        '--ind_path',
        help='AUGMENTATION: Mask for the input fODF nii file (default: None)',
        type=str
    )
    args = parser.parse_args()
    root_path = args.root_path
    scheme = args.scheme
    snr = args.snr

    # Train properties
    data_path = os.path.join(root_path, scheme, '{0}_snr'.format(snr), 'gradients', 'data')
    epoch = args.epoch
    shell_list = [int(s) for s in args.shell_list.split(',')]

    # Model architecture properties
    activation = args.activation
    normalization = args.normalization
    pooling = args.pooling
    filter_start = args.filter_start
    max_order = args.max_order
    interpolation = args.interpolation

    # Result path
    result_path = os.path.join(root_path, scheme, '{0}_snr'.format(snr), 'gradients', 'result')

    # Load pre-trained model and response functions
    model_name = args.model_name
    model_path = os.path.join(result_path, model_name)
    gm = args.gm
    csf = args.csf

    # Validation properties
    split_nb = args.split_nb
    split_path = val_path = os.path.join(root_path, 'ground_truth')

    # Input augmentation properties
    fodf_name = args.fodf_name
    fodf_path = ''
    if fodf_name:
        fodf_path = os.path.join(result_path, fodf_name, 'fodf', 'fodf_cat')
    wm_in = args.wm_in
    wm_in_path = os.path.join(fodf_path, 'fodf.nii')
    gm_in = args.gm_in
    gm_in_path = os.path.join(fodf_path, 'fodf_gm.nii')
    csf_in = args.csf_in
    csf_in_path = os.path.join(fodf_path, 'fodf_csf.nii')
    mrtrix_input = args.mrtrix_input
    ind_path = args.ind_path

    if not os.path.isdir(data_path):
        print("Path doesn't exist: {0}".format(data_path))
        raise NotImplementedError

    if bool(fodf_name) and not os.path.isdir(fodf_path):
        print("Path doesn't exist: {0}".format(fodf_path))
        raise NotImplementedError
    if wm_in and not os.path.isfile(wm_in_path):
        print("Path doesn't exist: {0}".format(wm_in_path))
        raise NotImplementedError
    if gm_in and not os.path.isfile(gm_in_path):
        print("Path doesn't exist: {0}".format(gm_in_path))
        raise NotImplementedError
    if csf_in and not os.path.isfile(csf_in_path):
        print("Path doesn't exist: {0}".format(csf_in_path))
        raise NotImplementedError

    # bvecs scheme
    bvec_path = os.path.join(root_path, scheme, 'scheme', 'bvecs.bvecs')
    if not os.path.isfile(bvec_path):
        print("Path doesn't exist: {0}".format(bvec_path))
        raise NotImplementedError
    bvecs = np.loadtxt(bvec_path)
    print('Bvec size: {0}'.format(bvecs.shape))

    # bvals scheme
    bval_path = os.path.join(root_path, scheme, 'scheme', 'bvals.bvals')
    if not os.path.isfile(bval_path):
        print("Path doesn't exist: {0}".format(bval_path))
        raise NotImplementedError
    bvals = np.loadtxt(bval_path)
    print('Bval size: {0}'.format(bvals.shape))

    # Model grid
    grid_path = os.path.join('grid', 'healpix_16', 'gradients', 'gradients.pkl')
    if not os.path.isfile(grid_path):
        print("Path doesn't exist: {0}".format(grid_path))
        raise NotImplementedError
    with open(grid_path, 'rb') as file:
        grid_vec = pickle.load(file)
    print('Grid size: {0}'.format(grid_vec.shape))

    # Create the result directory
    fodf_path = os.path.join(model_path, 'fodf')
    if not os.path.exists(fodf_path):
        print('Create new directory: {0}'.format(fodf_path))
        os.makedirs(fodf_path)
    nii_path = os.path.join(fodf_path, 'fodf_cat')
    if not os.path.exists(nii_path):
        print('Create new directory: {0}'.format(nii_path))
        os.makedirs(nii_path)

    main(data_path, epoch,
         shell_list, bvals, bvecs, grid_vec,
         activation, normalization, pooling, filter_start, max_order, interpolation,
         model_path, gm, csf, nii_path,
         split_path, split_nb,
         bool(fodf_name), fodf_path, wm_in, gm_in, csf_in, mrtrix_input, ind_path)
