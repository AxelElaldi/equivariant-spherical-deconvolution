#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the code to train a model.

Contents
--------
    main() : Run training.

If you find bugs and/or limitations, please email axel DOT elaldi AT nyu DOT edu.
"""

import sys
sys.path.append("./generate_result")
sys.path.append("./generate_data")
import torch
import argparse
import os
import numpy as np
import pickle
import json
import time
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils_result import compute_D, peak_detection_, compute_result
from utils_dataset import dataset, dataset_val, dataset_augmented, dataset_val_augmented, \
    load_pretrained, WeightedLoss, load_fodf_input, gen_distortion_weights
from utils import load_data, load_path
from model import DeepCSD


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(data_path, batch_size, lr, n_epoch,
         shell_list, bvals, bvecs, grid_vec,
         activation, normalization, pooling, filter_start, max_order, interpolation,
         loss_fn_intensity, loss_fn_non_negativity, loss_fn_sparsity, sigma_sparsity,
         intensity_weight, nn_fodf_weight, sparsity_weight,
         load_state, wm_path, gm_path, csf_path,
         val, val_path, threshold, sep_min, max_fiber, split_nb,
         fodf_in, fodf_path, wm_in, gm_in, csf_in, mrtrix_input, ind_path,
         save_path, save_every):
    """
    Train the model

    Parameters
    ----------
    data_path: str
        Root path of the data.
    batch_size: int
        Size of the batch
    lr: float
        The learning rate of the optimizer
    n_epoch : int
        Max number of epoch
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
    loss_fn_intensity : str
        Intensity loss name.
    loss_fn_non_negativity : str
        Non negativity loss name.
    loss_fn_sparsity : str
        Sparsity loss name.
    sigma_sparsity : float
        Hyper parameter for the sparsity loss.
    intensity_weight: float
        Weight of the intensity loss in the loss sum
    nn_fodf_weight: float
        Weight of the wm fODF non negativity loss in the loss sum
    sparsity_weight: float
        Weight of the wm sparsity loss in the loss sum
    load_state : str
        If specified, load saved model weights.
    wm_path: str
        If specified, load white matter rf. Must be a txt file from MRtrix of dimension
        (n_shell, max_order).
    gm_path: str
        If specified, load white matter rf. Must be a txt file from MRtrix of dimension
        (n_shell,).
    csf_path: str
        If specified, load white matter rf. Must be a txt file from MRtrix of dimension
        (n_shell,).
    val: bool
        Use a validation set
    val_path:
        Name of the dataset for the validation
    threshold:
        Threshold for peak detection
    sep_min:
        Min angle between fibers for peak detection
    max_fiber:
        max number of fibers per voxel for peak detection
    split_nb:
        Split number
    save_path: str
        Path to save the model and losses
    save_every: int
        Saving frequency
    """

    multi_tissue = False
    if csf_path or gm_path:
        multi_tissue = True

    # Load path of the files then the data
    print('LOAD INPUT SIGNAL')
    gradients_path = load_path(data_path, 'features')
    gradients = load_data(data_path, gradients_path)
    gradients = np.array(gradients)
    # With validation: need to load ground truth
    if val:
        # Get signal on the peak detection grid
        path_grid_vec = os.path.join('grid', 'equiangular_64', 'gradients', 'gradients.pkl')
        with open(path_grid_vec, 'rb') as f:
            grid_peak_detection = pickle.load(f)
        D = compute_D(grid_peak_detection, max_order)
        D = torch.Tensor(D).to(DEVICE)
        # Get peak detection grid edges
        grid_edges_path = os.path.join('grid', 'equiangular_64', 'edges', 'edges.pkl')
        with open(grid_edges_path, 'rb') as f:
            edges = pickle.load(f)
        # Get peak detection grid spherical coordinates
        grid_spherical_path = os.path.join('grid', 'equiangular_64', 'spherical', 'spherical.pkl')
        with open(grid_spherical_path, 'rb') as f:
            spherical_coordinate = pickle.load(f)
        # Ground truth
        with open(os.path.join(val_path, 'split', 'split_'+str(split_nb)+'.pkl'), 'rb') as f:
            split = pickle.load(f)
        gt_path = os.path.join(val_path, 'angles')
        direction_path = load_path(gt_path, 'angles')
        direction_gt = load_data(gt_path, direction_path)
        if multi_tissue:
            frac_path_gt = os.path.join(val_path, 'tissue_fractions')
            frac_path = load_path(frac_path_gt, 'tf')
            frac_gt = load_data(frac_path_gt, frac_path)
        # Load dataset
        if fodf_in:
            if ind_path:
                ind_path = pickle.load(open(ind_path, 'rb'))
            print('ind path: ', ind_path)
            fodf = load_fodf_input(fodf_path, wm_in, gm_in, csf_in, ind_path)
            train_loader, val_loader,\
            train_dataset, val_dataset, val_ind_loader,\
            distance_all, bvec_all, normalize_all = dataset_val_augmented(fodf, gradients, bvals,
                                                                          bvecs, grid_vec,
                                                                          shell_list, batch_size,
                                                                          interpolation, split)
        else:
            train_loader, val_loader, \
            train_dataset, val_dataset, val_ind_loader, \
            distance_all, bvec_all, normalize_all = dataset_val(gradients, bvals,
                                                                bvecs, grid_vec,
                                                                shell_list, batch_size,
                                                                interpolation, split)
    # Without validation
    else:
        if fodf_in:
            if ind_path:
                ind_path = pickle.load(open(ind_path, 'rb'))
            print('ind path: ', ind_path)
            fodf = load_fodf_input(fodf_path, wm_in, gm_in, csf_in, ind_path)
            train_loader, train_dataset,\
            distance_all, bvec_all, normalize_all = dataset_augmented(fodf, gradients, bvals, bvecs,
                                                                      grid_vec, shell_list, batch_size,
                                                                     interpolation)
        else:
            train_loader, train_dataset, \
            distance_all, bvec_all, normalize_all = dataset(gradients, bvals, bvecs,
                                                            grid_vec, shell_list, batch_size,
                                                            interpolation)

    n_grid = len(grid_vec)
    n_batch = len(train_loader)
    n_side = int(math.sqrt(int(n_grid/12)))

    # Model
    model = DeepCSD(shell_list, max_order, grid_vec, bvec_all, n_side,
                    activation, normalization, pooling, filter_start,
                    bool(csf_path), bool(gm_path), interpolation,
                    fodf_in, wm_in, gm_in, csf_in, mrtrix_input)

    # Load a pre train network
    model = load_pretrained(model, load_state, wm_path, gm_path, csf_path, normalize_all)
    wts = gen_distortion_weights('healpix', n_side)

    # Load model in GPU
    model = model.to(DEVICE)
    wts = wts.to(DEVICE)

    # Loss
    intensity_criterion = WeightedLoss(loss_fn_intensity)
    non_negativity_criterion = WeightedLoss(loss_fn_non_negativity)
    sparsity_criterion = WeightedLoss(loss_fn_sparsity, sigma_sparsity)

    # Optimizer/Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, threshold=0.01, factor=0.1, patience=3, verbose=True)
    save_loss = {}
    save_loss['train'] = {}
    save_loss['val'] = {}

    # Training loop
    for epoch in range(n_epoch):
        # TRAIN
        model.train()

        # Initialize loss to save and plot.
        loss_intensity_ = 0
        loss_sparsity_ = 0
        loss_non_negativity_fodf_ = 0

        # Train on batch.
        for batch, data in enumerate(train_loader):
            # Delete all previous gradients
            optimizer.zero_grad()
            to_print = ''

            # Load the data in the DEVICE
            dwi_signal = data[0].to(DEVICE)
            if fodf_in:
                dwi_list = [None] * (len(data) - 2)
                for i in range(len(data) - 2):
                    dwi_list[i] = data[i + 1].to(DEVICE)
                fodf_t = data[-1].to(DEVICE)

                # Output of the model
                dwi_reconstruction, fodf_wm, fodf_gm, fodf_csf,\
                fodf_shc_wm, fodf_shc_gm, fodf_shc_csf = model(dwi_list, distance_all, fodf_t)
            else:
                dwi_list = [None] * (len(data)-1)
                for i in range(len(data)-1):
                    dwi_list[i] = data[i+1].to(DEVICE)

                # Output of the model
                dwi_reconstruction, fodf_wm, fodf_gm, fodf_csf, \
                fodf_shc_wm, fodf_shc_gm, fodf_shc_csf = model(dwi_list, distance_all)

            ###############################################################################################
            ###############################################################################################
            # Loss
            ###############################################################################################
            ###############################################################################################
            # Intensity loss
            loss_intensity = intensity_criterion(dwi_reconstruction, dwi_signal, wts=None)
            loss_intensity_ += loss_intensity.item()
            loss = intensity_weight * loss_intensity
            to_print += ', Intensity: {0:.10f}'.format(loss_intensity.item())

            ###############################################################################################
            # Sparsity loss
            fodf_sparse_wm = torch.zeros(fodf_wm.shape).to(DEVICE)
            loss_sparsity = sparsity_criterion(fodf_wm, fodf_sparse_wm,  wts=None)  # wts[0])  #
            loss_sparsity_ += loss_sparsity.item()
            loss += sparsity_weight * loss_sparsity
            to_print += ', Sparsity: {0:.10f}'.format(loss_sparsity.item())

            ###############################################################################################
            # Non negativity loss
            # fODF
            fodf_neg = torch.min(fodf_wm, torch.zeros_like(fodf_wm))
            fodf_neg_zeros = torch.zeros(fodf_neg.shape).to(DEVICE)
            loss_non_negativity_fodf = non_negativity_criterion(fodf_neg, fodf_neg_zeros,  wts=None)  # wts[0])  # #
            loss_non_negativity_fodf_ += loss_non_negativity_fodf.item()
            loss += nn_fodf_weight * loss_non_negativity_fodf
            to_print += ', WM fODF NN: {0:.10f}'.format(loss_non_negativity_fodf.item())
            to_print += ', WM volume: {0:.10f}'.format(torch.mean(fodf_shc_wm[:, 0])*np.sqrt(4*np.pi))

            # GM
            if gm_path:
                to_print += ', GM volume: {0:.10f}'.format(torch.mean(fodf_shc_gm[:, 0])*np.sqrt(4*np.pi))

            # CSF
            if csf_path:
                to_print += ', CSF volume: {0:.10f}'.format(torch.mean(fodf_shc_csf[:, 0])*np.sqrt(4*np.pi))

            ###############################################################################################
            # Loss backward
            loss = loss
            loss.backward()
            optimizer.step()

            ###############################################################################################
            # To print loss
            to_print = 'Epoch [{0}/{1}], Iter [{2}/{3}]: Loss: {4:.10f}'.format(epoch + 1, n_epoch,
                                                                                batch + 1, n_batch,
                                                                                loss.item()) \
                       + to_print
            print(to_print, end="\r")

        ###############################################################################################
        # Save and print mean loss for the epoch
        print("")
        to_print = ''
        loss_ = 0
        # Mean results of the last epoch
        save_loss['train'][epoch] = {}

        save_loss['train'][epoch]['loss_intensity'] = loss_intensity_ / n_batch
        save_loss['train'][epoch]['weight_loss_intensity'] = intensity_weight
        loss_ += intensity_weight * loss_intensity_
        to_print += ', Intensity: {0:.10f}'.format(loss_intensity_ / n_batch)

        save_loss['train'][epoch]['loss_sparsity'] = loss_sparsity_ / n_batch
        save_loss['train'][epoch]['weight_loss_sparsity'] = sparsity_weight
        loss_ += sparsity_weight * loss_sparsity_
        to_print += ', Sparsity: {0:.10f}'.format(loss_sparsity_ / n_batch)

        save_loss['train'][epoch]['loss_non_negativity_fodf'] = loss_non_negativity_fodf_ / n_batch
        save_loss['train'][epoch]['weight_loss_non_negativity_fodf'] = nn_fodf_weight
        loss_ += nn_fodf_weight * loss_non_negativity_fodf_
        to_print += ', WM fODF NN: {0:.10f}'.format(loss_non_negativity_fodf_ / n_batch)

        save_loss['train'][epoch]['loss'] = loss_ / n_batch
        to_print = 'Epoch [{0}/{1}], Train Loss: {2:.10f}'.format(epoch + 1, n_epoch, loss_ / n_batch) + to_print
        print(to_print)

        ###############################################################################################
        # VALIDATION
        if val:
            model.eval()
            n_batch_val = len(val_loader)
            # Initialize loss to save and plot.
            loss_intensity_ = 0
            loss_sparsity_ = 0
            loss_non_negativity_fodf_ = 0
            direction = []
            angles_gt = []
            frac_t_gt = []
            frac_t = []
            with torch.no_grad():
                for batch, (data, ind) in enumerate(zip(val_loader, val_ind_loader)):
                    # Load the data on the DEVICE
                    dwi_signal = data[0].to(DEVICE)
                    if fodf_in:
                        dwi_list = [None] * (len(data) - 2)
                        for i in range(len(data) - 2):
                            dwi_list[i] = data[i + 1].to(DEVICE)
                        fodf_t = data[-1].to(DEVICE)

                        # Output of the model
                        dwi_reconstruction, fodf_wm, fodf_gm, fodf_csf, \
                        fodf_shc_wm, fodf_shc_gm, fodf_shc_csf = model(dwi_list, distance_all, fodf_t)
                    else:
                        dwi_list = [None] * (len(data) - 1)
                        for i in range(len(data) - 1):
                            dwi_list[i] = data[i + 1].to(DEVICE)

                        # Output of the model
                        dwi_reconstruction, fodf_wm, fodf_gm, fodf_csf, \
                        fodf_shc_wm, fodf_shc_gm, fodf_shc_csf = model(dwi_list, distance_all)

                    ###############################################################################################
                    ###############################################################################################
                    # Loss
                    ###############################################################################################
                    ###############################################################################################
                    # Intensity loss
                    loss_intensity = intensity_criterion(dwi_reconstruction, dwi_signal, wts=None)
                    loss_intensity_ += loss_intensity.item()
                    loss = intensity_weight * loss_intensity

                    ###############################################################################################
                    # Sparsity loss
                    fodf_sparse_wm = torch.zeros(fodf_wm.shape).to(DEVICE)
                    loss_sparsity = sparsity_criterion(fodf_wm, fodf_sparse_wm,  wts=None)  # wts[0])  #
                    loss_sparsity_ += loss_sparsity.item()
                    loss += sparsity_weight * loss_sparsity

                    ###############################################################################################
                    # Non negativity loss
                    # fODF
                    fodf_neg = torch.min(fodf_wm, torch.zeros_like(fodf_wm))
                    fodf_neg_zeros = torch.zeros(fodf_neg.shape).to(DEVICE)
                    loss_non_negativity_fodf = non_negativity_criterion(fodf_neg, fodf_neg_zeros,  wts=None)  # wts[0])  #
                    loss_non_negativity_fodf_ += loss_non_negativity_fodf.item()
                    loss += nn_fodf_weight * loss_non_negativity_fodf

                    if multi_tissue:
                        frac_wm = fodf_shc_wm[:, 0].detach().cpu().numpy() * np.sqrt(4*np.pi)
                        T = frac_wm
                        frac_gm, frac_csf = 0, 0
                        if gm_path:
                            frac_gm = fodf_shc_gm[:, 0].detach().cpu().numpy() * np.sqrt(4*np.pi)
                            T += frac_gm
                        if csf_path:
                            frac_csf = fodf_shc_csf[:, 0].detach().cpu().numpy() * np.sqrt(4*np.pi)
                            T += frac_csf
                        frac_wm, frac_gm, frac_csf = frac_wm / T, frac_gm / T, frac_csf / T

                    values = fodf_shc_wm.matmul(D).detach().cpu().numpy()
                    values = values.astype('float64')
                    for i in range(len(values)):
                        direction.append(peak_detection_(values[i], edges, spherical_coordinate,
                                                         threshold, sep_min, max_fiber))
                        angles_gt.append(direction_gt[ind[i]])
                        if multi_tissue:
                            frac_t.append([frac_wm[i], frac_gm[i], frac_csf[i]])
                            frac_t_gt.append(frac_gt[ind[i]])

                ###############################################################################################
                # Save and print mean loss for the epoch
                print("")
                to_print = ''
                loss_ = 0
                # Mean results of the last epoch
                save_loss['val'][epoch] = {}

                save_loss['val'][epoch]['loss_intensity'] = loss_intensity_ / n_batch_val
                save_loss['val'][epoch]['weight_loss_intensity'] = intensity_weight
                loss_ += intensity_weight * loss_intensity_
                to_print += ', Intensity: {0:.10f}'.format(loss_intensity_ / n_batch_val)

                save_loss['val'][epoch]['loss_sparsity'] = loss_sparsity_ / n_batch_val
                save_loss['val'][epoch]['weight_loss_sparsity'] = sparsity_weight
                loss_ += sparsity_weight * loss_sparsity_
                to_print += ', Sparsity: {0:.10f}'.format(loss_sparsity_ / n_batch_val)

                save_loss['val'][epoch]['loss_non_negativity_fodf'] = loss_non_negativity_fodf_ / n_batch_val
                save_loss['val'][epoch]['weight_loss_non_negativity_fodf'] = nn_fodf_weight
                loss_ += nn_fodf_weight * loss_non_negativity_fodf_
                to_print += ', WM fODF NN: {0:.10f}'.format(loss_non_negativity_fodf_ / n_batch_val)

                save_loss['val'][epoch]['loss'] = loss_ / n_batch_val
                to_print = 'Epoch [{0}/{1}], Val Loss: {2:.10f}'.format(epoch + 1, n_epoch,
                                                                        loss_ / n_batch_val) + to_print
                print(to_print)
                scheduler.step(loss_ / n_batch_val)
                if multi_tissue:
                    frac_t = np.array(frac_t)
                    frac_t_gt = np.array(frac_t_gt)
                else:
                    frac_t = None
                    frac_t_gt = None
                compute_result(direction, angles_gt, frac_t, frac_t_gt)

                if epoch == 0:
                    min_val_loss = loss_
                    epochs_no_improve = 0
                    n_epochs_stop = 5
                    early_stop = False
                elif loss_ < min_val_loss*0.999:
                    epochs_no_improve = 0
                    min_val_loss = loss_
                else:
                    epochs_no_improve += 1
                if epoch > 5 and epochs_no_improve == n_epochs_stop:
                    print('Early stopping!')
                    early_stop = True
        else:
            scheduler.step(loss_ / n_batch)
            if epoch == 0:
                min_loss = loss_
                epochs_no_improve = 0
                n_epochs_stop = 5
                early_stop = False
            elif loss_ < min_loss * 0.999:
                epochs_no_improve = 0
                min_loss = loss_
            else:
                epochs_no_improve += 1
            if epoch > 5 and epochs_no_improve == n_epochs_stop:
                print('Early stopping!')
                early_stop = True

        ###############################################################################################
        # Save the loss and model
        with open(os.path.join(save_path, 'loss.pkl'), 'wb') as f:
            pickle.dump(save_loss, f)
        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'epoch_{0}.pth'.format(epoch + 1)))
        if early_stop:
            print("Stopped")
            break


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
        '--batch_size',
        default=32,
        help='Batch size (default: 32)',
        type=int
    )
    parser.add_argument(
        '--lr',
        default=1e-2,
        help='Learning rate (default: 1e-2)',
        type=float
    )
    parser.add_argument(
        '--epoch',
        default=100,
        help='Epoch (default: 100)',
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
        '--save_every',
        help='Saving periodicity (default: 2)',
        default=2,
        type=int
    )
    parser.add_argument(
        '--shell_list',
        required=True,
        help='List of shell in the data, sep by coma (default: None)',
        type=str
    )
    parser.add_argument(
        '--loss_intensity',
        choices=('L1', 'L2'),
        default='L2',
        help='Objective function (default: L2)',
        type=str
    )
    parser.add_argument(
        '--intensity_weight',
        default=1.,
        help='Intensity weight (default: 1.)',
        type=float
    )
    parser.add_argument(
        '--loss_sparsity',
        choices=('L1', 'L2', 'cauchy', 'welsch', 'geman'),
        default='cauchy',
        help='Objective function (default: cauchy)',
        type=str
    )
    parser.add_argument(
        '--sigma_sparsity',
        default=1e-5,
        help='Sigma for sparsity (default: 1e-5)',
        type=float
    )
    parser.add_argument(
        '--sparsity_weight',
        default=1.,
        help='Sparsity weight (default: 1.)',
        type=float
    )
    parser.add_argument(
        '--loss_non_negativity',
        choices=('L1', 'L2'),
        default='L2',
        help='Objective function (default: L2)',
        type=str
    )
    parser.add_argument(
        '--nn_fodf_weight',
        default=1.,
        help='Non negativity fODF weight (default: 1.)',
        type=float
    )
    parser.add_argument(
        '--load_state',
        help='Load a saved model (default: None)',
        type=str
    )
    parser.add_argument(
        '--wm_path',
        required=True,
        help='Response function initialization path (white matter - sh coef) (default: None)',
        type=str
    )
    parser.add_argument(
        '--gm_path',
        help='Response function initialization path (grey matter - sh coef) (default: None)',
        type=str
    )
    parser.add_argument(
        '--csf_path',
        help='Response function initialization path (csf - sh coef) (default: None)',
        type=str
    )
    parser.add_argument(
        '--val',
        action='store_true',
        help='VAL: Validation (default: False)'
    )
    parser.add_argument(
        '--threshold',
        default=0.1,
        help='VAL: Threshold for peak detection (default: 0.1)',
        type=float
    )
    parser.add_argument(
        '--sep_min',
        default=15,
        help='VAL: Min angle between predicted fibers (default: 15)',
        type=float
    )
    parser.add_argument(
        '--max_fiber',
        default=3,
        help='VAL: Maximum of predicted fibers per voxel (default: 3)',
        type=int
    )
    parser.add_argument(
        '--split_nb',
        default=1,
        help='VAL: Split for validation (default: 1)',
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
    batch_size = args.batch_size
    lr = args.lr
    n_epoch = args.epoch
    shell_list = [int(s) for s in args.shell_list.split(',')]

    # Model architecture properties
    activation = args.activation
    normalization = args.normalization
    pooling = args.pooling
    filter_start = args.filter_start
    max_order = args.max_order
    interpolation = args.interpolation

    # Saving parameters
    save_path = os.path.join(root_path, scheme, '{0}_snr'.format(snr), 'gradients', 'result')
    save_every = args.save_every

    # Intensity loss
    loss_fn_intensity = args.loss_intensity
    intensity_weight = args.intensity_weight
    # Sparsity loss
    loss_fn_sparsity = args.loss_sparsity
    sigma_sparsity = args.sigma_sparsity
    sparsity_weight = args.sparsity_weight
    # Non-negativity loss
    loss_fn_non_negativity = args.loss_non_negativity
    nn_fodf_weight = args.nn_fodf_weight

    # Load pre-trained model and response functions
    load_state = args.load_state
    wm_path = args.wm_path
    gm_path = args.gm_path
    csf_path = args.csf_path

    # Validation properties
    val = args.val
    val_path = os.path.join(root_path, 'ground_truth')
    threshold = args.threshold
    sep_min = args.sep_min
    max_fiber = args.max_fiber
    split_nb = args.split_nb

    # Input augmentation properties
    fodf_name = args.fodf_name
    fodf_path = ''
    if fodf_name:
        fodf_path = os.path.join(save_path, fodf_name, 'fodf', 'fodf_cat')
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

    if bool(load_state) and not os.path.isfile(load_state):
        print("Path doesn't exist: {0}".format(load_state))
        raise NotImplementedError
    if bool(wm_path) and not os.path.isfile(wm_path):
        print("Path doesn't exist: {0}".format(wm_path))
        raise NotImplementedError
    if bool(gm_path) and not os.path.isfile(gm_path):
        print("Path doesn't exist: {0}".format(gm_path))
        raise NotImplementedError
    if bool(csf_path) and not os.path.isfile(csf_path):
        print("Path doesn't exist: {0}".format(csf_path))
        raise NotImplementedError

    if val and not os.path.isdir(val_path):
        print("Path doesn't exist: {0}".format(val_path))
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

    # Save directory
    if not os.path.exists(save_path):
        print('Create new directory: {0}'.format(save_path))
        os.makedirs(save_path)
    save_path = os.path.join(save_path, time.strftime("%d_%m_%Y_%H_%M_%S", time.gmtime()))
    print('Save path: {0}'.format(save_path))

    # History directory
    history_path = os.path.join(save_path, 'history')
    if not os.path.exists(history_path):
        print('Create new directory: {0}'.format(history_path))
        os.makedirs(history_path)

    # Save parameters
    with open(os.path.join(save_path, 'args.txt'), 'w') as file:
        json.dump(args.__dict__, file, indent=2)

    main(data_path, batch_size, lr, n_epoch,
         shell_list, bvals, bvecs, grid_vec,
         activation, normalization, pooling, filter_start, max_order, interpolation,
         loss_fn_intensity, loss_fn_non_negativity, loss_fn_sparsity, sigma_sparsity,
         intensity_weight, nn_fodf_weight, sparsity_weight,
         load_state, wm_path, gm_path, csf_path,
         val, val_path, threshold, sep_min, max_fiber, split_nb,
         bool(fodf_name), fodf_path, wm_in, gm_in, csf_in, mrtrix_input, ind_path,
         history_path, save_every)

