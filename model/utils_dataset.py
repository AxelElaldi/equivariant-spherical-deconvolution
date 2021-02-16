#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the function to load the training data.

Contents
--------
If you find bugs and/or limitations, please email axel DOT elaldi AT nyu DOT edu.
"""

import os
import torch
import torch.utils.data as data_utils
import numpy as np
import nibabel as nib
import math
import healpy as hp

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dataset(gradients, bvals, bvecs, grid, bsel, batch_size, interpolation='linear'):
    grad_all_target = []
    grad_all_input = []
    distance_all = []
    bvec_all = []
    normalize_all = []
    for b in bsel:
        if b not in bvals:
            print('{0} is not an available shell: {1}'.format(b, np.unique(bvals)))
            raise ValueError
        ind = bvals == b
        grad_sel = gradients[:, ind]
        std_sel = np.std(grad_sel)
        if std_sel == 0:
            std_sel = np.mean(grad_sel)
        # Normalization of the target data and the response functions
        normalize_all.append(std_sel)
        # Target data
        grad_target = grad_sel / std_sel
        grad_target = torch.from_numpy(grad_target.astype(np.float32))
        grad_all_target.append(grad_target)
        # Input data
        grad_input = grad_sel / std_sel
        grad_input = torch.from_numpy(grad_input.astype(np.float32))
        grad_all_input.append(grad_input)
        # b vectors of the shell
        bvecs_sel = bvecs[ind]
        bvec_all.append(bvecs_sel)
        # Distance between b vectors and interpolation grid
        if interpolation == 'linear':
            distance_all.append(
                torch.from_numpy(np.arccos(np.vstack((bvecs_sel, -bvecs_sel)).dot(grid.T)).astype(np.float32)).to(
                    DEVICE))
        else:
            distance_all.append(torch.from_numpy(np.arccos(bvecs_sel.dot(grid.T)).astype(np.float32)).to(DEVICE))

    grad_tuple_input = tuple(grad_all_input)
    grad_all_target = torch.cat(grad_all_target, dim=1)
    n_train = len(grad_all_target)
    print(grad_all_target.shape)
    train_dataset = data_utils.TensorDataset(grad_all_target, *grad_tuple_input)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print('Train set: %i' % n_train)
    return train_loader, train_dataset, distance_all, bvec_all, normalize_all


def dataset_val(gradients, bvals, bvecs, grid, bsel, batch_size, interpolation='linear', split=None):
    grad_all_target_train = []
    grad_all_target_val = []
    grad_all_input_train = []
    grad_all_input_val = []
    distance_all = []
    bvec_all = []
    normalize_all = []
    for b in bsel:
        if b not in bvals:
            print(b, ' is not an available shell: ', np.unique(bvals))
            raise ValueError
        ind = bvals == b
        grad_sel = gradients[:, ind]
        std_sel = np.std(grad_sel)
        if std_sel == 0:
            std_sel = np.mean(grad_sel)
        # Normalization of the target data and the response functions
        normalize_all.append(std_sel)
        # Target data
        grad_target = grad_sel / std_sel
        grad_target = torch.from_numpy(grad_target.astype(np.float32))
        grad_all_target_train.append(grad_target[split[0]])
        grad_all_target_val.append(grad_target[split[1]])
        # Input data
        grad_input = grad_sel / std_sel
        grad_input = torch.from_numpy(grad_input.astype(np.float32))
        grad_all_input_train.append(grad_input[split[0]])
        grad_all_input_val.append(grad_input[split[1]])
        # b vectors of the shell
        bvecs_sel = bvecs[ind]
        bvec_all.append(bvecs_sel)
        # Distance between b vectors and interpolation grid
        if interpolation == 'linear':
            distance_all.append(
                torch.from_numpy(np.arccos(np.vstack((bvecs_sel, -bvecs_sel)).dot(grid.T)).astype(np.float32)).to(
                    DEVICE))
        else:
            distance_all.append(torch.from_numpy(np.arccos(bvecs_sel.dot(grid.T)).astype(np.float32)).to(DEVICE))

    grad_tuple_input_train = tuple(grad_all_input_train)
    grad_tuple_input_val = tuple(grad_all_input_val)
    grad_all_target_train = torch.cat(grad_all_target_train, dim=1)
    grad_all_target_val = torch.cat(grad_all_target_val, dim=1)
    print('Train: ', grad_all_target_train.shape)
    print('Test: ', grad_all_target_val.shape)

    train_dataset = data_utils.TensorDataset(grad_all_target_train, *grad_tuple_input_train)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = data_utils.TensorDataset(grad_all_target_val, *grad_tuple_input_val)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_ind_loader = data_utils.DataLoader(torch.from_numpy(split[1]), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset, val_dataset, val_ind_loader, distance_all, bvec_all, normalize_all


def dataset_augmented(fodf, gradients, bvals, bvecs, grid, bsel, batch_size, interpolation='linear'):
    grad_all_target = []
    grad_all_input = []
    distance_all = []
    bvec_all = []
    normalize_all = []
    fodf = torch.Tensor(fodf)
    for b in bsel:
        if b not in bvals:
            print(b, ' is not an available shell: ', np.unique(bvals))
            raise ValueError
        ind = bvals == b
        grad_sel = gradients[:, ind]
        std_sel = np.std(grad_sel)
        if std_sel == 0:
            std_sel = np.mean(grad_sel)
        # Normalization of the target data and the response functions
        normalize_all.append(std_sel)
        # Target data
        grad_target = grad_sel / std_sel
        grad_target = torch.from_numpy(grad_target.astype(np.float32))
        grad_all_target.append(grad_target)
        # Input data
        grad_input = grad_sel / std_sel
        grad_input = torch.from_numpy(grad_input.astype(np.float32))
        grad_all_input.append(grad_input)
        # b vectors of the shell
        bvecs_sel = bvecs[ind]
        bvec_all.append(bvecs_sel)
        # Distance between b vectors and interpolation grid
        if interpolation == 'linear':
            distance_all.append(
                torch.from_numpy(np.arccos(np.vstack((bvecs_sel, -bvecs_sel)).dot(grid.T)).astype(np.float32)).to(
                    DEVICE))
        else:
            distance_all.append(torch.from_numpy(np.arccos(bvecs_sel.dot(grid.T)).astype(np.float32)).to(DEVICE))

    grad_tuple_input = tuple(grad_all_input)
    grad_all_target = torch.cat(grad_all_target, dim=1)
    n_train = len(grad_all_target)
    print(grad_all_target.shape)
    print(fodf.shape)

    train_dataset = data_utils.TensorDataset(grad_all_target, *grad_tuple_input, fodf)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print('Train set: %i' % n_train)
    return train_loader, train_dataset, distance_all, bvec_all, normalize_all


def dataset_val_augmented(fodf, gradients, bvals, bvecs, grid, bsel, batch_size, interpolation='linear', split=None):
    grad_all_target_train = []
    grad_all_target_val = []
    grad_all_input_train = []
    grad_all_input_val = []
    distance_all = []
    bvec_all = []
    normalize_all = []
    fodf = torch.Tensor(fodf)
    for b in bsel:
        if b not in bvals:
            print(b, ' is not an available shell: ', np.unique(bvals))
            raise ValueError
        ind = bvals == b
        grad_sel = gradients[:, ind]
        std_sel = np.std(grad_sel)
        if std_sel == 0:
            std_sel = np.mean(grad_sel)
        # Normalization of the target data and the response functions
        normalize_all.append(std_sel)
        # Target data
        grad_target = grad_sel / std_sel
        grad_target = torch.from_numpy(grad_target.astype(np.float32))
        grad_all_target_train.append(grad_target[split[0]])
        grad_all_target_val.append(grad_target[split[1]])
        # Input data
        grad_input = grad_sel / std_sel
        grad_input = torch.from_numpy(grad_input.astype(np.float32))
        grad_all_input_train.append(grad_input[split[0]])
        grad_all_input_val.append(grad_input[split[1]])
        # b vectors of the shell
        bvecs_sel = bvecs[ind]
        bvec_all.append(bvecs_sel)
        # Distance between b vectors and interpolation grid
        if interpolation == 'linear':
            distance_all.append(
                torch.from_numpy(np.arccos(np.vstack((bvecs_sel, -bvecs_sel)).dot(grid.T)).astype(np.float32)).to(
                    DEVICE))
        else:
            distance_all.append(torch.from_numpy(np.arccos(bvecs_sel.dot(grid.T)).astype(np.float32)).to(DEVICE))

    grad_tuple_input_train = tuple(grad_all_input_train)
    grad_tuple_input_val = tuple(grad_all_input_val)
    grad_all_target_train = torch.cat(grad_all_target_train, dim=1)
    grad_all_target_val = torch.cat(grad_all_target_val, dim=1)
    print('Train: ', grad_all_target_train.shape)
    print('Test: ', grad_all_target_val.shape)

    train_dataset = data_utils.TensorDataset(grad_all_target_train, *grad_tuple_input_train, fodf[split[0]])
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = data_utils.TensorDataset(grad_all_target_val, *grad_tuple_input_val, fodf[split[1]])
    val_loader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_ind_loader = data_utils.DataLoader(torch.from_numpy(split[1]), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset, val_dataset, val_ind_loader, distance_all, bvec_all, normalize_all


def dataset_test(gradients, bvals, bvecs, grid, bsel, interpolation='linear', split=None):
    grad_all_input = []
    distance_all = []
    bvec_all = []
    normalize_all = []
    for b in bsel:
        if b not in bvals:
            print(b, ' is not an available shell: ', np.unique(bvals))
            raise ValueError
        ind = bvals == b
        grad_sel = gradients[:, ind]
        std_sel = np.std(grad_sel)
        if std_sel == 0:
            std_sel = np.mean(grad_sel)
        # Normalization of the target data and the response functions
        normalize_all.append(std_sel)
        # Input data
        grad_input = grad_sel / std_sel
        grad_input = torch.from_numpy(grad_input.astype(np.float32))
        if split is not None:
            grad_all_input.append(grad_input[np.hstack((split[1], split[2]))])
        else:
            grad_all_input.append(grad_input)
        # b vectors of the shell
        bvecs_sel = bvecs[ind]
        bvec_all.append(bvecs_sel)
        # Distance between b vectors and interpolation grid
        if interpolation == 'linear':
            distance_all.append(
                torch.from_numpy(np.arccos(np.vstack((bvecs_sel, -bvecs_sel)).dot(grid.T)).astype(np.float32)).to(
                    DEVICE))
        else:
            distance_all.append(torch.from_numpy(np.arccos(bvecs_sel.dot(grid.T)).astype(np.float32)).to(DEVICE))
    n_train = len(grad_all_input[0])
    print('Test set: ', n_train)

    return grad_all_input, distance_all, bvec_all, normalize_all


def dataset_test_augmented(gradients, fodf, bvals, bvecs, grid, bsel, interpolation='linear', split=None):
    grad_all_input = []
    distance_all = []
    bvec_all = []
    normalize_all = []
    fodf = torch.Tensor(fodf)
    if split is not None:
        fodf = fodf[np.hstack((split[1], split[2]))]
    for b in bsel:
        if b not in bvals:
            print(b, ' is not an available shell: ', np.unique(bvals))
            raise ValueError
        ind = bvals == b
        grad_sel = gradients[:, ind]
        std_sel = np.std(grad_sel)
        if std_sel == 0:
            std_sel = np.mean(grad_sel)
        # Normalization of the target data and the response functions
        normalize_all.append(std_sel)
        # Input data
        grad_input = grad_sel / std_sel
        grad_input = torch.from_numpy(grad_input.astype(np.float32))
        if split is not None:
            grad_all_input.append(grad_input[np.hstack((split[1], split[2]))])
        else:
            grad_all_input.append(grad_input)
        # b vectors of the shell
        bvecs_sel = bvecs[ind]
        bvec_all.append(bvecs_sel)
        # Distance between b vectors and interpolation grid
        if interpolation == 'linear':
            distance_all.append(
                torch.from_numpy(np.arccos(np.vstack((bvecs_sel, -bvecs_sel)).dot(grid.T)).astype(np.float32)).to(
                    DEVICE))
        else:
            distance_all.append(torch.from_numpy(np.arccos(bvecs_sel.dot(grid.T)).astype(np.float32)).to(DEVICE))
    n_train = len(grad_all_input[0])
    print('Test set: ', n_train)
    return grad_all_input, distance_all, bvec_all, fodf, normalize_all


def load_fodf_input(fodf_path, wm=False, gm=False, csf=False, ind=None):
    print('---------------------------------------------')
    print('---------------------------------------------')
    print('LOAD fODF INPUT')
    print('---------------------------------------------')
    if (not wm) and (not gm) and (not csf):
        print('INPUT: You need to specify what tissue to load as input of the model')
        raise ValueError
    if wm:
        img = nib.load(fodf_path + '/fodf.nii')
        data_wm = img.get_fdata()
        if ind is not None:
            data = data_wm[ind]
        else:
            data = data_wm.reshape((-1, data_wm.shape[-1]))
        print('INPUT: load wm. Size: ', data.shape)
    if gm:
        img = nib.load(fodf_path + '/fodf_gm.nii')
        data_gm = img.get_fdata()
        if ind is not None:
            data_gm = data_gm[ind]
        else:
            data_gm = data_gm.reshape((-1, data_gm.shape[-1]))
        print('INPUT: load gm. Size: ', data_gm.shape)
        if wm:
            data = np.hstack((data, data_gm))
        else:
            data = data_gm
    if csf:
        img = nib.load(fodf_path + '/fodf_csf.nii')
        data_csf = img.get_fdata()
        if ind is not None:
            data_csf = data_csf[ind]
        else:
            data_csf = data_csf.reshape((-1, data_csf.shape[-1]))
        print('INPUT: load csf. Size: ', data_csf.shape)
        if gm or wm:
            data = np.hstack((data, data_csf))
        else:
            data = data_csf
    print('INPUT: final Size: ', data.shape)
    return data


def load_pretrained(model, load_state=None, wm_path=None, gm_path=None, csf_path=None, norm=None):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Base model
    load_state : str
        If specified, load saved model weights.
    wm_path: str
        If specified, load white matter rf. Must be a txt file from MRtrix of dimension
        (n_shell, max_order).
    gm_path: str
        If specified, load white matter rf. Must be a txt file from MRtrix of dimension
        (n_shell,).
    csf_path: str
        If specified, load white matter rf. Must be a txt file from MRtrixof dimension
        (n_shell,).
    norm: list
        Normalize the RF

    Returns
    -------
     model : torch.nn.Module
        Pre trained model
    """
    n_shell = model.n_shell
    max_order = model.max_sh_order
    if not norm:
        norm = [1]*n_shell
    if len(norm) != n_shell:
        raise NotImplementedError
    norm = np.array(norm)
    print('Norm RF: ', norm)

    if load_state:
        model.load_state_dict(torch.load(load_state), strict=False)
    # Load response functions
    if wm_path:
        rf = np.loadtxt(wm_path)
        if len(rf.shape) == 1:
            rf = rf.reshape(1, len(rf))
        if n_shell != rf.shape[0]:
            print("WM response function and shells doesn't match: ")
            print("WM rf: ", rf.shape[0])
            print("Shell: ", n_shell)
            raise NotImplementedError
        if max_order // 2 + 1 > rf.shape[1]:
            print("WM response function doesn't have enough coefficients: ")
            print("WM rf: ", rf.shape[1])
            print("Max order: ", max_order // 2 + 1)
            k = max_order // 2 + 1 - rf.shape[1]
            rf = np.hstack((rf, np.zeros((rf.shape[0], k))))
        rf = rf[:, :max_order // 2 + 1] / norm[:, None]
        print('WM rf shape: ', rf.shape)
        print(rf)
        model.rf_shc_wm = torch.nn.Parameter(torch.Tensor(rf),
                                             requires_grad=False)
    if gm_path:
        rf = np.loadtxt(gm_path)
        if rf.shape == ():
            rf = np.array([rf])
        if len(rf.shape) != 1:
            print("GM response function has too many dimension: ")
            print("GM rf: ", rf.shape)
            print("Should be: ", 1)
            raise NotImplementedError
        if n_shell != rf.shape[0]:
            print("Response function and shells doesn't match: ")
            print("GM rf: ", rf.shape[0])
            print("Should be: ", n_shell)
            raise NotImplementedError
        rf = rf.reshape(n_shell, 1) / norm[:, None]
        print('GM rf shape: ', rf.shape)
        print(rf)
        model.rf_shc_gm = torch.nn.Parameter(torch.Tensor(rf),
                                             requires_grad=False)
    if csf_path:
        rf = np.loadtxt(csf_path)
        if rf.shape == ():
            rf = np.array([rf])
        if len(rf.shape) != 1:
            print("CSF response function has too many dimension: ")
            print("CSF rf: ", rf.shape)
            print("Should be: ", 1)
            raise NotImplementedError
        if n_shell != rf.shape[0]:
            print("Response function and shells doesn't match: ")
            print("CSF rf: ", rf.shape[0])
            print("Should be: ", n_shell)
            raise NotImplementedError
        rf = rf.reshape(n_shell, 1) / norm[:, None]
        print('CSF rf shape: ', rf.shape)
        print(rf)
        model.rf_shc_csf = torch.nn.Parameter(torch.Tensor(rf),
                                              requires_grad=False)
    return model


class WeightedLoss(torch.nn.Module):
    def __init__(self, norm, sigma=None):
        """
        Parameters
        ----------
        norm : str
            Name of the loss.
        sigma : float
            Hyper parameter of the loss.
        """
        super(WeightedLoss, self).__init__()
        if norm not in ['L2', 'L1', 'cauchy', 'welsch', 'geman']:
            raise NotImplementedError('Expected L1, L2, cauchy, welsh, geman but got {}'.format(norm))
        if sigma is None and norm in ['cauchy', 'welsch', 'geman']:
            raise NotImplementedError('Expected a loss hyper parameter for {}'.format(norm))
        self.norm = norm
        self.sigma = sigma

    def forward(self, img1, img2, wts=None):
        """
        Parameters
        ----------
        img1 : torch.Tensor
            Prediction tensor
        img2 : torch.Tensor
            Ground truth tensor
        wts: torch.nn.Parameter
            If specified, the weight of the grid.
        Returns
        -------
         loss : torch.Tensor
            Loss of the predicted tensor
        """
        if self.norm == 'L2':
            out = (img1 - img2)**2
        elif self.norm == 'L1':
            out = torch.abs(img1 - img2)
        elif self.norm == 'cauchy':
            out = 2 * torch.log(1 + ((img1 - img2)**2 / (2*self.sigma)))
        elif self.norm == 'welsch':
            out = 2 * (1-torch.exp(-0.5 * ((img1 - img2)**2 / self.sigma)))
        elif self.norm == 'geman':
            out = 2 * (2*((img1 - img2)**2 / self.sigma) / ((img1 - img2)**2 / self.sigma + 4))
        else:
            raise ValueError('Expected L1, L2, cauchy, welsh, geman but got {}'.format(self.norm))

        if wts is not None:
            out = out * wts
        loss = out.sum() / (out.size(0) * out.size(1))
        return loss


def extract_rf(result_path, model, norm):
    """Extract and save the learned Response Function

    Parameters
    ----------
    result_path : str
        The path to save the results
    model : torch.nn.Module
        The model
    norm: list
        Normalize the RF

    """
    norm = np.array(norm)
    print('Norm RF: ', norm)
    # Save Response Function
    rf_list = [model.rf_shc_wm.cpu().detach().numpy() * norm[:, None]]
    rf_name = ['wm_response.txt']
    if model.gm:
        rf_list.append(model.rf_shc_gm.cpu().detach().numpy() * norm[:, None])
        rf_name.append('gm_response.txt')
    if model.csf:
        rf_list.append(model.rf_shc_csf.cpu().detach().numpy() * norm[:, None])
        rf_name.append('csf_response.txt')

    # Create the result directory
    rf_path = os.path.join(result_path, 'rf')
    if not os.path.exists(rf_path):
        os.makedirs(rf_path)
    # Save the result
    for name, rf in zip(rf_name, rf_list):
        np.savetxt(os.path.join(rf_path, name), rf)


def gen_distortion_weights(grid='equiangular', n_side=64):
    """
    Generate weights for loss function according to latitude to account for
    distortion due to equirectangular projection.

    Taken from https://github.com/xuyanyu-shh/Saliency-detection-in-360-video

    Parameters
    ----------
    grid : str
        Type of grid.
    n_side : int
        Number of rows/columns in equiangular projection

    Returns
    -------
    weight : torch.nn.Parameter
        Weights to account for projection error in loss.

    """
    if grid == "equiangular":
        if n_side % 2 != 0:
            raise ValueError('Need an even number of points on the latitude')

        weight = torch.zeros(1, 1, n_side, n_side)
        theta_range = torch.linspace(0, np.pi, steps=n_side + 1)
        dphi = 2 * np.pi / n_side
        for theta_idx in range(n_side):
            area = dphi * abs((math.cos(theta_range[theta_idx]) - math.cos(theta_range[theta_idx+1])))
            weight[:, :, theta_idx, :] = area
        weight = weight.flatten(-2)

    elif grid == "healpix":
        n = 12 * n_side ** 2
        area = hp.nside2pixarea(n_side)
        weight = torch.ones(1, 1, n) * area

    else:
        raise NotImplementedError

    weight = torch.nn.Parameter(weight, requires_grad=False)

    return weight
