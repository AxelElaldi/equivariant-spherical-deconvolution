#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the network architectures used in this project.

If you find bugs and/or limitations, please email axel DOT elaldi AT nyu DOT edu.
"""

import sys
sys.path.append("./generate_result")
sys.path.append("./generate_data")
import torch.nn as nn
import torch
import numpy as np
from deepsphere.layers.chebyshev import SphericalChebConv
from healpix_pooling import Healpix
from utils import sh_matrix
from utils_healpix import get_healpix_laplacians


class DeepCSD(nn.Module):
    def __init__(self, shell, max_sh_order, grid, bvec, n_side,
                 activation='relu', normalization='batch', pooling='mean', filter_start=2,
                 csf=False, gm=False, with_interpolation='linear',
                 fodf_in=False, wm_in=False, gm_in=False, csf_in=False, mrtrix_in=False):
        super(DeepCSD, self).__init__()
        # The shells of the input signal
        self.shell = shell
        self.n_shell = len(shell)

        # The cartesian coordinates of the input signal
        self.bvec = bvec
        self.n_grad = []
        for i in range(len(bvec)):
            self.n_grad.append(len(bvec[i]))
        self.n_grad_t = np.sum(self.n_grad)

        # The cartesian coordinate of the dense grid used by the model
        self.grid = grid
        self.n_side = n_side

        # Layer properties of the model
        self.max_sh_order = max_sh_order
        self.activation = activation
        self.normalization = normalization
        self.pooling = pooling
        self.filter_start = filter_start

        # Tissue properties of the model
        self.csf = csf
        self.gm = gm

        # The augmented input signal properties
        self.fodf_in = fodf_in
        self.wm_in = wm_in
        self.gm_in = gm_in
        self.csf_in = csf_in
        self.mrtrix_in = mrtrix_in

        # The interpolation layer of the model
        self.with_interpolation = with_interpolation
        self.interpolation = Interpolation(self.with_interpolation)

        # The response functions of the model. This is the tensor of SH coefficients
        self.nb_max_fiber = 1
        self.rf_shc_wm = nn.Parameter(torch.randn(self.n_shell, self.max_sh_order // 2 + 1),
                                      requires_grad=False)
        if self.gm:
            self.nb_max_fiber += 1
            self.rf_shc_gm = nn.Parameter(torch.randn(self.n_shell, 1),
                                          requires_grad=False)
        if self.csf:
            self.nb_max_fiber += 1
            self.rf_shc_csf = nn.Parameter(torch.randn(self.n_shell, 1),
                                           requires_grad=False)
        self.nb_in_fiber = wm_in + gm_in + csf_in

        # Encoder/Decoder of the model
        if fodf_in:
            self.compute_fodf = GraphCNNCatUnet2(self.n_shell + self.nb_in_fiber, self.filter_start,
                                                 self.n_side, self.nb_max_fiber,
                                                 self.activation, self.normalization,
                                                 self.pooling)
        else:
            self.compute_fodf = GraphCNNCatUnet2(self.n_shell, self.filter_start,
                                                 self.n_side, self.nb_max_fiber,
                                                 self.activation, self.normalization,
                                                 self.pooling)

        # Matrix to get the spherical harmonic coefficients from a signal
        # and to get the signal from the spherical harmonic coefficients
        self.S2SH, self.SH2S = sh_matrix(self.max_sh_order, self.grid, 1)
        self.S2SH = nn.Parameter(torch.Tensor(self.S2SH), requires_grad=False)
        self.SH2S = nn.Parameter(torch.Tensor(self.SH2S), requires_grad=False)

        self.S2SH_rf, self.SH2S_rf = sh_matrix(self.max_sh_order, self.grid, 0)
        self.S2SH_rf = nn.Parameter(torch.Tensor(self.S2SH_rf), requires_grad=False)
        self.SH2S_rf = nn.Parameter(torch.Tensor(self.SH2S_rf), requires_grad=False)

        self.SH2S_bvec = nn.ParameterList([])
        for i in range(self.n_shell):
            if self.shell[i] == 0:
                self.SH2S_bvec.append(nn.Parameter(torch.Tensor(np.array([0])), requires_grad=False))
            else:
               _, SH2S_bvec = sh_matrix(self.max_sh_order, self.bvec[i], 1)
               self.SH2S_bvec.append(nn.Parameter(torch.Tensor(SH2S_bvec), requires_grad=False))

        if self.with_interpolation == 'sh':
            self.int_weight = nn.ParameterList([])
            for i in range(self.n_shell):
                if self.shell[i] == 0:
                    self.int_weight.append(nn.Parameter(torch.Tensor(np.array([0])), requires_grad=False))
                else:
                    sh_order_shell = 8
                    while int((sh_order_shell + 1) * (sh_order_shell/2 + 1)) > self.bvec[i].shape[0]:
                        sh_order_shell -= 2
                    print(i, ' ,', self.shell[i], ' ,', self.bvec[i].shape, ' ,', sh_order_shell)
                    s2sh_shell, _ = sh_matrix(sh_order_shell, self.bvec[i], 1)
                    _, sh2s_grid = sh_matrix(sh_order_shell, self.grid, 1)
                    self.int_weight.append(nn.Parameter(torch.Tensor(s2sh_shell.dot(sh2s_grid)),
                                                        requires_grad=False))
        if self.mrtrix_in:
            input_grid = self.grid
            input_grid[:, 0] = - input_grid[:, 0]
            _, self.SH2S_input = sh_matrix(self.max_sh_order, input_grid, 1)
            self.SH2S_input = nn.Parameter(torch.Tensor(self.SH2S_input), requires_grad=False)
        else:
            self.SH2S_input = self.SH2S

    def convolution(self, x, fodf_shc_wm, fodf_shc_gm, fodf_shc_csf):
        """Spherical convolution on S2.

        Parameters
        ----------
        x : torch.Tensor (N_batch * N_shell * N_grid)
            The input spherical signal
        fodf_shc_wm : torch.Tensor (N_batch * N_coef)
            The WM fODF spherical harmonic coefficients
        fodf_shc_gm : torch.Tensor (N_batch * 1)
            The GM fODF spherical harmonic coefficients
        fodf_shc_csf : torch.Tensor (N_batch * 1)
            The CSF fODF spherical harmonic coefficients

        Returns
        -------
        output : torch.Tensor (N_batch * N_shell * N_grid)
            The convolved signal
        output_wm : torch.Tensor (N_batch * N_shell * N_grid)
            The white matter signal
        output_gm : torch.Tensor (N_batch * N_shell * N_grid)
            The gray matter signal
        output_csf : torch.Tensor (N_batch * N_shell * N_grid)
            The csf signal
        """
        # Compute the reconstruction of the input signal.
        # We do it in the SH space, such that the convolution between the fODF
        # and the response function is equivalent to a matrix multiplication.
        output_shc = x.new_zeros(x.shape[0], x.shape[1]-self.nb_in_fiber, self.S2SH.shape[1])
        # The GM and CSF have only one SHC (degree 0, order 0), and thus they are easier to compute
        if self.gm:
            output_gm_sell = fodf_shc_gm.matmul(self.rf_shc_gm.T)
        if self.csf:
            output_csf_sell = fodf_shc_csf.matmul(self.rf_shc_csf.T)
        nb_shc = 0
        for l in range(self.max_sh_order // 2 + 1):
            rf_shc_wm_l = self.rf_shc_wm[:, l:l + 1]
            fodf_shc_wm_l = fodf_shc_wm[:, None, nb_shc:nb_shc + 4 * l + 1]
            output_shc[:, :, nb_shc:nb_shc + 4 * l + 1] = np.sqrt(4 * np.pi / (4 * l + 1)) * rf_shc_wm_l.matmul(
                fodf_shc_wm_l)
            nb_shc += 4 * l + 1
        #
        # Get the reconstruction signal from the SH coefficients
        output_wm = x.new_zeros(output_shc.shape[0], self.n_grad_t)
        output_gm = x.new_zeros(output_shc.shape[0], self.n_grad_t)
        output_csf = x.new_zeros(output_shc.shape[0], self.n_grad_t)
        output = x.new_zeros(output_shc.shape[0], self.n_grad_t)
        start = 0
        for i in range(self.n_shell):
            if self.shell[i] == 0:
                output_wm[:, start:start + self.n_grad[i]] = (output_shc[:, 0, 0] / np.sqrt(4 * np.pi))[:, None]
                if self.gm:
                    output_gm[:, start:start + self.n_grad[i]] = output_gm_sell[:, i:i + 1]
                if self.csf:
                    output_csf[:, start:start + self.n_grad[i]] = output_csf_sell[:, i:i + 1]
                    output[:, start:start + self.n_grad[i]] = (output_wm + output_gm + output_csf)[:, start:start + self.n_grad[i]]
            else:
                output_wm[:, start:start + self.n_grad[i]] = output_shc[:, i].matmul(self.SH2S_bvec[i])
                if self.gm:
                    output_gm[:, start:start + self.n_grad[i]] = output_gm_sell[:, i:i+1]
                if self.csf:
                    output_csf[:, start:start + self.n_grad[i]] = output_csf_sell[:, i:i+1]
                output[:, start:start + self.n_grad[i]] = (output_wm + output_gm + output_csf)[:, start:start + self.n_grad[i]]
            start += self.n_grad[i]

        return output, output_wm, output_gm, output_csf

    def forward(self, dwi_list, distance, fodf=None):
        """Forward of the model.
        Try to reconstruct the input signal X with a decomposition
        X = Response function * fODF
        The model learns how to derive the fODF from the input signal

        Parameters
        ----------
        dwi_list : list of torch.Tensor (N_shell x N_batch x N_grad)
            The input signal

        Returns
        -------
        output : torch.Tensor (N_batch * N_grad_tot)
            The reconstructed signal
        fodf_wm : torch.Tensor (N_batch * N_grid)
            The predicted WM fODF
        fodf_gm : torch.Tensor (N_batch * 1)
            The predicted Gm fODF
        fodf_csf : torch.Tensor (N_batch * 1)
            The predicted CSF fODF
        fodf_shc_wm : torch.Tensor (N_batch * max_sh_order)
            The predicted WM fODF spherical harmonic coefficients
        fodf_shc_gm : torch.Tensor (N_batch * 1)
            The predicted Gm fODF spherical harmonic coefficients
        fodf_shc_csf : torch.Tensor (N_batch * 1)
            The predicted CSF fODF spherical harmonic coefficients
        """
        # Interpolation
        if self.fodf_in:
            x = dwi_list[0].new_zeros(dwi_list[0].shape[0], self.n_shell + self.nb_in_fiber, len(self.grid))
            k = 0
            if self.wm_in and self.nb_in_fiber == 1:
                nb_c = fodf.shape[1]
                x[:, k] = fodf.matmul(self.SH2S_input[:nb_c])
            else:
                if self.wm_in:
                    fodf1 = fodf[:, :-self.nb_in_fiber+1]
                    nb_c = fodf1.shape[1]
                    x[:, k] = fodf1.matmul(self.SH2S_input[:nb_c])
                    k += 1
                if self.gm_in:
                    if self.csf_in:
                        x[:, k] = fodf[:, -2:-1] * dwi_list[0].new_ones(len(dwi_list[0]), len(self.grid))
                    else:
                        x[:, k] = fodf[:, -1:] * dwi_list[0].new_ones(len(dwi_list[0]), len(self.grid))
                    k += 1
                if self.csf_in:
                    x[:, k] = fodf[:, -1:] * dwi_list[0].new_ones(len(dwi_list[0]), len(self.grid))
            start = self.nb_in_fiber
        else:
            x = dwi_list[0].new_zeros(dwi_list[0].shape[0], self.n_shell, len(self.grid))
            start = 0
        for i in range(len(dwi_list)):
            if self.shell[i] == 0:
                x[:, i + start] = dwi_list[0].new_ones(len(dwi_list[i]), len(self.grid)) * dwi_list[i].mean(axis=-1, keepdim=True)
            else:
                if self.with_interpolation == 'sh':
                    x[:, i + start] = self.interpolation(dwi_list[i], self.int_weight[i])
                else:
                    x[:, i + start] = self.interpolation(dwi_list[i], distance[i])
        # Encoder/Decoder
        fodf_wm, fodf_gm, fodf_csf = self.compute_fodf(x)

        # The fODF should be antipodal symmetric. But because of computation error, the previous output
        # might not be symmetric, thus we transform it into a symmetric signal by deleting its
        # odd degree spherical harmonic coefficients.

        # Get the fODF SH coefficients, then the symmetric fODF
        # BE CAREFUL: fODF GM and CSF are not a constant function over the sphere!! They are Dirac function!
        # which is equivalent to a constant function weight/(4*pi).
        fodf_shc_gm = fodf_gm * np.sqrt(4 * np.pi)  # fodf_gm is the PV/(4*pi) of the GM tissue.
        fodf_shc_csf = fodf_csf * np.sqrt(4 * np.pi)  # fodf_csf is the PV/(4*pi) of the CSF tissue
        fodf_shc_wm = fodf_wm.matmul(self.S2SH)
        fodf_wm = fodf_shc_wm.matmul(self.SH2S)
        # Compute the reconstruction of the input signal.
        # We do it in the SH space, such that the convolution between the fODF
        # and the response function is equivalent to a matrix multiplication.
        output, _, _, _ = self.convolution(x, fodf_shc_wm, fodf_shc_gm, fodf_shc_csf)

        return output, fodf_wm, fodf_gm, fodf_csf, fodf_shc_wm, fodf_shc_gm, fodf_shc_csf


class GraphCNNCatUnet2(nn.Module):
    """Spherical GCNN Autoencoder.
    """

    def __init__(self, nb_shell, filter_start, n_side, nb_max_fiber,
                 activation, normalization, pooling, kernel_size=6):
        """Initialization.
        """
        super(GraphCNNCatUnet2, self).__init__()
        self.kernel_size = kernel_size
        self.nb_shell = nb_shell
        self.filter_start = filter_start
        self.nb_max_fiber = nb_max_fiber

        self.depth = 5
        self.laps = get_healpix_laplacians(n_side, self.depth)

        # Normalization/Pooling/Unpooling/Activation
        self.normalization = normalization
        self.pooling = pooling
        if self.pooling:
            if self.pooling == 'mean':
                self.pooling = 'average'
            self.pooling_class = Healpix(mode=self.pooling)  # can be average or max
        self.pool = self.pooling_class.pooling
        self.unpool = self.pooling_class.unpooling
        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

        # First conv
        self.conv1 = self.conv(self.nb_shell, self.filter_start, self.laps[4], self.kernel_size,
                               self.normalization)
        # Encoder
        # Bloc 4
        self.enc_conv_4 = self.conv(self.filter_start, self.filter_start, self.laps[4], self.kernel_size,
                                    self.normalization)
        # Bloc 3
        self.enc_conv_3 = self.conv(self.filter_start, 2 * self.filter_start, self.laps[3], self.kernel_size,
                                    self.normalization)
        self.enc_conv_3_2 = self.conv(2 * self.filter_start, 2 * self.filter_start, self.laps[3], self.kernel_size,
                                      self.normalization)
        # Bloc 2
        self.enc_conv_2 = self.conv(2 * self.filter_start, 4 * self.filter_start, self.laps[2], self.kernel_size,
                                    self.normalization)
        self.enc_conv_2_2 = self.conv(4 * self.filter_start, 4 * self.filter_start, self.laps[2], self.kernel_size,
                                      self.normalization)
        # Bloc 1
        self.enc_conv_1 = self.conv(4 * self.filter_start, 8 * self.filter_start, self.laps[1], self.kernel_size,
                                    self.normalization)
        self.enc_conv_1_2 = self.conv(8 * self.filter_start, 8 * self.filter_start, self.laps[1], self.kernel_size,
                                      self.normalization)

        # Bottom
        self.bot = self.conv(8 * self.filter_start, 16 * self.filter_start, self.laps[0], self.kernel_size,
                             self.normalization)
        self.bot_2 = self.conv(16 * self.filter_start, 8 * self.filter_start, self.laps[0], self.kernel_size,
                               self.normalization)

        # Decoder
        # Bloc 1
        self.dec_conv_1 = self.conv(2*8*self.filter_start, 8*self.filter_start, self.laps[1], self.kernel_size,
                                    self.normalization)
        self.dec_conv_1_2 = self.conv(8 * self.filter_start, 4 * self.filter_start, self.laps[1], self.kernel_size,
                                      self.normalization)
        # Bloc 2
        self.dec_conv_2 = self.conv(2*4*self.filter_start, 4*self.filter_start, self.laps[2], self.kernel_size,
                                    self.normalization)
        self.dec_conv_2_2 = self.conv(4 * self.filter_start, 2 * self.filter_start, self.laps[2], self.kernel_size,
                                      self.normalization)
        # Bloc 3
        self.dec_conv_3 = self.conv(2*2*self.filter_start, 2 * self.filter_start, self.laps[3], self.kernel_size,
                                    self.normalization)
        self.dec_conv_3_2 = self.conv(2 * self.filter_start, self.filter_start, self.laps[3], self.kernel_size,
                                      self.normalization)
        # Bloc 4
        self.dec_conv_4 = self.conv(2*self.filter_start, self.filter_start, self.laps[4], self.kernel_size,
                                    self.normalization)
        self.dec_conv_4_2 = self.conv(self.filter_start, self.filter_start, self.laps[4], self.kernel_size,
                                      self.normalization)
        # fODF prediction
        self.prediction = self.conv(self.filter_start, self.nb_max_fiber, self.laps[4], self.kernel_size,
                                    None)

    def conv(self, nfeature_in, nfeature_out, lap, kernel_size,
             normalization):
        # Convolution
        layers = [Permute(), SphericalChebConv(nfeature_in, nfeature_out, lap, kernel_size), Permute()]
        # Normalization
        if normalization:
            if normalization == 'batch':
                layers.append(nn.BatchNorm1d(nfeature_out))
            elif normalization == 'instance':
                layers.append(nn.InstanceNorm1d(nfeature_out))
            else:
                raise NotImplementedError
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded.
        Returns:
            :obj:`torch.Tensor`: output
        """
        x_1 = self.conv1(x)
        x_enc4 = self.activation(x_1)

        # Encoder
        x_enc4 = self.enc_conv_4(x_enc4)
        x_enc4 = self.activation(x_enc4)
        x_enc3, ind_3 = self.pool(x_enc4)

        x_enc3 = self.enc_conv_3(x_enc3)
        x_enc3 = self.activation(x_enc3)
        x_enc3 = self.enc_conv_3_2(x_enc3)
        x_enc3 = self.activation(x_enc3)
        x_enc2, ind_2 = self.pool(x_enc3)

        x_enc2 = self.enc_conv_2(x_enc2)
        x_enc2 = self.activation(x_enc2)
        x_enc2 = self.enc_conv_2_2(x_enc2)
        x_enc2 = self.activation(x_enc2)
        x_enc1, ind_1 = self.pool(x_enc2)

        x_enc1 = self.enc_conv_1(x_enc1)
        x_enc1 = self.activation(x_enc1)
        x_enc1 = self.enc_conv_1_2(x_enc1)
        x_enc1 = self.activation(x_enc1)
        x_bot, ind_bot = self.pool(x_enc1)

        # Bottom
        x_bot = self.bot(x_bot)
        x_bot = self.activation(x_bot)
        x_bot = self.bot_2(x_bot)
        x_bot = self.activation(x_bot)

        # Decoder
        x_dec1 = self.unpool(x_bot, ind_bot)
        x_dec1 = torch.cat((x_dec1, x_enc1), dim=1)
        x_dec1 = self.dec_conv_1(x_dec1)
        x_dec1 = self.activation(x_dec1)
        x_dec1 = self.dec_conv_1_2(x_dec1)
        x_dec1 = self.activation(x_dec1)

        x_dec2 = self.unpool(x_dec1, ind_1)
        x_dec2 = torch.cat((x_dec2, x_enc2), dim=1)
        x_dec2 = self.dec_conv_2(x_dec2)
        x_dec2 = self.activation(x_dec2)
        x_dec2 = self.dec_conv_2_2(x_dec2)
        x_dec2 = self.activation(x_dec2)

        x_dec3 = self.unpool(x_dec2, ind_2)
        x_dec3 = torch.cat((x_dec3, x_enc3), dim=1)
        x_dec3 = self.dec_conv_3(x_dec3)
        x_dec3 = self.activation(x_dec3)
        x_dec3 = self.dec_conv_3_2(x_dec3)
        x_dec3 = self.activation(x_dec3)

        x_dec4 = self.unpool(x_dec3, ind_3)
        x_dec4 = torch.cat((x_dec4, x_enc4), dim=1)
        x_dec4 = self.dec_conv_4(x_dec4)
        x_dec4 = self.activation(x_dec4)
        x_dec4 = self.dec_conv_4_2(x_dec4)
        x_dec4 = self.activation(x_dec4)

        # fODF prediction
        if self.nb_max_fiber > 1:
            fodf_flatten = self.softplus(self.prediction(x_dec4))
        else:
            fodf_flatten = self.activation(self.prediction(x_dec4))

        fodf_wm = fodf_flatten[:, 0]
        fodf_t2 = 0
        fodf_t3 = 0
        if self.nb_max_fiber > 1:
            fodf_t2 = torch.max(fodf_flatten[:, 1], keepdim=True, dim=-1)[0]
        if self.nb_max_fiber > 2:
            fodf_t3 = torch.max(fodf_flatten[:, 2], keepdim=True, dim=-1)[0]
        return fodf_wm, fodf_t2, fodf_t3


class Permute(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1).contiguous()


class Interpolation(nn.Module):
    def __init__(self, interpolation):
        super(Interpolation, self).__init__()
        self.interpolation = interpolation
        if interpolation == 'network':
            self.fc1 = nn.Linear(1, 16)
            self.norm1 = nn.BatchNorm1d(16)
            self.fc2 = nn.Linear(16, 32)
            self.norm2 = nn.BatchNorm1d(32)
            self.fc3 = nn.Linear(32, 1)
            self.activation = nn.ReLU()
            self.softmax = nn.Softmax(dim=0)

    def forward(self, dwi_list, distance):
        if self.interpolation == 'network':
            n, v = distance.shape
            distance = distance.view(-1, 1)
            w = self.norm1(self.activation(self.fc1(distance)))
            w = self.norm2(self.activation(self.fc2(w)))
            w = self.fc3(w)
            w = w.view(n, v)
            w = self.softmax(w)
        elif self.interpolation == 'linear':
            w = 1 / (distance + 1e-16) ** 6
            w = w / w.sum(0)
            dwi_list = torch.cat((dwi_list, dwi_list), 1)
        elif self.interpolation == 'sh':
            w = distance
        else:
            raise NotImplementedError
        x = dwi_list.matmul(w)
        return x

    def get_weight(self, distance):
        if self.interpolation == 'network':
            n, v = distance.shape
            distance = distance.reshape(-1, 1)
            w = self.norm1(self.activation(self.fc1(distance)))
            w = self.norm2(self.activation(self.fc2(w)))
            w = self.fc3(w)
            w = w.reshape(n, v)
            w = self.softmax(w)
        elif self.interpolation == 'linear':
            w = 1 / (distance + 1e-16) ** 6
            w = w / w.sum(0)
        elif self.interpolation == 'sh':
            w = distance
        else:
            raise NotImplementedError
        return w
