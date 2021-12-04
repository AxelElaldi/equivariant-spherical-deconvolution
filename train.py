import argparse
import os
import numpy as np
import pickle
import json
import time

from utils.loss import Loss
from utils.sampling import HealpixSampling, ShellSampling
from utils.dataset import DMRIDataset
from model.shutils import ComputeSignal
from utils.response import load_response_function
from model.model import Model

import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(data_path, batch_size, lr, n_epoch, kernel_size, 
         filter_start, sh_degree, depth, n_side,
         rf_name, wm, gm, csf,
         loss_fn_intensity, loss_fn_non_negativity, loss_fn_sparsity, sigma_sparsity,
         intensity_weight, nn_fodf_weight, sparsity_weight,
         save_path, save_every, normalize, load_state):
    """Train a model
    Args:
        data_path (str): Data path
        batch_size (int): Batch size
        lr (float): Learning rate
        n_epoch (int): Number of training epoch
        kernel_size (int): Kernel Size
        filter_start (int): Number of output features of the first convolution layer
        sh_degree (int): Spherical harmonic degree of the fODF
        depth (int): Graph subsample depth
        n_side (int): Resolution of the Healpix map
        rf_name (str): Response function algorithm name
        wm (float): Use white matter
        gm (float): Use gray matter
        csf (float): Use CSF
        loss_fn_intensity (str): Name of the intensity loss
        loss_fn_non_negativity (str): Name of the nn loss
        loss_fn_sparsity (str): Name of the sparsity loss
        intensity_weight (float): Weight of the intensity loss
        nn_fodf_weight (float): Weight of the nn loss
        sparsity_weight (float): Weight of the sparsity loss
        save_path (str): Save path
        save_every (int): Frequency to save the model
        normalize (bool): Normalize the fODFs
        load_state (str): Load pre trained network
    """
    
    # Load the shell and the graph samplings
    shellSampling = ShellSampling(f'{data_path}/bvecs.bvecs', f'{data_path}/bvals.bvals', sh_degree=sh_degree, max_sh_degree=8)
    graphSampling = HealpixSampling(n_side, depth, sh_degree=sh_degree)

    # Load the image and the mask
    dataset = DMRIDataset(f'{data_path}/features.nii', f'{data_path}/mask.nii')
    dataloader_train = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    n_batch = len(dataloader_train)
    
    # Load the Polar filter used for the deconvolution
    polar_filter_equi, polar_filter_inva = load_response_function(f'{data_path}/response_functions/{rf_name}', wm=wm, gm=gm, csf=csf, max_degree=sh_degree, n_shell=len(shellSampling.shell_values))

    # Create the deconvolution model
    model = Model(polar_filter_equi, polar_filter_inva, shellSampling, graphSampling, filter_start, kernel_size, normalize)
    if load_state:
        print(load_state)
        model.load_state_dict(torch.load(load_state), strict=False)
    # Load model in GPU
    model = model.to(DEVICE)
    torch.save(model.state_dict(), os.path.join(save_path, 'history', 'epoch_0.pth'))

    # Loss
    intensity_criterion = Loss(loss_fn_intensity)
    non_negativity_criterion = Loss(loss_fn_non_negativity)
    sparsity_criterion = Loss(loss_fn_sparsity, sigma_sparsity)
    # Create dense interpolation used for the non-negativity and the sparsity losses
    denseGrid_interpolate = ComputeSignal(torch.Tensor(graphSampling.sampling.SH2S))
    denseGrid_interpolate = denseGrid_interpolate.to(DEVICE)

    # Optimizer/Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, threshold=0.01, factor=0.1, patience=3, verbose=True)
    save_loss = {}
    save_loss['train'] = {}
    writer = SummaryWriter(log_dir=os.path.join(data_path, 'result', 'run', save_path.split('/')[-1]))
    tb_j = 0
    # Training loop
    for epoch in range(n_epoch):
        # TRAIN
        model.train()

        # Initialize loss to save and plot.
        loss_intensity_ = 0
        loss_sparsity_ = 0
        loss_non_negativity_fodf_ = 0

        # Train on batch.
        for batch, data in enumerate(dataloader_train):
            # Delete all previous gradients
            optimizer.zero_grad()
            to_print = ''

            # Load the data in the DEVICE
            input = data['input'].to(DEVICE)
            output = data['output'].to(DEVICE)
            mask = data['mask'].to(DEVICE)

            x_reconstructed, x_deconvolved_equi_shc, x_deconvolved_inva_shc = model(input)
            ###############################################################################################
            ###############################################################################################
            # Loss
            ###############################################################################################
            ###############################################################################################
            # Intensity loss
            loss_intensity = intensity_criterion(x_reconstructed, output, mask)
            loss_intensity_ += loss_intensity.item()
            loss = intensity_weight * loss_intensity
            to_print += ', Intensity: {0:.10f}'.format(loss_intensity.item())
            
            if not x_deconvolved_equi_shc  is None:
                x_deconvolved_equi = denseGrid_interpolate(x_deconvolved_equi_shc)
                ###############################################################################################
                # Sparsity loss
                equi_sparse = torch.zeros(x_deconvolved_equi.shape).to(DEVICE)
                loss_sparsity = sparsity_criterion(x_deconvolved_equi, equi_sparse, mask)
                loss_sparsity_ += loss_sparsity.item()
                loss += sparsity_weight * loss_sparsity
                to_print += ', Equi Sparsity: {0:.10f}'.format(loss_sparsity.item())

                ###############################################################################################
                # Non negativity loss
                fodf_neg = torch.min(x_deconvolved_equi, torch.zeros_like(x_deconvolved_equi))
                fodf_neg_zeros = torch.zeros(fodf_neg.shape).to(DEVICE)
                loss_non_negativity_fodf = non_negativity_criterion(fodf_neg, fodf_neg_zeros, mask)
                loss_non_negativity_fodf_ += loss_non_negativity_fodf.item()
                loss += nn_fodf_weight * loss_non_negativity_fodf
                to_print += ', Equi NN: {0:.10f}'.format(loss_non_negativity_fodf.item())

                ###############################################################################################
                # Partial volume regularizer
                regularizer_equi = 0.00001 * 1/torch.mean(x_deconvolved_equi_shc[mask==1][:, :, 0])*np.sqrt(4*np.pi)
                loss += regularizer_equi
                to_print += ', Equi regularizer: {0:.10f}'.format(regularizer_equi.item())
            
            if not x_deconvolved_inva_shc  is None:
                ###############################################################################################
                # Partial volume regularizer
                regularizer_inva = 0.00001 * 1/torch.mean(x_deconvolved_inva_shc[mask==1][:, :, 0])*np.sqrt(4*np.pi)
                loss += regularizer_inva
                to_print += ', Inva regularizer: {0:.10f}'.format(regularizer_inva.item())


            ###############################################################################################
            # Tensorboard
            tb_j += 1
            writer.add_scalar('Batch/train_intensity', loss_intensity.item(), tb_j)
            writer.add_scalar('Batch/train_sparsity', loss_sparsity.item(), tb_j)
            writer.add_scalar('Batch/train_nn', loss_non_negativity_fodf.item(), tb_j)
            writer.add_scalar('Batch/train_total', loss.item(), tb_j)

            ###############################################################################################
            # To print loss
            to_print = 'Epoch [{0}/{1}], Iter [{2}/{3}]: Loss: {4:.10f}'.format(epoch + 1, n_epoch,
                                                                                batch + 1, n_batch,
                                                                                loss.item()) \
                       + to_print
            print(to_print, end="\r")
            ###############################################################################################
            # Loss backward
            loss = loss
            loss.backward()
            optimizer.step()

            if (batch + 1) % 500 == 0:
                torch.save(model.state_dict(), os.path.join(save_path, 'history', 'epoch_{0}.pth'.format(epoch + 1)))

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

        writer.add_scalar('Epoch/train_intensity', loss_intensity_ / n_batch, epoch)
        writer.add_scalar('Epoch/train_sparsity', loss_sparsity_ / n_batch, epoch)
        writer.add_scalar('Epoch/train_nn', loss_non_negativity_fodf_ / n_batch, epoch)
        writer.add_scalar('Epoch/train_total', loss_ / n_batch, epoch)

        ###############################################################################################
        # VALIDATION
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
        with open(os.path.join(save_path, 'history', 'loss.pkl'), 'wb') as f:
            pickle.dump(save_loss, f)
        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'history', 'epoch_{0}.pth'.format(epoch + 1)))
        if early_stop:
            print("Stopped")
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        required=True,
        help='Root path of the data (default: None)',
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
        '--filter_start',
        help='Number of filters for the first convolution (default: 8)',
        default=8,
        type=int
    )
    parser.add_argument(
        '--sh_degree',
        help='Max spherical harmonic order (default: 20)',
        default=20,
        type=int
    )
    parser.add_argument(
        '--kernel_size',
        help='Kernel size (default: 5)',
        default=5,
        type=int
    )
    parser.add_argument(
        '--depth',
        help='Graph subsample depth (default: 5)',
        default=5,
        type=int
    )
    parser.add_argument(
        '--n_side',
        help='Healpix resolution (default: 16)',
        default=16,
        type=int
    )
    parser.add_argument(
        '--save_every',
        help='Saving periodicity (default: 2)',
        default=2,
        type=int
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
        '--rf_name',
        required=True,
        help='Response function folder name (default: None)',
        type=str
    )
    parser.add_argument(
        '--wm',
        required=True,
        action='store_true',
        help='Estimate white matter fODF (default: False)',
    )
    parser.add_argument(
        '--gm',
        action='store_true',
        help='Estimate grey matter fODF (default: False)',
    )
    parser.add_argument(
        '--csf',
        action='store_true',
        help='Estimate CSF fODF (default: False)',
    )
    parser.add_argument(
        '--normalize',
        action='store_false',
        help='Norm the partial volume sum to be 1 (default: True)',
    )
    parser.add_argument(
        '--concatenate',
        action='store_true',
        help='TODO: Concatenate features from a grid (default: False)',
    )
    parser.add_argument(
        '--b_grid',
        default=1,
        help='TODO: Grid size to concatenate (default: 1)',
        type=int
    )
    args = parser.parse_args()
    data_path = args.data_path
    
    # Train properties
    batch_size = args.batch_size
    lr = args.lr
    n_epoch = args.epoch
    
    # Model architecture properties
    filter_start = args.filter_start
    sh_degree = args.sh_degree
    kernel_size = args.kernel_size
    depth = args.depth
    n_side = args.n_side
    normalize = args.normalize

    # Saving parameters
    save_path = os.path.join(data_path, 'result')
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
    rf_name = args.rf_name
    wm = args.wm
    gm = args.gm
    csf = args.csf

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

    main(data_path, batch_size, lr, n_epoch, kernel_size, 
         filter_start, sh_degree, depth, n_side,
         rf_name, wm, gm, csf,
         loss_fn_intensity, loss_fn_non_negativity, loss_fn_sparsity, sigma_sparsity,
         intensity_weight, nn_fodf_weight, sparsity_weight,
         save_path, save_every, normalize, load_state)

