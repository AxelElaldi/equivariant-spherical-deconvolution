import argparse
import os
import numpy as np
import nibabel as nib
import json

from utils.sampling import HealpixSampling, ShellSampling
from utils.dataset import DMRIDataset
from utils.response import load_response_function
from model.model import Model

import torch
from torch.utils.data.dataloader import DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(data_path, batch_size, kernel_size, 
         filter_start, sh_degree, depth, n_side,
         rf_name, wm, gm, csf,
         normalize, model_name, epoch):
    """Test a model
    Args:
        data_path (str): Data path
        batch_size (int): Batch size
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
        normalize (bool): Normalize the fODFs
        load_state (str): Load pre trained network
        model_name (str): Name of the model folder
        epoch (int): Epoch to use for testing
    """
    
    # Load the shell and the graph samplings
    shellSampling = ShellSampling(f'{data_path}/bvecs.bvecs', f'{data_path}/bvals.bvals', sh_degree=sh_degree, max_sh_degree=8)
    graphSampling = HealpixSampling(n_side, depth, sh_degree=sh_degree)

    # Load the image and the mask
    dataset = DMRIDataset(f'{data_path}/features.nii', f'{data_path}/mask.nii')
    dataloader_test = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    n_batch = len(dataloader_test)
    
    # Load the Polar filter used for the deconvolution
    polar_filter_equi, polar_filter_inva = load_response_function(f'{data_path}/response_functions/{rf_name}', wm=wm, gm=gm, csf=csf, max_degree=sh_degree, n_shell=len(shellSampling.shell_values))

    # Create the deconvolution model and load the trained model
    model = Model(polar_filter_equi, polar_filter_inva, shellSampling, graphSampling, filter_start, kernel_size, normalize)
    model.load_state_dict(torch.load(f'{data_path}/result/{model_name}/history/epoch_{epoch}.pth'), strict=False)
    # Load model in GPU
    model = model.to(DEVICE)

    # Output initialization
    nb_coef = int((sh_degree + 1) * (sh_degree / 2 + 1))
    reconstruction_list = np.zeros((dataset.data.shape[0],
                                    dataset.data.shape[1],
                                    dataset.data.shape[2], len(shellSampling.vectors)))
    if wm:
        fodf_shc_wm_list = np.zeros((dataset.data.shape[0],
                                     dataset.data.shape[1],
                                     dataset.data.shape[2], nb_coef))
    if gm:
        fodf_shc_gm_list = np.zeros((dataset.data.shape[0],
                                     dataset.data.shape[1],
                                     dataset.data.shape[2], 1))
    if csf:
        fodf_shc_csf_list = np.zeros((dataset.data.shape[0],
                                      dataset.data.shape[1],
                                      dataset.data.shape[2], 1))
    # Test on batch.
    for i, data in enumerate(dataloader_test):
        print(str(i * 100 / n_batch) + " %", end='\r')
        # Load the data in the DEVICE
        input = data['input'].to(DEVICE)
        sample_id = data['sample_id']

        x_reconstructed, x_deconvolved_equi_shc, x_deconvolved_inva_shc = model(input)
        for j in range(len(input)):
            sample_id_j = sample_id[j]
            reconstruction_list[dataset.x[sample_id_j],
                                dataset.y[sample_id_j],
                                dataset.z[sample_id_j]] += x_reconstructed[j].cpu().detach().numpy()
            if wm:
                fodf_shc_wm_list[dataset.x[sample_id_j],
                                 dataset.y[sample_id_j],
                                 dataset.z[sample_id_j]] += x_deconvolved_equi_shc[j, 0].cpu().detach().numpy()
            index = 0
            if gm:
                fodf_shc_gm_list[dataset.x[sample_id_j],
                                 dataset.y[sample_id_j],
                                 dataset.z[sample_id_j]] += x_deconvolved_inva_shc[j, index].cpu().detach().numpy()
                index += 1
            if csf:
                fodf_shc_csf_list[dataset.x[sample_id_j],
                                  dataset.y[sample_id_j],
                                  dataset.z[sample_id_j]] += x_deconvolved_inva_shc[j, index].cpu().detach().numpy()

    # Save the results
    reconstruction_list = np.array(reconstruction_list).astype(np.float32)
    img = nib.Nifti1Image(reconstruction_list, dataset.affine, dataset.header)
    nib.save(img, f'{data_path}/result/{model_name}/test/epoch_{epoch}/reconstruction.nii')
    if wm:
        fodf_shc_wm_nii = np.array(fodf_shc_wm_list).astype(np.float32)
        img = nib.Nifti1Image(fodf_shc_wm_nii, dataset.affine, dataset.header)
        nib.save(img, f'{data_path}/result/{model_name}/test/epoch_{epoch}/fodf.nii')
    if gm:
        fodf_shc_gm_nii = np.array(fodf_shc_gm_list).astype(np.float32)
        img = nib.Nifti1Image(fodf_shc_gm_nii, dataset.affine, dataset.header)
        nib.save(img, f'{data_path}/result/{model_name}/test/epoch_{epoch}/fodf_gm.nii')
    if csf:
        fodf_shc_csf_nii = np.array(fodf_shc_csf_list).astype(np.float32)
        img = nib.Nifti1Image(fodf_shc_csf_nii, dataset.affine, dataset.header)
        nib.save(img, f'{data_path}/result/{model_name}/test/epoch_{epoch}/fodf_csf.nii')


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
        '--model_name',
        required=True,
        help='Epoch (default: None)',
        type=str
    )
    parser.add_argument(
        '--epoch',
        required=True,
        help='Epoch (default: None)',
        type=int
    )
    args = parser.parse_args()
    # Test properties
    batch_size = args.batch_size
    
    # Data path
    data_path = args.data_path
    assert os.path.exists(data_path)

    # Load trained model
    model_name = args.model_name
    assert os.path.exists(f'{data_path}/result/{model_name}')
    epoch = args.epoch
    assert os.path.exists(f'{data_path}/result/{model_name}/history/epoch_{epoch}.pth')

    # Load parameters
    with open(f'{data_path}/result/{model_name}/args.txt', 'r') as file:
        args_json = json.load(file)
    
    # Model architecture properties
    filter_start = int(args_json['filter_start'])
    sh_degree = int(args_json['sh_degree'])
    kernel_size = int(args_json['kernel_size'])
    depth = int(args_json['depth'])
    n_side = int(args_json['n_side'])
    normalize = bool(args_json['normalize'])

    # Load response functions
    rf_name = str(args_json['rf_name'])
    wm = bool(args_json['wm'])
    gm = bool(args_json['gm'])
    csf = bool(args_json['csf'])

    # Test directory
    test_path = f'{data_path}/result/{model_name}/test/epoch_{epoch}'
    if not os.path.exists(test_path):
        print('Create new directory: {0}'.format(test_path))
        os.makedirs(test_path)


    main(data_path, batch_size, kernel_size, 
         filter_start, sh_degree, depth, n_side,
         rf_name, wm, gm, csf,
         normalize, model_name, epoch)

