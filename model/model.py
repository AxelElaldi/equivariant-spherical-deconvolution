import torch
from .deconvolution import Deconvolution
from .reconstruction import Reconstruction


class Model(torch.nn.Module):
    def __init__(self, polar_filter_equi, polar_filter_inva, shellSampling, graphSampling, filter_start, kernel_size, normalize):
        super(Model, self).__init__()
        
        n_equi = polar_filter_equi.shape[0]
        n_inva = polar_filter_inva.shape[0]
        self.deconvolution = Deconvolution(shellSampling, graphSampling, filter_start, kernel_size, n_equi, n_inva, normalize)
        self.reconstruction = Reconstruction(polar_filter_equi, polar_filter_inva, shellSampling)

    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x in_channels x V']
        Returns:
            :obj:`torch.Tensor`: output [B x V']
            :obj:`torch.Tensor`: output [B x out_channels_equi x C]
            :obj:`torch.Tensor`: output [B x out_channels_inva x 1]
        """
        # Deconvolve the signal and get the spherical harmonic coefficients
        x_deconvolved_equi_shc, x_deconvolved_inva_shc = self.deconvolution(x) # B x out_channels_equi x C , B x out_channels_inva x 1

        # Reconstruct the signal
        x_reconstructed = self.reconstruction(x_deconvolved_equi_shc, x_deconvolved_inva_shc) # B x V'

        return x_reconstructed, x_deconvolved_equi_shc, x_deconvolved_inva_shc
