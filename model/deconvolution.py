import torch
import math
from .unet import GraphCNNUnet
from .shutils import ComputeSHC
from .interpolation import Interpolation

class Deconvolution(torch.nn.Module):
    def __init__(self, shellSampling, graphSampling, filter_start, kernel_size, n_equi, n_inva, normalize):
        """Separate equivariant and invariant features from the deconvolved model
        Args:
            x (:obj:`torch.Tensor`): input. [B x V x out_channels]
            shellSampling (:obj:`sampling.ShellSampling`): Input sampling scheme
            graphSampling (:obj:`sampling.Sampling`): Interpolation grid scheme
            filter_start (int): First intermediate channel (then multiply by 2)
            kernel_size (int): Number of trainable parameters per filter, which is also the size of the convolutional kernel.
                                The order of the Chebyshev polynomials is kernel_size - 1.
            n_equi (int): Number of equivariant deconvolved channel
            n_inva (int): Number of invariant deconvolved channel
            normalize (bool): Normalize the output such that the sum of the SHC of order and degree 0 of the deconvolved channels is math.sqrt(4 * math.pi)
        """
        super(Deconvolution, self).__init__()
        pooling = graphSampling.pooling
        laps = graphSampling.laps
        in_channels = len(shellSampling.shell_values)
        self.n_equi = n_equi
        self.n_inva = n_inva
        self.normalize = normalize
        self.deconvolve = GraphCNNUnet(in_channels, n_equi + n_inva, filter_start, kernel_size, pooling, laps)
        self.get_equi_sh = ComputeSHC(torch.Tensor(graphSampling.sampling.S2SH))
        self.interpolate = Interpolation(shellSampling, graphSampling)
        self.eps = 1e-16

    def separate(self, x):
        """Separate equivariant and invariant features from the deconvolved model
        Args:
            x (:obj:`torch.Tensor`): input. [B x out_channels x V]
        Returns:
            x_equi (:obj:`torch.Tensor`): equivariant part of the deconvolution [B x out_channels_equi x V]
            x_inva (:obj:`torch.Tensor`): invariant part of the deconvolution [B x out_channels_inva]
        """
        if self.n_equi != 0:
            x_equi = x[:, :self.n_equi]
        else:
            x_equi = None
        if self.n_inva != 0:
            x_inva = x[:, self.n_equi:]
            x_inva = torch.max(x_inva, dim=2)[0]
        else:
            x_inva = None
        return x_equi, x_inva

    def norm(self, x_equi, x_inva):
        """Separate equivariant and invariant features from the deconvolved model
        Args:
            x_equi (:obj:`torch.Tensor`): shc equivariant part of the deconvolution [B x out_channels_equi x C]
            x_inva (:obj:`torch.Tensor`): shc invariant part of the deconvolution [B x out_channels_inva x 1]
        Returns:
            x_equi (:obj:`torch.Tensor`): normed shc equivariant part of the deconvolution [B x out_channels_equi x C]
            x_inva (:obj:`torch.Tensor`): normed shc invariant part of the deconvolution [B x out_channels_inva x 1]
        """
        to_norm = 0
        if self.n_equi != 0:
            to_norm = to_norm + torch.sum(x_equi[:, :, 0:1], axis=1, keepdim=True)
        if self.n_inva != 0:
            to_norm = to_norm + torch.sum(x_inva, axis=1, keepdim=True)
        to_norm = to_norm * math.sqrt(4 * math.pi) + self.eps
        if self.n_equi != 0:
            x_equi = x_equi / to_norm
        if self.n_inva != 0:
            x_inva = x_inva / to_norm
        return x_equi, x_inva


    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x in_channels x V']
        Returns:
            x_deconvolved_equi_shc (:obj:`torch.Tensor`): SHC equivariant part of the deconvolution [B x out_channels_equi x C]
            x_deconvolved_inva_shc (:obj:`torch.Tensor`): SHC invariant part of the deconvolution [B x out_channels_inva x 1]
        """
        # Interpolation of the input signal on the Graph used for the graph convolution
        # in_channels' can be different than in_channels, for example in the case multi-shell S : in_channels' = in_channels * S
        x = self.interpolate(x) # B x in_channels' x V

        # Deconvolve the input signal (compute the fODFs)
        x_deconvolved = self.deconvolve(x) # B x out_channels x V

        # Separate invariant and equivariant to rotation channels (separate white matter (equivariant) and CSF + gray matter (invariant))
        x_deconvolved_equi, x_deconvolved_inva = self.separate(x_deconvolved) # B x out_channels_equi x V, B x out_channels_inva
        
        # Symmetrized and get the spherical harmonic coefficients of the equivariant channels
        if self.n_equi != 0:
            x_deconvolved_equi_shc = self.get_equi_sh(x_deconvolved_equi) # B x out_channels_equi x C
        else:
            x_deconvolved_equi_shc = None

        # Get the spherical harmonic coefficients of the invariant channels
        if self.n_inva != 0:
            x_deconvolved_inva_shc = (x_deconvolved_inva * math.sqrt(4 * math.pi))[:, :, None] # B x out_channels_inva x 1
        else:
            x_deconvolved_inva_shc = None
        
        # Normalize
        if self.normalize:
            x_deconvolved_equi_shc, x_deconvolved_inva_shc = self.norm(x_deconvolved_equi_shc, x_deconvolved_inva_shc)

        return x_deconvolved_equi_shc, x_deconvolved_inva_shc