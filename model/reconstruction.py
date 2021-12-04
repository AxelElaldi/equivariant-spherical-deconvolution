import torch
import math
from .shutils import ShellComputeSignal


class Reconstruction(torch.nn.Module):
    """Building Block for spherical harmonic convolution with a polar filter
    """

    def __init__(self, polar_filter_equi, polar_filter_inva, shellSampling):
        """Initialization.
        Args:
            polar_filter (:obj:`torch.Tensor`): [in_channel x S x L] Polar filter spherical harmonic coefficients
            polar_filter_inva (:obj:`torch.Tensor`): [in_channel x S x 1] Polar filter spherical harmonic coefficients
            shellSampling (:obj:`sampling.ShellSampling`): Input sampling scheme
        """
        super(Reconstruction, self).__init__()
        if polar_filter_equi is None:
            self.equi = False
        else:
            self.conv_equi = IsoSHConv(polar_filter_equi)
            self.equi = True

        if polar_filter_inva is None:
            self.inva = False
        else:
            self.conv_inv = IsoSHConv(polar_filter_inva)
            self.inva = True

        self.get_signal = ShellComputeSignal(shellSampling)

    def forward(self, x_equi_shc, x_inva_shc):
        """Forward pass.
        Args:
            x_equi_shc (:obj:`torch.tensor`): [B x equi_channel x C] Signal spherical harmonic coefficients.
            x_inva_shc (:obj:`torch.tensor`): [B x inva_channel x 1] Signal spherical harmonic coefficients.
        Returns:
            :obj:`torch.tensor`: [B x V] Reconstruction of the signal
        """
        x_convolved_equi, x_convolved_inva = 0, 0
        if self.equi:
            x_convolved_equi_shc = self.conv_equi(x_equi_shc) # B x equi_channel x S x C
            x_convolved_equi = self.get_signal(x_convolved_equi_shc) # B x equi_channel x V
            x_convolved_equi = torch.sum(x_convolved_equi, axis=1) # B x V
        if self.inva:
            x_convolved_inva_shc = self.conv_inv(x_inva_shc) # B x inva_channel x S x 1
            x_convolved_inva = self.get_signal(x_convolved_inva_shc) # B x inva_channel x V
            x_convolved_inva = torch.sum(x_convolved_inva, axis=1) # B x V
        # Get reconstruction
        x_reconstructed =  x_convolved_equi + x_convolved_inva

        return x_reconstructed


class IsoSHConv(torch.nn.Module):
    """Building Block for spherical harmonic convolution with a polar filter
    """

    def __init__(self, polar_filter):
        """Initialization.
        Args:
            polar_filter (:obj:`torch.Tensor`): [in_channel x S x L] Polar filter spherical harmonic coefficients
        where in_channel is the number of tissue, S is the number of shell and L is the number of odd spherical harmonic order
        C is the number of coefficients for the L odd spherical harmonic order
        """
        super().__init__()
        # We need to multiply each coefficient order by sqrt(4pi/(4l+1)) and copy it 2*l+1 times
        polar_filter = self.construct_filter(polar_filter) # 1 x in_channel x S x C
        self.register_buffer("polar_filter", polar_filter)

    def construct_filter(self, filter):
        """Reformate the polar filter (scale and extand the coefficients).
        Args:
            polar_filter (:obj:`torch.Tensor`): [in_channel x S x L] Polar filter spherical harmonic coefficients
        Returns:
            :obj:`torch.tensor`: [1 x in_channel x S x C] Extanded polar filter spherical harmonic coefficients
        """
        L = filter.shape[2]
        # Scale by sqrt(4*pi/4*l+1))
        scale = torch.Tensor([math.sqrt(4*math.pi/(4*l+1)) for l in range(L)])[None, None, :] # 1 x 1 x L
        filter = scale*filter # in_channel x S x L
        # Repeat each coefficient 4*l+1 times
        repeat = torch.Tensor([int(4*l+1) for l in range(L)]).type(torch.int64) # L
        filter = filter.repeat_interleave(repeat, dim=2) # in_channel x S x C
        # Add the first dimension for multiplication convenience
        filter = filter[None, :, :, :] # 1 x in_channel x S x C
        return filter

    def forward(self, x):
        """Forward pass.
        Args:
            x (:obj:`torch.tensor`): [B x in_channel x C] Signal spherical harmonic coefficients.
        Returns:
            :obj:`torch.tensor`: [B x in_channel x S x C] Spherical harmonic coefficient of the output
        """
        x = x[:, :, None, :] # B x in_channel x 1 x C
        x = x*self.polar_filter # B x in_channel x S x C
        return x
