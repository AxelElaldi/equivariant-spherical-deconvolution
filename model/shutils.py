import torch


class ComputeSHC(torch.nn.Module):
    """Extract the spherical harmonic coefficient.
    """
    def __init__(self, S2SH):
        """Initialization.
        Args:
            S2SH (:obj:`torch.Tensor`): [V x C] Signal to spherical harmomic matrix
        """
        super(ComputeSHC, self).__init__()
        self.register_buffer('S2SH', S2SH)

    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x in_channels x V]
        Returns:
            :obj:`torch.Tensor`: output [B x in_channels x C]
        """
        x = x.matmul(self.S2SH) # B x in_channels x C
        return x


class ComputeSignal(torch.nn.Module):
    """Extract the signal.
    """
    def __init__(self, SH2S):
        """Initialization.
        Args:
            2 (:obj:`torch.Tensor`): [C x V] Spherical harmomic to signal matrix
        """
        super(ComputeSignal, self).__init__()
        self.register_buffer('SH2S', SH2S)

    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x in_channels x C]
        Returns:
            :obj:`torch.Tensor`: output [B x in_channels x V]
        """
        x = x.matmul(self.SH2S) # B x in_channels x V
        return x


class ShellComputeSignal(torch.nn.Module):
    """Extract the spherical harmonic coefficient per shell.
    """
    def __init__(self, shellSampling):
        """Initialization.
        Args:
            shellSampling (:obj:`sampling.ShellSampling`): Shell sampling object
        """
        super(ShellComputeSignal, self).__init__()
        self.shellSampling = shellSampling
        self.S = len(shellSampling.shell_values)
        self.V = shellSampling.vectors.shape[0]
        for i, sampling in enumerate(shellSampling.sampling):
            self.register_buffer(f'SH2S_{i}', torch.Tensor(sampling.SH2S))
        
    def SH2S(self, i):
        return self.__getattr__('SH2S_'+str(i)) # C x V_i

    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x in_channels x S x C]
        Returns:
            :obj:`torch.Tensor`: output [B x in_channels x V]
        """
        y = x.new_zeros((x.shape[0], x.shape[1], self.V)) # B x in_channels x V
        C = x.shape[3]
        for i, sampling in enumerate(self.shellSampling.sampling):
            y[:, :, self.shellSampling.shell_inverse == i] = x[:, :, i].matmul(self.SH2S(i)[:C]) # B x in_channels x V_i
        return y


class ShellComputeSHC(torch.nn.Module):
    """Extract the spherical harmonic coefficient per shell.
    """
    def __init__(self, shellSampling):
        """Initialization.
        Args:
            shellSampling (:obj:`sampling.ShellSampling`): Shell sampling object
        """
        super(ShellComputeSHC, self).__init__()
        self.shellSampling = shellSampling
        self.S = len(self.shellSampling)
        for i, sampling in enumerate(shellSampling):
            self.register_buffer(f'S2SH_{i}', torch.Tensor(sampling.S2SH))
        self.C = sampling.S2SH.shape[1]
        
    def S2SH(self, i):
        return self.__getattr__('S2SH_'+str(i))

    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x in_channels x V]
        Returns:
            :obj:`torch.Tensor`: output [B x in_channels x S x C]
        """
        y = x.new_zeros((x.shape[0], x.shape[1], self.S, self.C)) # B x in_channels x S x C
        for i, sampling in enumerate(self.shellSampling.sampling):
            y[:, :, i] = x[:, :, self.shellSampling.shell_inverse == i].matmul(self.S2SH(i)) # B x in_channels x C
        return y
