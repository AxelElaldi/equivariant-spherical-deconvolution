import torch


class Interpolation(torch.nn.Module):
    def __init__(self, shellSampling, gridSampling):
        """Initialization.
        Args:
            shellSampling (:obj:`sampling.ShellSampling`): Input sampling scheme
            gridSampling (:obj:`sampling.Sampling`): Interpolation grid scheme
        """
        super(Interpolation, self).__init__()
        self.shellSampling = shellSampling
        self.gridSampling = gridSampling
        self.S = len(shellSampling.sampling)
        SH2S = self.gridSampling.sampling.SH2S
        C_grid = SH2S.shape[0]
        self.V = SH2S.shape[1]
        for i, sampling in enumerate(self.shellSampling.sampling):
            S2SH = sampling.S2SH
            C = S2SH.shape[1]
            assert C <= C_grid
            self.register_buffer(f'sampling2sampling_{i}', torch.Tensor(S2SH.dot(SH2S[:C])))

    def sampling2sampling(self, i):
        return self.__getattr__('sampling2sampling_'+str(i)) # C x V_i

    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x in_channels x V']
        Returns:
            :obj:`torch.Tensor`: interpolated input [B x in_channels*S x V]
        """
        B, F_in, _ = x.shape
        y = x.new_zeros(B, F_in*self.S, self.V) # B x in_channels' x V
        for i, sampling in enumerate(self.shellSampling.sampling):
            x_shell = x[:, :, self.shellSampling.shell_inverse == i] # B x in_channels x V_i
            y[:, i*F_in:(i+1)*F_in] = x_shell.matmul(self.sampling2sampling(i)) # B x in_channels x V
        return y
