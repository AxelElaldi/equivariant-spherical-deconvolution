import torch
import math


class Conv(torch.nn.Module):
    """Building Block with a Chebyshev Convolution.
    """

    def __init__(self, in_channels, out_channels, lap, kernel_size=3, bias=True):
        """Initialization.
        Args:
            in_channels (int): initial number of channels
            out_channels (int): output number of channels
            lap (:obj:`torch.sparse.FloatTensor`): laplacian
            kernel_size (int): Number of trainable parameters per filter, which is also the size of the convolutional kernel.
                                The order of the Chebyshev polynomials is kernel_size - 1. Defaults to 3.
            bias (bool): Whether to add a bias term.
        """
        super(Conv, self).__init__()
        self.register_buffer("laplacian", lap)
        self.chebconv = ChebConv(in_channels, out_channels, kernel_size, bias)

    def state_dict(self, *args, **kwargs):
        """! WARNING !
        This function overrides the state dict in order to be able to save the model.
        This can be removed as soon as saving sparse matrices has been added to Pytorch.
        """
        state_dict = super().state_dict(*args, **kwargs)
        del_keys = []
        for key in state_dict:
            if key.endswith("laplacian"):
                del_keys.append(key)
        for key in del_keys:
            del state_dict[key]
        return state_dict

    def forward(self, x):
        """Forward pass.
        Args:
            x (:obj:`torch.tensor`): input [B x Fin x V]
        Returns:
            :obj:`torch.tensor`: output [B x Fout x V]
        """
        x = self.chebconv(self.laplacian, x)
        return x


class ChebConv(torch.nn.Module):
    """Graph convolutional layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        """Initialize the Chebyshev layer.
        Args:
            in_channels (int): Number of channels/features in the input graph.
            out_channels (int): Number of channels/features in the output graph.
            kernel_size (int): Number of trainable parameters per filter, which is also the size of the convolutional kernel.
                                The order of the Chebyshev polynomials is kernel_size - 1.
            bias (bool): Whether to add a bias term.
        """
        super(ChebConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._conv = cheb_conv

        shape = (kernel_size, in_channels, out_channels)
        self.weight = torch.nn.Parameter(torch.Tensor(*shape))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.kaiming_initialization()

    def kaiming_initialization(self):
        """Initialize weights and bias.
        """
        std = math.sqrt(2 / (self.in_channels * self.kernel_size))
        self.weight.data.normal_(0, std)
        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def forward(self, laplacian, inputs):
        """Forward graph convolution.
        Args:
            laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
            inputs (:obj:`torch.Tensor`): The current input data being forwarded.
        Returns:
            :obj:`torch.Tensor`: The convoluted inputs.
        """
        outputs = self._conv(laplacian, inputs, self.weight)
        if self.bias is not None:
            outputs += self.bias[None, :, None]
        return outputs


def cheb_conv(laplacian, inputs, weight):
    """Chebyshev convolution.
    Args:
        laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
        inputs (:obj:`torch.Tensor`): The current input data being forwarded.
        weight (:obj:`torch.Tensor`): The weights of the current layer.
    Returns:
        :obj:`torch.Tensor`: Inputs after applying Chebyshev convolution.
    """
    # Get tensor dimensions
    B, Fin, V = inputs.shape
    K, Fin, Fout = weight.shape
    # B = batch size
    # V = nb vertices
    # Fin = nb input features
    # Fout = nb output features
    # K = order of Chebyshev polynomials + 1

    # Transform to Chebyshev basis
    x0 = inputs.permute(2, 1, 0).contiguous()  # V x Fin x B
    x0 = x0.view([V, Fin * B])  # V x Fin*B
    inputs = project_cheb_basis(laplacian, x0, K) # K x V x Fin*B

    # Look at the Chebyshev transforms as feature maps at each vertex
    inputs = inputs.view([K, V, Fin, B])  # K x V x Fin x B
    inputs = inputs.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
    inputs = inputs.view([B * V, Fin * K])  # B*V x Fin*K

    # Linearly compose Fin features to get Fout features
    weight = weight.view(Fin * K, Fout) # K*Fin x Fout
    inputs = inputs.matmul(weight)  # B*V x Fout
    inputs = inputs.view([B, V, Fout])  # B x V x Fout

    # Get final output tensor
    inputs = inputs.permute(0, 2, 1).contiguous()  # B x Fout x V

    return inputs


def project_cheb_basis(laplacian, x0, K):
    """Project vector x on the Chebyshev basis of order K
    \hat{x}_0 = x
    \hat{x}_1 = Lx
    \hat{x}_k = 2*L\hat{x}_{k-1} - \hat{x}_{k-2}
    Args:
        laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
        x0 (:obj:`torch.Tensor`): The initial data being forwarded. [V x D]
        K (:obj:`torch.Tensor`): The order of Chebyshev polynomials + 1.
    Returns:
        :obj:`torch.Tensor`: Inputs after applying Chebyshev projection.
    """
    inputs = x0.unsqueeze(0)  # 1 x V x D
    if K > 1:
        x1 = torch.sparse.mm(laplacian, x0)  # V x D
        inputs = torch.cat((inputs, x1.unsqueeze(0)), 0)  # 2 x V x D
        for _ in range(2, K):
            x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
            inputs = torch.cat((inputs, x2.unsqueeze(0)), 0)  # _ x V x D
            x0, x1 = x1, x2
    return inputs # K x V x D
