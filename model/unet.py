import torch.nn as nn
import torch
from .graphconv import Conv

class GraphCNNUnet(nn.Module):
    """GCNN Autoencoder.
    """
    def __init__(self, in_channels, out_channels, filter_start, kernel_size, pooling, laps):
        """Initialization.
        Args:
            in_channels (int): Number of input channel
            out_channels (int): Number of output channel
            filter_start (int): Number of feature channel after first convolution. Then, multiplied by 2 after every poolig
            kernel_size (int): Size of the kernel (i.e. Order of the Chebyshev polynomials + 1)
            pooling (:obj:'PoolingClass'): Pooling operator
            laps (list): Increasing list of laplacians from smallest to largest resolution, D: size of the list
        """
        super(GraphCNNUnet, self).__init__()
        self.encoder = Encoder(in_channels, filter_start, kernel_size, pooling.pooling, laps)
        self.decoder = Decoder(out_channels, filter_start, kernel_size, pooling.unpooling, laps)

    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x in_channels x V]
        Returns:
            :obj:`torch.Tensor`: output [B x out_channels x V]
        """
        x, enc_ftrs = self.encoder(x)
        x = self.decoder(x, enc_ftrs)
        return x


class Encoder(nn.Module):
    """GCNN Encoder.
    """
    def __init__(self, in_channels, filter_start, kernel_size, pooling, laps):
        """Initialization.
        Args:
            in_channels (int): Number of input channel
            filter_start (int): Number of feature channel after first convolution. Then, multiplied by 2 after every poolig
            kernel_size (int): Size of the kernel (i.e. Order of the Chebyshev polynomials + 1)
            pooling (:obj:'PoolingClass'): Pooling operator
            laps (list): Increasing list of laplacians from smallest to largest resolution, D: size of the list
        """
        super(Encoder, self).__init__()
        D = len(laps)
        assert D > 1 # Otherwise there is no encoding/decoding to perform
        self.enc_blocks = [Block(in_channels, filter_start, filter_start, laps[-1], kernel_size)]
        self.enc_blocks += [Block((2**i)*filter_start, (2**(i+1))*filter_start, (2**(i+1))*filter_start, laps[-i-2], kernel_size) for i in range(D-2)]
        self.enc_blocks = nn.ModuleList(self.enc_blocks)
        self.pool = pooling
    
    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): Input to be forwarded. [B x in_channels x V]          
        Returns:
            :obj:`torch.Tensor`: output [B x (2**(D-2))*filter_start x V_encoded]
            encoder_features (list): Hierarchical encoding. [B x (2**(i))*filter_start x V_encoded_i] for i in [0,D-2]
        """
        ftrs = []
        for block in self.enc_blocks: # len(self.enc_blocks) = D - 2
            x = block(x) # B x (2**(i))*filter_start x V_encoded_i
            ftrs.append(x) 
            x, _ = self.pool(x) # B x (2**(i))*filter_start x V_encoded_(i+1)
        return x, ftrs


class Decoder(nn.Module):
    """GCNN Decoder.
    """
    def __init__(self, out_channels, filter_start, kernel_size, unpooling, laps):
        """Initialization.
        Args:
            out_channels (int): Number of output channel
            filter_start (int): Number of feature channel after first convolution. Then, multiplied by 2 after every poolig
            kernel_size (int): Size of the kernel (i.e. Order of the Chebyshev polynomials + 1)
            pooling (:obj:'PoolingClass'): Pooling operator
            laps (list): Increasing list of laplacians from smallest to largest resolution, D: size of the list
        """
        super(Decoder, self).__init__()
        D = len(laps)
        assert D > 1 # Otherwise there is no encoding/decoding to perform
        self.dec_blocks = [Block((2**(D-2))*filter_start, (2**(D-1))*filter_start, (2**(D-2))*filter_start, laps[0], kernel_size)]
        self.dec_blocks += [Block((2**(D-i))*filter_start, (2**(D-i-1))*filter_start, (2**(D-i-2))*filter_start, laps[i], kernel_size) for i in range(1, D-1)]
        self.dec_blocks += [Block(2*filter_start, filter_start, filter_start, laps[-1], kernel_size)]
        self.dec_blocks = nn.ModuleList(self.dec_blocks)
        self.head = Conv(filter_start, out_channels, laps[-1], kernel_size)
        self.activation = nn.ReLU()
        self.unpool = unpooling

    def forward(self, x, encoder_features):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): Input to be forwarded. [B x (2**(D-2))*filter_start x V_encoded_(D-1)]
            encoder_features (list): Hierarchical encoding to be forwarded. [B x (2**(i))*filter_start x V_encoded_i] for i in [0,D-2]
        Returns:
            :obj:`torch.Tensor`: output [B x V x out_channels]
        """
        x = self.dec_blocks[0](x) # B x (2**(D-2))*filter_start x V_encoded_(D-1)
        x = self.unpool(x, None) # B x (2**(D-2))*filter_start x V_encoded_(D-2)
        x = torch.cat([x, encoder_features[-1]], dim=1) # B x 2*(2**(D-2))*filter_start x V_encoded_(D-2)
        for i in range(1, len(self.dec_blocks)-1):
            x = self.dec_blocks[i](x) # B x (2**(D-i-2))*filter_start x V_encoded_(D-i-1)
            x = self.unpool(x, None) # B x (2**(D-i-2))*filter_start x V_encoded_(D-i-2)
            x = torch.cat([x, encoder_features[-1-i]], dim=1) # B x 2*(2**(D-i-2))*filter_start x V_encoded_(D-i-2)
        x = self.dec_blocks[-1](x) # B x filter_start x V
        x = self.activation(self.head(x)) # B x out_channels x V
        return x
    

class Block(nn.Module):
    """GCNN Unet block.
    """
    def __init__(self, in_ch, int_ch, out_ch, lap, kernel_size):
        """Initialization.
        Args:
            in_ch (int): Number of input channel
            int_ch (int): Number of intermediate channel
            out_ch (int): Number of output channel
            lap (list): Increasing list of laplacians from smallest to largest resolution
            kernel_size (int): Size of the kernel (i.e. Order of the Chebyshev polynomials + 1)
        """
        super(Block, self).__init__()
        # Conv 1
        self.conv1 = Conv(in_ch, int_ch, lap, kernel_size)
        self.bn1 = nn.BatchNorm1d(int_ch)
        # Conv 2
        self.conv2 = Conv(int_ch, out_ch, lap, kernel_size)
        self.bn2 = nn.BatchNorm1d(out_ch)
        # Activation
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x F_in_ch x V]
        Returns:
            :obj:`torch.Tensor`: output [B x F_out_ch x V]
        """
        x = self.activation(self.bn1(self.conv1(x))) # B x F_int_ch x V
        x = self.activation(self.bn2(self.conv2(x))) # B x F_out_ch x V
        return x
