import torch.nn as nn
import torch.nn.functional as F


class Healpix:
    """Healpix class, which groups together the corresponding pooling and unpooling.
    """

    def __init__(self, mode="average"):
        """Initialize healpix pooling and unpooling objects.
        Args:
            mode (str, optional): specify the mode for pooling/unpooling.
                                    Can be maxpooling or averagepooling. Defaults to 'average'.
        """
        if mode == "max":
            self.__pooling = HealpixMaxPool()
            self.__unpooling = HealpixMaxUnpool()
        else:
            self.__pooling = HealpixAvgPool()
            self.__unpooling = HealpixAvgUnpool()

    @property
    def pooling(self):
        """Get pooling
        """
        return self.__pooling

    @property
    def unpooling(self):
        """Get unpooling
        """
        return self.__unpooling

## Max Pooling/Unpooling

class HealpixMaxPool(nn.MaxPool1d):
    """Healpix Maxpooling module
    """

    def __init__(self):
        """Initialization
        """
        super().__init__(kernel_size=4, return_indices=True)

    def forward(self, x):
        """Forward call the 1d Maxpooling of pytorch
        Args:
            x (:obj:`torch.tensor`):[B x Fin x V]
        Returns:
            tuple((:obj:`torch.tensor`), indices (int)): [B x Fin x V_pool] and indices of pooled pixels
        """
        x, indices = F.max_pool1d(x, self.kernel_size, return_indices=True) # B x Fin x V_pool
        return x, indices


class HealpixMaxUnpool(nn.MaxUnpool1d):
    """Healpix Maxunpooling using the MaxUnpool1d of pytorch
    """

    def __init__(self):
        """initialization
        """
        super().__init__(kernel_size=4)

    def forward(self, x, indices):
        """calls MaxUnpool1d using the indices returned previously by HealpixMaxPool
        Args:
            tuple(x (:obj:`torch.tensor`) : [B x Fin x V]
            indices (int)): indices of pixels equiangular maxpooled previously
        Returns:
            [:obj:`torch.tensor`] -- [B x Fin x V_unpool]
        """
        x = F.max_unpool1d(x, indices, self.kernel_size) # B x Fin x V_unpool
        return x


## Avereage Pooling/Unpooling

class HealpixAvgPool(nn.AvgPool1d):
    """Healpix Average pooling module
    """

    def __init__(self):
        """initialization
        """
        super().__init__(kernel_size=4)

    def forward(self, x):
        """forward call the 1d Averagepooling of pytorch
        Arguments:
            x (:obj:`torch.tensor`): [B x Fin x V]
        Returns:
            tuple((:obj:`torch.tensor`), indices (None)): [B x Fin x V_pool] and indices for consistence
            with maxPool
        """
        x = F.avg_pool1d(x, self.kernel_size) # B x Fin x V_pool
        return x, None


class HealpixAvgUnpool(nn.Module):
    """Healpix Average Unpooling module
    """

    def __init__(self):
        """initialization
        """
        self.kernel_size = 4
        super().__init__()

    def forward(self, x, indices):
        """forward repeats (here more like a numpy tile for the moment) the incoming tensor
        Arguments:
            x (:obj:`torch.tensor`), indices (None): [B x Fin x V] and indices for consistence with maxUnPool
        Returns:
            [:obj:`torch.tensor`]: [B x Fin x V_unpool]
        """
        x = F.interpolate(x, scale_factor=self.kernel_size, mode="nearest") # B x Fin x V_unpool
        return x
