import torch


class Loss(torch.nn.Module):
    def __init__(self, norm, sigma=None):
        """
        Parameters
        ----------
        norm : str
            Name of the loss.
        sigma : float
            Hyper parameter of the loss.
        """
        super(Loss, self).__init__()
        if norm not in ['L2', 'L1', 'cauchy', 'welsch', 'geman']:
            raise NotImplementedError('Expected L1, L2, cauchy, welsh, geman but got {}'.format(norm))
        if sigma is None and norm in ['cauchy', 'welsch', 'geman']:
            raise NotImplementedError('Expected a loss hyper parameter for {}'.format(norm))
        self.norm = norm
        self.sigma = sigma

    def forward(self, img1, img2, mask, wts=None):
        """
        Parameters
        ----------
        img1 : torch.Tensor
            Prediction tensor
        img2 : torch.Tensor
            Ground truth tensor
        wts: torch.nn.Parameter
            If specified, the weight of the grid.
        Returns
        -------
         loss : torch.Tensor
            Loss of the predicted tensor
        """
        img1 = img1[mask==1]
        img2 = img2[mask==1]
        if self.norm == 'L2':
            out = (img1 - img2)**2
        elif self.norm == 'L1':
            out = torch.abs(img1 - img2)
        elif self.norm == 'cauchy':
            out = 2 * torch.log(1 + ((img1 - img2)**2 / (2*self.sigma)))
        elif self.norm == 'welsch':
            out = 2 * (1-torch.exp(-0.5 * ((img1 - img2)**2 / self.sigma)))
        elif self.norm == 'geman':
            out = 2 * (2*((img1 - img2)**2 / self.sigma) / ((img1 - img2)**2 / self.sigma + 4))
        else:
            raise ValueError('Expected L1, L2, cauchy, welsh, geman but got {}'.format(self.norm))

        if wts is not None:
            out = out * wts
        loss = out.mean()
        return loss