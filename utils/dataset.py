from torch.utils.data import Dataset
import nibabel as nib
import torch
import os
import numpy as np


class DMRIDataset(Dataset):
    def __init__(self, data_path, mask_path):
        self.data = nib.load(data_path)
        self.affine = self.data.affine
        self.header = self.data.header
        self.data = torch.Tensor(self.data.get_fdata()) # Load image X x Y x Z x V
        # Load mask
        if os.path.isfile(mask_path):
            self.mask = torch.Tensor(nib.load(mask_path).get_fdata()) # X x Y x Z
        else:
            self.mask = torch.ones(self.data.shape[:-1])
        # Save the non-null index of the mask
        ind = np.arange(self.mask.nelement())[torch.flatten(self.mask) != 0]
        self.x, self.y, self.z = np.unravel_index(ind, self.mask.shape)
        self.N = len(self.x)
        print('Dataset size: {}'.format(self.N))
        print('Padded scan size: {}'.format(self.data.shape))
        print('Padded mask size: {}'.format(self.mask.shape))

    def __len__(self):
        return int(self.N)

    def __getitem__(self, i):
        input = self.data[None, self.x[i], self.y[i], self.z[i]] # 1 x V
        output = self.data[self.x[i], self.y[i], self.z[i]] # V
        mask = self.mask[self.x[i], self.y[i], self.z[i]] # 1

        return {'sample_id': i, 'input': input, 'output': output, 'mask': mask}
