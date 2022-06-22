from torch.utils.data import Dataset
import h5py
import numpy as np
from torchvision import transforms
import random


class Hdf5Dataset(Dataset):
    def __init__(self, hdf5_path, idx, augment=True, output_mask=True, normalize=False):
        self.inf_only = False

        self.idx_map = idx

        self.hdf5_path = hdf5_path

        self.augment = augment

        self.output_mask = output_mask

        file_handle = h5py.File(self.hdf5_path, 'r')

        if type(idx) != np.ndarray:
            self.idx_map = np.arange(file_handle["nDSM/A"].shape[0])

        self.sampled_idx = np.empty((0))

        self.keys = []

        self.traverse_hdf5(file_handle, '/', self.keys)

        self.keys = [x for x in self.keys if "IDX" not in x]
        

        self.dsm_mask_transform = [transforms.ToTensor()]
        self.ortho_transform = [transforms.ToTensor()]
        if normalize:
            self.ortho_transform.append(transforms.Normalize((0, 0, 0), (1, 1, 1)))

        self.dsm_mask_transform = transforms.Compose(self.dsm_mask_transform)
        self.ortho_transform = transforms.Compose(self.ortho_transform)


    def __len__(self):
        return self.idx_map.shape[0]


    def flip_augment(self, image_arrs):
        vflip = bool(random.getrandbits(1))
        hflip = bool(random.getrandbits(1))

        return_arr = []

        for arr in image_arrs:
            if hflip:
                arr = np.flip(arr, 0)
            if vflip:
                arr = np.flip(arr, 0)
            return_arr.append(arr)

        return return_arr


    def traverse_hdf5(self, file_handle, current_name, keys):

        if type(file_handle[current_name]) == h5py.Dataset:
            keys.append(current_name)
        elif type(file_handle[current_name]) == h5py.Group:
            for next_k in file_handle[current_name].keys():
                if current_name == '/':
                    self.traverse_hdf5(file_handle, current_name + next_k, keys)
                else:
                    self.traverse_hdf5(file_handle, current_name + '/' + next_k, keys)

        return keys


    def __getitem__(self, idx):
        self.sampled_idx = np.append(self.sampled_idx, idx)
        file_handle = h5py.File(self.hdf5_path, 'r')
        ret_dict = {}
        
        for k in self.keys:
            data = file_handle[k][self.idx_map[idx]]
            if data.shape[0] == 1:
                data = np.moveaxis(data, 0, -1)
            if "DSM" in k:
                data[data > 10000] = 0

            ret_dict.update({k: data})

        data = ret_dict.values()

        if not self.inf_only:
            if self.augment:
                data = self.flip_augment(ret_dict.values())
        
        data = [self.dsm_mask_transform(x.copy()) for x in data]

        ret_dict = dict(zip(ret_dict.keys(), data))

        return ret_dict