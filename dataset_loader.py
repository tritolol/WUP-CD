import numpy as np
import h5py
from hdf5_dataset import Hdf5Dataset
import torch


class WupCdLoader:
    def __init__(self, test_split=0.2, folds=10, fold_idx=0):
        self.test_split = test_split
        self.folds = folds
        self.fold_idx = fold_idx

    def get_train_test_idx(self, input_size, idx_map, base_idx_shuffled):
        train_split_ind = int((1 - self.test_split) * base_idx_shuffled.shape[0])
        fold_ind = np.linspace(0, base_idx_shuffled.shape[0], self.folds + 1).astype('i')

        ring_inds = np.array(list(range(input_size)) + list(range(input_size)))
        train_start_ind = fold_ind[self.fold_idx]
        train_end_ind = fold_ind[self.fold_idx] + train_split_ind

        test_start_ind = fold_ind[self.fold_idx] + train_split_ind
        test_end_ind = fold_ind[self.fold_idx] + base_idx_shuffled.shape[0]

        train_inds = ring_inds[np.arange(train_start_ind, train_end_ind)]
        test_inds = ring_inds[np.arange(test_start_ind, test_end_ind)]

        shuffled_train_inds = base_idx_shuffled[train_inds]
        shuffled_test_inds = base_idx_shuffled[test_inds]

        train_idx = idx_map[shuffled_train_inds].flatten()
        test_idx = idx_map[shuffled_test_inds].flatten()

        return train_idx, test_idx

    def get_train_test_datasets(self, dataset_path, normalize=False):
        f = h5py.File(dataset_path, 'r')

        base_idx_shuffled = f["BASE_IDX_SHUFFLED"][:]
        idx_map = f["nDSM/A_IDX_MAP"][:]

        train_idx, test_idx = self.get_train_test_idx(idx_map.size, idx_map, base_idx_shuffled)
       
        chg_class_idx = f["MASKS/CONSTRUCTION_DEMOLITION_CHANGE_IDX"][:]            # idx in all samples
        train_chg_class_idx = [idx for idx in chg_class_idx if idx in train_idx]    # filter idx that are in train_idx
        len_train_no_chg_class_idx = len(train_idx) - len(train_chg_class_idx)

        train_tdomCD = Hdf5Dataset(dataset_path, train_idx, normalize=normalize, augment=True)
        test_tdomCD = Hdf5Dataset(dataset_path, test_idx, augment=False, output_mask=False)

        # weighted tile sampler
        all_sample_to_train_set_map = np.zeros((np.max(train_idx) + 1,)).astype("int")
        all_sample_to_train_set_map[train_idx] = np.arange(0, train_idx.size)

        weights = np.array(len(train_idx) * [1/ len_train_no_chg_class_idx])
        weights[all_sample_to_train_set_map[train_chg_class_idx]] = 1 / len(train_chg_class_idx)
        weighted_train_sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

        return train_tdomCD, test_tdomCD, weighted_train_sampler