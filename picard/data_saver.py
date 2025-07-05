import abc
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset


class Saver:
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def save(self, data: Sequence[torch.Tensor], length: int):
        pass

    def close(self):
        pass

    def __del__(self):
        self.close()


class H5Saver(Saver):
    def __init__(self, save_file_path, n_total, n_dims, labels, dtype):
        super().__init__()
        self.f = h5py.File(save_file_path, "w")
        self.save_file_path = save_file_path
        self.labels = labels
        # we don't use a compound type because they are of different shape.
        self.dataset = [self.f.create_dataset(l, shape=(n_total, d), dtype=dtype) for d, l in zip(n_dims, labels)]
        self.position = 0

    def save(self, data: Sequence[torch.Tensor], length: int):
        self.save_np([d.cpu().numpy() for d in data], length)

    def save_np(self, data: Sequence[np.ndarray], length: int):
        """
        :param data:
        :param length: the length of data, to avoid taking the length of data every time
        :return:
        """
        for i, d in enumerate(data):
            self.dataset[i][self.position : self.position + length] = d
        self.position += length

    def close(self):
        self.f.close()

    def create_torch_dataset(self, batch_size):
        if self.position < self.dataset[0].shape[0]:
            raise ValueError("Not all data are filled.")
        return H5Dataset(self.save_file_path, batch_size, self.labels)


class TensorDatasetBuiltInShuffle(IterableDataset):
    def __init__(self, *tensors, batch_size, **dataloader_kwargs):
        self.dataset = torch.utils.data.TensorDataset(*tensors)
        self.dataset = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, **dataloader_kwargs)

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)


class InMemorySaver(Saver):
    def __init__(self, n_total, n_dims):
        super().__init__()
        self.position = 0
        self.data = [torch.empty(n_total, d) for d in n_dims]

    def save(self, data: Sequence[torch.Tensor], length: int):
        for i, d in enumerate(data):
            self.data[i][self.position : self.position + length] = d
        self.position += length

    def create_torch_dataset(self, batch_size, **dataloader_kwargs):
        if self.data[0].size(0) > self.position:
            raise ValueError("Not all data are filled.")
        return TensorDatasetBuiltInShuffle(*self.data, batch_size=batch_size, **dataloader_kwargs)


class H5Dataset(IterableDataset):
    """
    To use this dataset, one must set persistent_workers=True in DataLoader.
    """

    def __init__(self, save_file_path, batch_size, labels):
        self.f = h5py.File(save_file_path, "r")
        self.batch_size = batch_size
        self.labels = labels
        self.dataset = [self.f[la] for la in labels]
        self.len = len(self.dataset[0]) // batch_size

    def __iter__(self):
        for i in range(self.len):
            yield [torch.from_numpy(d[i * self.batch_size : (i + 1) * self.batch_size]) for d in self.dataset]

    def __len__(self):
        return self.len

    def close(self):
        self.f.close()

    def __del__(self):
        self.close()
