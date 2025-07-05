import os
from typing import Union, Tuple, Optional, Sequence

import torch
from rich.console import Console
from rich import print as rprint
from rich.panel import Panel
from torch.utils.data import IterableDataset

from picard.data_saver import (
    Saver,
    H5Saver,
    H5Dataset,
    InMemorySaver,
    TensorDatasetBuiltInShuffle,
)
from picard.utils import count_cuda_time_wrapper, rich_track


class IterableDatasetWithInternalBatch(IterableDataset):
    """
    This class implements an iterable dataset. It internally generates data in batch to accelerate the data generation.

    To use this with DataLoader, set batch_size=None.
    """

    def __init__(self, n: int, n_batch_buffer: Union[int, float], batch_size: int, batch_data_generator: callable):
        """
        :param n: total number of data to generate
        :param n_batch_buffer: number of batches to generate in each call to batch_data_generator.
            A large number will consume more memory and introduce more latency.
            Can be a float in (0, 1) in case the batch size is large and the memory is limited.
            If it is a float, it is required that 1/n_batch_buffer is an integer for simplicity.
        :param batch_size: batch size
        :param batch_data_generator: the function to generate data in batch, it should take an integer as input,
            and output two batches of data (x: [n_batch, shape_x], y: [n_batch, shape_y]).

        Note: n must be a multiple of batch_size*n_batch_buffer.
         This is to avoid the need of determining the last buffer.
        """
        super().__init__()
        (
            self.n_batch_buffer,
            self.n_samples_each_buffer,
            self._n_calls_to_generator_each_buffer,
            self._n_samples_each_call,
        ) = self._get_sampling_number_info(n_batch_buffer, batch_size)
        self.batch_size = batch_size
        self.n_buffer_refresh = self._get_n_refresh(n, "Init")
        self.batch_data_generator = batch_data_generator

        self.saver = None

    @staticmethod
    def _get_sampling_number_info(n_batch_buffer: Union[float, int], batch_size: int) -> Tuple[int, int, int, int]:
        """
        :param n_batch_buffer:
        :param batch_size:
        :return:
            the number of batches to generate in each buffer,
            the number of samples in each buffer, = n_batch_buffer * batch_size
            the number of calls to generator in each buffer,
            the number of samples in each call to generator.
        """
        if abs(round(n_batch_buffer) - n_batch_buffer) < 1e-5:
            n_batch_buffer = round(n_batch_buffer)
        if isinstance(n_batch_buffer, int):
            n_samples = n_batch_buffer * batch_size
            return n_batch_buffer, n_samples, 1, n_samples
        assert isinstance(n_batch_buffer, float) and 0 < n_batch_buffer < 1
        n_calls_to_generator_each_buffer = round(1 / n_batch_buffer)
        n_samples_each_call = batch_size // n_calls_to_generator_each_buffer
        assert n_samples_each_call * n_calls_to_generator_each_buffer == batch_size, (
            f"batch_size {batch_size} must be a multiple of "
            f"n_calls_to_generator_each_buffer {n_calls_to_generator_each_buffer}"
        )
        return 1, batch_size, n_calls_to_generator_each_buffer, n_samples_each_call

    def _get_n_refresh(self, n: int, stage="Change Size"):
        # each buffer generates n_samples_each_buffer data, so we need n/n_samples_each_buffer buffers
        n_buffer_refresh = n // self.n_samples_each_buffer
        assert n_buffer_refresh * self.n_samples_each_buffer == n, (
            f"Total number of data {n} must be a multiple of "
            f"batch_size({self.batch_size})*n_batch_buffer({self.n_batch_buffer})={self.n_samples_each_buffer}"
        )
        rprint(
            Panel(
                (
                    f"total number of data: {n}\n"
                    f"number of buffers: {n_buffer_refresh}\n"
                    f"each buffer: {self.n_samples_each_buffer}\n"
                    f"{self._n_calls_to_generator_each_buffer} calls to generator "
                    f"(each generate {self._n_samples_each_call})\n"
                ),
                title=f"{os.getpid()}-{self.__class__.__qualname__} {stage}",
            )
        )
        return n_buffer_refresh

    def set_size(self, n):
        self.n_buffer_refresh = self._get_n_refresh(n)

    def attach_saver(self, saver: Saver):
        assert self.saver is None, "Saver can only be attached once"
        self.saver = saver

    @count_cuda_time_wrapper
    def refresh_buffer(self):
        buffers_x = []
        buffers_y = []
        for _ in range(self._n_calls_to_generator_each_buffer):
            buffer_in_call = self.batch_data_generator(self._n_samples_each_call)
            buffers_x.append(buffer_in_call[0])
            buffers_y.append(buffer_in_call[1])
        buffer = (torch.cat(buffers_x, dim=0), torch.cat(buffers_y, dim=0))
        return buffer

    def __iter__(self):
        for _ in range(self.n_buffer_refresh):
            buffer = self.refresh_buffer()
            if self.saver is not None:
                self.saver.save(buffer, self.n_samples_each_buffer)
            x, y = buffer
            x = x.view(self.n_batch_buffer, self.batch_size, -1)
            y = y.view(self.n_batch_buffer, self.batch_size, -1)
            for i in range(self.n_batch_buffer):
                yield x[i], y[i]
        if self.saver is not None:
            self.saver.close()

    def __len__(self):
        """
        No need to consider the number of workers, simply return the total number of data.
        Dataloader will call this method to get the length of the dataset before creating the workers.
        Note: As we do batching internally, the length of the dataset is the total number of batches.
        """
        return self.n_buffer_refresh * self.n_batch_buffer


def _preload(dataset: IterableDatasetWithInternalBatch):
    pid = os.getpid()
    console = Console()
    for _ in rich_track(
        dataset,
        console=console,
        total=len(dataset),
        description=f"[{pid}]Preloading({len(dataset)} Batches)",
    ):
        pass


class CacheToFileWrapper(IterableDataset):
    def __init__(self, dataset: IterableDatasetWithInternalBatch):
        self.dataset = dataset
        self.on_gen_stage = True
        self.saver: Optional[H5Saver] = None
        self.dataset_from_file: Optional[H5Dataset] = None

    def init(
        self,
        save_file_path,
        n_total: int,
        n_dims: Sequence[int],
        labels: Sequence[str],
        dtype,
        preload: bool = False,
    ):
        """
        Worker init function must call this method to initialize the saver.
        :param save_file_path: pathlike
        :param n_total: total number of data
        :param n_dims: list of dimensions of each data
        :param labels: list of labels of each data
        :param dtype:
        :param preload: whether to preload the data to memory on `init`
        :return:
        """
        self.dataset.set_size(n_total)
        self.saver = H5Saver(save_file_path, n_total, n_dims, labels, dtype)
        self.dataset.attach_saver(self.saver)
        if preload:
            _preload(self.dataset)
            self.dataset_from_file = self.saver.create_torch_dataset(self.dataset.batch_size)
            self.on_gen_stage = False

    def __iter__(self):
        assert self.saver is not None, "Must call init() before using the iterator"
        if self.on_gen_stage:
            self.on_gen_stage = False
            return iter(self.dataset)
        if self.dataset_from_file is None:
            self.dataset_from_file = self.saver.create_torch_dataset(self.dataset.batch_size)
            assert len(self.dataset_from_file) == len(self.dataset), (
                f"len(self.dataset_from_file) {len(self.dataset_from_file)} must be equal to "
                f"len(self.dataset) {len(self.dataset)}"
            )
        return iter(self.dataset_from_file)

    def __len__(self):
        return len(self.dataset)


class CacheToMemoryWrapper(IterableDataset):
    def __init__(
        self,
        dataset: IterableDatasetWithInternalBatch,
        batch_size: int = None,
        **dataloader_kwargs,
    ):
        self.dataset = dataset
        self.saver: Optional[InMemorySaver] = None
        self.dataset_from_memory: Optional[TensorDatasetBuiltInShuffle] = None
        self.on_gen_stage = True
        self.batch_size = batch_size or self.dataset.batch_size
        self.batch_size_modified = self.batch_size != self.dataset.batch_size
        self.dataloader_kwargs = dataloader_kwargs
        self.len = len(self.dataset)

        self.h5_saver_args = None

    def init(self, n_total: int, n_dims: Sequence[int], preload: bool = False):
        self.dataset.set_size(n_total)
        self.saver = InMemorySaver(n_total, n_dims)
        self.dataset.attach_saver(self.saver)
        do_preload = preload
        if self.batch_size_modified:
            print("The batch size is modified; to support this, we need to preload the data first.")
            do_preload = True
        if self.h5_saver_args is not None:
            print("The data will be saved to file, so we need to preload the data first.")
            do_preload = True
        if do_preload:
            _preload(self.dataset)
            self.dataset_from_memory = self.saver.create_torch_dataset(self.batch_size, **self.dataloader_kwargs)
            self.len = len(self.dataset_from_memory)
            self.on_gen_stage = False
            if self.h5_saver_args is not None:
                h5_saver = H5Saver(*self.h5_saver_args)
                h5_saver.save(self.saver.data, self.saver.data[0].size(0))

    def enable_save_to_file(self, h5_saver_args):
        self.h5_saver_args = h5_saver_args

    def __iter__(self):
        assert self.saver is not None, "Must call init() before using the iterator"
        if self.on_gen_stage:
            self.on_gen_stage = False
            return iter(self.dataset)
        if self.dataset_from_memory is None:
            self.dataset_from_memory = self.saver.create_torch_dataset(self.batch_size, **self.dataloader_kwargs)
            self.len = len(self.dataset_from_memory)
        return iter(self.dataset_from_memory)

    def __len__(self):
        return self.len


class DummyDataset(IterableDataset):
    def __iter__(self):
        yield torch.tensor([1.0]), torch.tensor([1.0])
        return self

    def __len__(self):
        return 1
