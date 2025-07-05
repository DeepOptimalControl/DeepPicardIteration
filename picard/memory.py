import contextlib
import dataclasses
import functools
import gc

import torch
from torch.cuda import reset_peak_memory_stats, max_memory_allocated


def skip_if_not_enabled(func):
    def wrapper(self, *args, **kwargs):
        if self.enabled:
            return func(self, *args, **kwargs)
        return

    return wrapper


@dataclasses.dataclass
class GPUMemoryReport:
    peak_allocated: int = 0  # in bytes
    enabled: bool = torch.cuda.is_available()

    @skip_if_not_enabled
    def start(self):
        reset_peak_memory_stats()
        self.peak_allocated = max_memory_allocated()

    @skip_if_not_enabled
    def end(self) -> float:
        """
        Returns peak allocated memory in MB
        """
        return (max_memory_allocated() - self.peak_allocated) / 1024 / 1024


def do_memory_clean_on_exit_wrapper(fn):
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        results = fn(*args, **kwargs)
        gc.collect()
        torch.cuda.empty_cache()
        return results

    return _fn


class GPUMemoryTracker:
    """
    https://pytorch.org/docs/stable/notes/cuda.html#memory-management
    PyTorch uses a caching memory allocator to speed up memory allocations.
    Therefore,
     the total amount of memory available is more than the total amount of free memory reported by the CUDA API.
    Some utils to be used:
    - torch.cuda.memory_allocated(): current memory occupied by tensors
    - torch.cuda.memory_reserved(): current memory reserved by the caching allocator
        reserved = allocated + cached
    """

    _debug = True

    @classmethod
    @contextlib.contextmanager
    def enable_debug(cls):
        cls._debug = True
        yield
        cls._debug = False

    @staticmethod
    def get_memory_available():
        # mem_get_info calls cudaMemGetInfo, the returned memory is in bytes
        free_memory_in_gpu = torch.cuda.mem_get_info()[0] / 1024 / 1024
        free_memory_reserved = GPUMemoryTracker.get_memory_free_in_torch()
        return free_memory_in_gpu + free_memory_reserved

    @staticmethod
    def get_memory_allocated_by_torch():
        return torch.cuda.memory_allocated() / 1024 / 1024

    @staticmethod
    def get_memory_reserved_by_torch():
        return torch.cuda.memory_reserved() / 1024 / 1024

    @staticmethod
    def get_memory_free_in_torch():
        return GPUMemoryTracker.get_memory_reserved_by_torch() - GPUMemoryTracker.get_memory_allocated_by_torch()

    @classmethod
    def print(cls, *args, **kwargs):
        if cls._debug:
            print(*args, **kwargs)

    @staticmethod
    @do_memory_clean_on_exit_wrapper
    def try_dataset(dataset_fn, *dataset_fn_args):
        try:
            dataset = dataset_fn(*dataset_fn_args)
            for _ in dataset:
                pass
        except torch.cuda.OutOfMemoryError:
            return False
        return True

    @dataclasses.dataclass
    class EstimationResult:
        peak_memory_usage: float
        n_buffer: float
        batch_size: int
        n_data_per_sample: int

        def print(self):
            msg = f"Peak memory usage: {self.peak_memory_usage} MB (Total {self.n_data_per_sample} data per sample)"
            msg += f"with {self.n_buffer} buffer and batch size {self.batch_size}" if self.batch_size > 0 else ""
            print(msg)

    @classmethod
    def estimate_largest_data_points(cls, dataset_fn, reserved_memory=None, n_data_per_sample_init=1024):
        reporter = GPUMemoryReport()
        reserved_memory = reserved_memory or 0.0
        # 1. to get a rough estimate of memory vs number of data points
        n_data_to_try = n_data_per_sample_init
        cls.print("Initial trials with buffer size: ", n_data_to_try)
        while True:
            reporter.start()
            memory_ok = cls.try_dataset(dataset_fn, n_data_to_try * 2, 1, n_data_to_try)
            if memory_ok:
                memory_used = reporter.end()
                cls.print(f"Memory used: {memory_used} MB with {n_data_to_try} data points")
                break
            else:
                n_data_to_try = n_data_to_try // 2
                if n_data_to_try == 0:
                    raise RuntimeError("Cannot sample each a single data point!")
                cls.print(f"\tOOM happened, retrying with smaller data points: {n_data_to_try}")
        # 2. to estimate the maximal number of data points that can be sampled
        step_percent = 0.1
        n_data_to_try_ok = n_data_to_try
        memory_used_ok = memory_used
        total_available_memory = cls.get_memory_available()
        available_memory = total_available_memory - reserved_memory
        n_data_to_try = round(n_data_to_try * available_memory / memory_used)
        cls.print(f"Available memory (MB): {available_memory} = {total_available_memory} - {reserved_memory}")
        cls.print("Estimated maximal data points: ", n_data_to_try)
        failed_once = False
        cls.print("Finding maximal data points...")
        while True:
            reporter.start()
            memory_ok = cls.try_dataset(dataset_fn, n_data_to_try * 2, 1, n_data_to_try)
            memory_used = reporter.end()
            memory_ok = memory_ok and memory_used < available_memory
            if memory_ok:
                memory_used_ok = memory_used
                n_data_to_try_ok = n_data_to_try
                if failed_once:
                    cls.print(f"Was failed, now it fits: {n_data_to_try_ok} data points used {memory_used_ok} MB")
                    return cls.EstimationResult(memory_used_ok, 0, 0, n_data_per_sample=n_data_to_try_ok)
                cls.print(f"Memory fits! Used: {memory_used} MB with {n_data_to_try} data points")
                n_data_to_try = max(
                    round(n_data_to_try * available_memory / memory_used),
                    round(n_data_to_try * (1 + step_percent)),
                )
            else:
                failed_once = True
                cls.print(f"OOM or exceed reserved memory happened with {n_data_to_try} data points")
                n_data_to_try = round(n_data_to_try * (1 - step_percent))
                if n_data_to_try <= n_data_to_try_ok:
                    cls.print(
                        f"Memory can fit {n_data_to_try_ok} data points at most "
                        f"(with memory usage {memory_used_ok} MB)"
                    )
                    return cls.EstimationResult(memory_used, 0, 0, n_data_per_sample=n_data_to_try)

    # noinspection PyArgumentList
    @classmethod
    def estimate_memory_usage(
        cls, dataset_fn, batch_size: int, initial_n_buffer_to_try: float = 16.0
    ) -> EstimationResult:
        """
        Will report the peak memory usage in MB with the number of buffer.
        :param dataset_fn: a function that returns a dataset. The function has three arguments:
            n_total: int, number of total data points to be sampled
            n_batch_buffer: int or float, number of batches to be sampled in each buffer
            batch_size: int, batch size
            Therefore, n_batch_buffer * batch_size is the number of data points sampled in one call to the generator.
        :return:
            Memory usage in MB, n_buffer
        """
        reporter = GPUMemoryReport()
        assert batch_size > 0

        # when batch_size is 0, we estimate the maximal number of data points that can be sampled

        n_buffer = initial_n_buffer_to_try
        n_data_batch = round(n_buffer * 2)  # number of batch to sampled for each dataset
        while True:
            reporter.start()
            memory_ok = cls.try_dataset(dataset_fn, n_data_batch * batch_size, n_buffer, batch_size)
            if memory_ok:
                return cls.EstimationResult(
                    reporter.end(),
                    n_buffer,
                    batch_size,
                    n_data_per_sample=round(n_buffer * batch_size),
                )
            else:
                n_data_batch = max(round(n_data_batch / 2), 1)
                n_buffer = n_buffer / 2
                print("OOM happened, retrying with smaller buffer: ", n_buffer)
