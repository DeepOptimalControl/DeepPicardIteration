import functools
import math
import os
from datetime import timedelta
from typing import Union, Tuple, Iterable, Sequence, Optional, List

import lightning.pytorch as pl
import numpy as np
import torch
from rich.console import Console
from rich.progress import (
    ProgressColumn,
    Task,
    ProgressType,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    Progress,
)
from rich.style import Style, StyleType
from rich.text import Text
from torch import nn as nn
from yacs.config import CfgNode


def get_device_prefer_cpu(device):
    if device is None:
        return torch.device("cpu")
    elif isinstance(device, str):
        return torch.device(device)
    return device


def return_cuda_if_available():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_a_valid_device(device):
    device_types = ["cpu", "cuda"]
    if device is None:
        return True
    elif isinstance(device, str):
        return any([device.startswith(device_type) for device_type in device_types])
    elif isinstance(device, torch.device):
        return True
    else:
        return False


def count_cuda_time_wrapper(func):
    name = func.__name__
    if not os.environ.get("PROFILE_CUDA"):
        return func

    @functools.wraps(func)
    def _fn(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        print(f"{name} spent: {start.elapsed_time(end)} ms")
        return result

    return _fn


def get_device_prefer_cuda(device):
    if device is None:
        return return_cuda_if_available()
    elif isinstance(device, str):
        return torch.device(device)
    elif isinstance(device, torch.device):
        return device
    else:
        raise ValueError(f"device must be None, str or torch.device, got {device}")


class ZeroFunction(torch.autograd.Function):
    """
    This function implements the zero function:
        it accepts a 2d tensor and return a 2d tensor of zeros (with dimension 1).
    """

    @staticmethod
    def forward(ctx, *args, **kwargs):
        tx = args[0]
        return torch.zeros(tx.size(0), 1, device=tx.device)

    @staticmethod
    def backward(ctx, *grad_output):
        # Return None for the gradient with respect to the input, as it's independent of the input
        return None


class MetaRegistry(type):
    """
    Metaclass that registers subclasses in a dictionary.
    """

    def __init__(cls, name, bases, nmspc):
        super(MetaRegistry, cls).__init__(name, bases, nmspc)
        if not hasattr(cls, "_registry"):
            cls._registry = {}
        cls._registry[name] = cls

    def get_class(cls, name):
        """
        Method to get a class from the registry by its name.
        """
        if name not in cls._registry:
            raise ValueError(f"Unknown class name {name}")
        return cls._registry[name]


def compute_at_t(
    u: nn.Module,
    t: torch.Tensor,
    equation,
    maximal_batch_size: int = None,
    return_x: bool = True,
    eval_gradient: bool = False,
    eval_hessian: bool = False,
) -> Union[
    Tuple[float, float],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    :param u: the Picard solution
    :param equation: the equation
    :param maximal_batch_size: maximal batch size to use when estimating the solution
    :return: (u of network, u ground truth, [optional] x)
    """
    grad_option = torch.enable_grad() if eval_gradient else torch.no_grad()
    with grad_option:
        x_all = []
        n_batch = t.size(0)
        if maximal_batch_size is not None and n_batch > maximal_batch_size:
            u_value_batches = []
            u_exact_batches = []
            u_x_value_batches = []
            u_x_exact_batches = []
            u_xx_value_batches = []
            u_xx_exact_batches = []
            for i in range(0, n_batch, maximal_batch_size):
                t_this_batch = t[i : i + maximal_batch_size]
                x_this_batch = equation.sample_x(t_this_batch)
                if return_x:
                    x_all.append(x_this_batch)
                if eval_gradient:
                    x_this_batch.requires_grad_()
                tx_this_batch = torch.cat([t_this_batch, x_this_batch], dim=-1)
                u_value = u(tx_this_batch)
                u_value_batches.append(u_value)
                u_exact_batches.append(equation.exact_solution(t_this_batch, x_this_batch))
                if eval_gradient:
                    u_x_value = torch.autograd.grad(
                        outputs=u_value.sum(),
                        inputs=x_this_batch,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    u_x_exact = equation.u_x(t_this_batch, x_this_batch)
                    u_x_value_batches.append(u_x_value)
                    u_x_exact_batches.append(u_x_exact)
                    if eval_hessian:
                        u_xx_value = get_hessian(x_this_batch, u_value, u_x_value)
                        u_xx_exact = equation.u_hessian(t_this_batch, x_this_batch)
                        u_xx_value_batches.append(u_xx_value)
                        u_xx_exact_batches.append(u_xx_exact)
            u_value = torch.cat(u_value_batches, dim=0)
            u_exact = torch.cat(u_exact_batches, dim=0)
            if eval_gradient:
                u_x_value = torch.cat(u_x_value_batches, dim=0)
                u_x_exact = torch.cat(u_x_exact_batches, dim=0)
                if eval_hessian:
                    u_xx_value = torch.cat(u_xx_value_batches, dim=0)
                    u_xx_exact = torch.cat(u_xx_exact_batches, dim=0)
            if return_x:
                x = torch.cat(x_all, dim=0)
        else:
            x = equation.sample_x(t)
            x.requires_grad_()
            tx = torch.cat([t, x], dim=-1)
            u_value = u(tx)
            u_exact = equation.exact_solution(t, x)
            if eval_gradient:
                u_x_value = torch.autograd.grad(
                    outputs=u_value.sum(),
                    inputs=x,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                u_x_exact = equation.u_x(t, x)
                if eval_hessian:
                    u_xx_value = get_hessian(x, u_value, u_x_value)
                    u_xx_exact = equation.u_hessian(t, x)
    u_value = u_value.detach().cpu().numpy()
    u_exact = u_exact.detach().cpu().numpy()
    if eval_gradient:
        if return_x:
            return (
                u_value,
                u_exact,
                u_x_value.detach().cpu().numpy(),
                u_x_exact.detach().cpu().numpy(),
                x.detach().cpu().numpy(),
            )
        else:
            if eval_hessian:
                return (
                    u_value,
                    u_exact,
                    u_x_value.detach().cpu().numpy(),
                    u_x_exact.detach().cpu().numpy(),
                    u_xx_value.detach().cpu().numpy(),
                    u_xx_exact.detach().cpu().numpy(),
                )
            else:
                return u_value, u_exact, u_x_value.detach().cpu().numpy(), u_x_exact.detach().cpu().numpy(), None, None
    else:
        if return_x:
            return u_value, u_exact, x.detach().cpu().numpy()
        else:
            return u_value, u_exact


def compute_at_t_valuegrad(
    solution: nn.Module,
    t: torch.Tensor,
    equation,
    maximal_batch_size: int = None,
) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    :param u: the Picard solution
    :param equation: the equation
    :param maximal_batch_size: maximal batch size to use when estimating the solution
    :return: (u of network, u ground truth, [optional] x)
    """
    n_batch = t.size(0)
    if maximal_batch_size is not None and n_batch > maximal_batch_size:
        u_value_batches = []
        u_exact_batches = []
        u_x_value_batches = []
        u_x_exact_batches = []
        for i in range(0, n_batch, maximal_batch_size):
            t_this_batch = t[i : i + maximal_batch_size]
            x_this_batch = equation.sample_x(t_this_batch)
            tx_this_batch = torch.cat([t_this_batch, x_this_batch], dim=-1)
            u_u_x = solution(tx_this_batch)
            u_value = u_u_x[:, 0:1]
            u_value_batches.append(u_value)
            u_value_exact = equation.exact_solution(t_this_batch, x_this_batch)
            u_exact_batches.append(u_value_exact)
            u_x_value = u_u_x[:, 1:]
            u_x_exact = equation.u_x(t_this_batch, x_this_batch)
            u_x_value_batches.append(u_x_value)
            u_x_exact_batches.append(u_x_exact)
        u_value = torch.cat(u_value_batches, dim=0)
        u_exact = torch.cat(u_exact_batches, dim=0)
        u_x_value = torch.cat(u_x_value_batches, dim=0)
        u_x_exact = torch.cat(u_x_exact_batches, dim=0)
    else:
        x = equation.sample_x(t)
        x.requires_grad_()
        tx = torch.cat([t, x], dim=-1)
        u_u_x = solution(tx)
        u_value = u_u_x[:, 0:1]
        u_exact = equation.exact_solution(t, x)
        u_x_value = u_u_x[:, 1:]
        u_x_exact = equation.u_x(t, x)
    u_value = u_value.detach().cpu().numpy()
    u_exact = u_exact.detach().cpu().numpy()
    return u_value, u_exact, u_x_value.detach().cpu().numpy(), u_x_exact.detach().cpu().numpy()


def compute_at_t_onlygrad(
    solution: nn.Module,
    t: torch.Tensor,
    equation,
    maximal_batch_size: int = None,
) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    :param u: the Picard solution
    :param equation: the equation
    :param maximal_batch_size: maximal batch size to use when estimating the solution
    :return: (u of network, u ground truth, [optional] x)
    """
    n_batch = t.size(0)
    if maximal_batch_size is not None and n_batch > maximal_batch_size:
        u_value_batches = []
        u_exact_batches = []
        u_x_value_batches = []
        u_x_exact_batches = []
        for i in range(0, n_batch, maximal_batch_size):
            t_this_batch = t[i : i + maximal_batch_size]
            x_this_batch = equation.sample_x(t_this_batch)
            tx_this_batch = torch.cat([t_this_batch, x_this_batch], dim=-1)
            u_x = solution(tx_this_batch)
            u_value = u_x[:, 0:1] * 0
            u_value_batches.append(u_value)
            u_value_exact = equation.exact_solution(t_this_batch, x_this_batch)
            u_exact_batches.append(u_value_exact)
            u_x_value = u_x
            u_x_exact = equation.u_x(t_this_batch, x_this_batch)
            u_x_value_batches.append(u_x_value)
            u_x_exact_batches.append(u_x_exact)
        u_value = torch.cat(u_value_batches, dim=0)
        u_exact = torch.cat(u_exact_batches, dim=0)
        u_x_value = torch.cat(u_x_value_batches, dim=0)
        u_x_exact = torch.cat(u_x_exact_batches, dim=0)
    else:
        x = equation.sample_x(t)
        x.requires_grad_()
        tx = torch.cat([t, x], dim=-1)
        u_x = solution(tx)
        u_value = u_x[:, 0:1] * 0
        u_exact = equation.exact_solution(t, x)
        u_x_value = u_x
        u_x_exact = equation.u_x(t, x)
    u_value = u_value.detach().cpu().numpy()
    u_exact = u_exact.detach().cpu().numpy()
    return u_value, u_exact, u_x_value.detach().cpu().numpy(), u_x_exact.detach().cpu().numpy()


class EvalCallback(pl.Callback):
    def __init__(self, eval_cfg: CfgNode, equation, writer, iter=0, output_dim=1):
        self.n_points = eval_cfg.L2_N_POINTS
        self.batch_size = eval_cfg.BATCH_SIZE
        self.equation = equation
        self.eval_gradient = eval_cfg.TEST_GRAD
        self.eval_hessian = eval_cfg.TEST_HESSIAN
        # load fixed dataset for eval if available
        try:
            t_test = torch.zeros(1, 1).to(self.equation.device)
            x_test = torch.zeros(1, self.equation.nx).to(self.equation.device)
            self.equation.exact_solution(t_test, x_test)
            print("Exact solution implemented, using exact solution for evaluation")
            self.eval_exact = True
        except NotImplementedError:
            print("Exact solution not implemented, using data for evaluation")
            load_file = (
                f"../../data/{type(self.equation).__name__}_nx={self.equation.nx}_T={self.equation.T}_N=7_100.npy"
            )
            print("Loading evaluation data from", load_file)
            data = np.load(load_file)
            self.t_eval = torch.tensor(data[:, 0:1]).to(self.equation.device)
            self.x_eval = torch.tensor(data[:, 1:-1]).to(self.equation.device)
            self.u_eval = torch.tensor(data[:, -1:]).to(self.equation.device)
            self.eval_exact = False
        if self.eval_gradient:
            assert self.eval_exact

        self.writer = writer
        self.iter = iter
        self.output_dim = output_dim

    # @rank_zero_only
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        try:
            self.equation = self.equation.to(device=pl_module.device)
            if self.eval_exact:
                t = torch.linspace(0.0, self.equation.T, self.n_points, device=pl_module.device)
                t.resize_(self.n_points, 1)
                if self.output_dim == 1:
                    if self.eval_gradient:
                        u_value, u_exact, u_x_value, u_x_exact, u_xx_value, u_xx_exact = compute_at_t(
                            pl_module,
                            t,
                            self.equation,
                            maximal_batch_size=self.batch_size,
                            return_x=False,
                            eval_gradient=self.eval_gradient,
                            eval_hessian=self.eval_hessian,
                        )
                    else:
                        u_value, u_exact = compute_at_t(
                            pl_module,
                            t,
                            self.equation,
                            maximal_batch_size=self.batch_size,
                            return_x=False,
                            eval_gradient=self.eval_gradient,
                        )
                elif self.output_dim == self.equation.nx:
                    u_value, u_exact, u_x_value, u_x_exact = compute_at_t_onlygrad(
                        pl_module,
                        t,
                        self.equation,
                        maximal_batch_size=self.batch_size,
                    )
                elif self.output_dim == 1 + self.equation.nx:
                    u_value, u_exact, u_x_value, u_x_exact = compute_at_t_valuegrad(
                        pl_module,
                        t,
                        self.equation,
                        maximal_batch_size=self.batch_size,
                    )
                else:
                    raise ValueError(f"output_dim {self.output_dim} not supported")
            else:
                self.t_eval = self.t_eval.to(pl_module.device)
                self.x_eval = self.x_eval.to(pl_module.device)
                u_value = pl_module(torch.cat([self.t_eval, self.x_eval], dim=-1)).detach().cpu().numpy()
                u_exact = self.u_eval.detach().cpu().numpy()

            error = abs(u_value - u_exact)

            # we do sum instead of mean
            u_exact_l2 = np.sqrt((u_exact**2).sum()).item()
            u_exact_l1 = np.abs(u_exact).sum().item()
            error_l2 = np.sqrt((error**2).sum()).item()
            error_l1 = np.abs(error).sum().item()
            rRMSE = error_l2 / u_exact_l2
            rMAE = error_l1 / u_exact_l1
            MSE = np.sqrt((error**2).mean()).item()
            MArE = (error / abs(u_exact)).mean().item()
            metrics = {"MSE": MSE, "rRMSE": rRMSE, "rMAE": rMAE, "MArE": MArE}
            self.writer.add_scalar("MSE", MSE, self.iter)
            self.writer.add_scalar("rRMSE", rRMSE, self.iter)
            self.writer.add_scalar("rMAE", rMAE, self.iter)
            self.writer.add_scalar("MArE", MArE, self.iter)

            if self.eval_gradient:
                error_x = abs(u_x_value - u_x_exact)
                # we do sum instead of mean
                # u_x_exact_l2 = np.sqrt((u_x_exact**2).sum()).item()
                # u_x_exact_l1 = np.abs(u_x_exact).sum().item()
                # error_x_l2 = np.sqrt((error_x**2).sum()).item()
                # error_x_l1 = np.abs(error_x).sum().item()
                # rRMSEg = error_x_l2 / u_x_exact_l2
                # rMAEg = error_x_l1 / u_x_exact_l1
                # MSEg = np.sqrt((error_x**2).mean()).item()
                # MArEg = (error_x / abs(u_x_exact)).mean().item()
                # metrics.update({"MSEg": MSEg, "rRMSEg": rRMSEg, "rMAEg": rMAEg, "MArEg": MArEg})
                # self.writer.add_scalar("MSEg", MSEg, self.iter)
                # self.writer.add_scalar("rRMSEg", rRMSEg, self.iter)
                # self.writer.add_scalar("rMAEg", rMAEg, self.iter)
                # self.writer.add_scalar("MArEg", MArEg, self.iter)

                # we do sum instead of mean
                u_x_exact_l2 = np.sqrt((u_x_exact**2).sum(0))
                u_x_exact_l1 = np.abs(u_x_exact).sum(0)
                error_x_l2 = np.sqrt((error_x**2).sum(0))
                error_x_l1 = np.abs(error_x).sum(0)
                rRMSEg = (error_x_l2 / u_x_exact_l2).mean().item()
                rMAEg = (error_x_l1 / u_x_exact_l1).mean().item()
                MSEg = np.sqrt((error_x**2).mean(0)).mean().item()
                MArEg = (error_x / abs(u_x_exact)).mean().item()

                metrics.update({"MSEg": MSEg, "rRMSEg": rRMSEg, "rMAEg": rMAEg, "MArEg": MArEg})
                self.writer.add_scalar("MSEg", MSEg, self.iter)
                self.writer.add_scalar("rRMSEg", rRMSEg, self.iter)
                self.writer.add_scalar("rMAEg", rMAEg, self.iter)
                self.writer.add_scalar("MArEg", MArEg, self.iter)

                if self.eval_hessian:
                    error_xx = abs(u_xx_value - u_xx_exact)
                    u_xx_exact_l2 = np.sqrt((u_xx_exact**2).sum(0))
                    u_xx_exact_l1 = np.abs(u_xx_exact).sum(0)
                    error_xx_l2 = np.sqrt((error_xx**2).sum(0))
                    error_xx_l1 = np.abs(error_xx).sum(0)
                    rRMSEh = (error_xx_l2 / u_xx_exact_l2).mean().item()
                    rMAEh = (error_xx_l1 / u_xx_exact_l1).mean().item()
                    MSEh = np.sqrt((error_xx**2).mean(0)).mean().item()
                    MArEh = (error_xx / abs(u_xx_exact)).mean().item()
                    metrics.update({"MSEh": MSEh, "rRMSEh": rRMSEh, "rMAEh": rMAEh, "MArEh": MArEh})
                    self.writer.add_scalar("MSEh", MSEh, self.iter)
                    self.writer.add_scalar("rRMSEh", rRMSEh, self.iter)
                    self.writer.add_scalar("rMAEh", rMAEh, self.iter)
                    self.writer.add_scalar("MArEh", MArEh, self.iter)

            pl_module.log_dict(metrics)
        except NotImplementedError:
            pass


def compute_metrics(u_pred, u_exact):
    error = ((u_pred - u_exact) ** 2).mean()
    MArE = torch.abs(((u_pred - u_exact) / u_exact)).mean()
    rRMSE = torch.sqrt(((u_pred - u_exact) ** 2).sum()) / torch.sqrt((u_exact**2).sum())
    rMAE = torch.abs((u_pred - u_exact)).sum() / torch.abs(u_exact).sum()
    print(f"Error: {error}, Relative Error: {MArE}, {rRMSE}, {rMAE}")
    return error, MArE, rRMSE, rMAE


def compute_grad_metrics(u_pred, u_exact):
    error = ((u_pred - u_exact) ** 2).mean()
    MArE = torch.abs(((u_pred - u_exact) / u_exact)).mean()
    rRMSE = (torch.sqrt(((u_pred - u_exact) ** 2).sum(0)) / torch.sqrt((u_exact**2).sum(0))).mean()
    rMAE = (torch.abs((u_pred - u_exact)).sum(0) / torch.abs(u_exact).sum(0)).mean()
    print(f"Error: {error}, Relative Error: {MArE}, {rRMSE}, {rMAE}")
    return error, MArE, rRMSE, rMAE


def hutchinson_trace_estimation_batch(u, t, x, num_samples=16):
    """
    Estimates the trace of the Hessian of `function` with respect to `inputs`.

    Args:
    - function: A callable that takes `inputs` and returns a scalar output.
    - inputs: A PyTorch tensor with respect to which we want to compute the trace of the Hessian.
    - num_samples: The number of random vectors to use in the estimation.

    Returns:
    - An estimate of the trace of the Hessian.
    """
    # Ensure inputs have gradient
    # (dim_batch, dim_d)
    inputs = x.clone().detach().requires_grad_(True)

    # Generate a batch of random Rademacher vectors (+1 or -1)
    # (dim_v, dim_batch, dim_d)
    rademacher_vectors = (
        torch.randint(
            0,
            2,
            (num_samples,) + inputs.shape,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        * 2
        - 1
    )

    # Expand inputs to match the number of samples
    # (dim_v, dim_batch, dim_d)
    expanded_inputs = inputs.unsqueeze(0).expand(num_samples, *inputs.shape)
    expanded_t = t.unsqueeze(0).expand(num_samples, *t.shape)

    # Compute the function outputs for all samples
    # (dim_v, dim_batch, dim_scalar)
    outputs = u(torch.cat([expanded_t, expanded_inputs], dim=-1))

    # Compute gradients for all samples
    # (dim_v, dim_batch, dim_scalar)
    grad_outputs = torch.autograd.grad(
        outputs=outputs.sum(),
        inputs=expanded_inputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Compute the vector-Hessian products (VHPs) for all samples
    # (dim_v, dim_batch, dim_scalar)
    vhps = torch.autograd.grad(
        outputs=grad_outputs,
        inputs=expanded_inputs,
        grad_outputs=rademacher_vectors,
        only_inputs=True,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Compute the trace estimate as the mean of the dot products of VHPs with the Rademacher vectors
    trace_estimate = (vhps * rademacher_vectors).sum(dim=-1).mean(0)

    return trace_estimate.reshape(*inputs.shape[:-1], 1)


def get_laplacian(x, u, u_x=None):
    if u_x is None:
        u_x = torch.autograd.grad(outputs=u.sum(), inputs=x, create_graph=True, retain_graph=True)[0]
    u_xx = torch.zeros_like(x[:, 0:1])
    for i in range(x.shape[1]):
        u_xx += torch.autograd.grad(
            outputs=u_x[:, i].sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0][:, i : i + 1]
    return u_xx


def get_hessian(x, u, u_x=None):
    if u_x is None:
        u_x = torch.autograd.grad(outputs=u.sum(), inputs=x, create_graph=True, retain_graph=True)[0]
    u_xx = torch.zeros(x.shape[0], x.shape[1], x.shape[1])
    for i in range(x.shape[1]):
        u_xx[:, i] = torch.autograd.grad(
            outputs=u_x[:, i].sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
    return u_xx


class RichTimeColumn(ProgressColumn):
    # Only refresh twice a second to prevent jitter
    max_refresh = 0.5

    def __init__(self, style: Union[str, Style] = "grey54") -> None:
        self.style = style
        super().__init__()

    def render(self, task: "Task") -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        remaining = task.time_remaining
        elapsed_delta = "-:--:--.------" if elapsed is None else str(timedelta(milliseconds=round(1000 * elapsed)))
        # Due to the way `task.time_remaining` is calculated, it is always an integer. Thus show only seconds.
        remaining_delta = "-:--:--" if remaining is None else str(timedelta(seconds=int(remaining)))
        return Text(f"{elapsed_delta} • {remaining_delta}", style=self.style)


def rich_track(
    sequence: Union[Sequence[ProgressType], Iterable[ProgressType]],
    description: str = "Working...",
    total: Optional[float] = None,
    auto_refresh: bool = True,
    console: Optional[Console] = None,
    transient: bool = False,
    refresh_per_second: float = 10,
    style: StyleType = "bar.back",
    complete_style: StyleType = "bar.complete",
    finished_style: StyleType = "bar.finished",
    pulse_style: StyleType = "bar.pulse",
    update_period: float = 0.1,
    disable: bool = False,
    show_speed: bool = True,
) -> Iterable[ProgressType]:
    """
    Copied from rich.progress.track with modifications:
    - replace TimeRemainingColumn with RichTimeColumn
    """

    columns: List["ProgressColumn"] = [TextColumn("[progress.description]{task.description}")] if description else []
    columns.extend(
        (
            BarColumn(
                style=style,
                complete_style=complete_style,
                finished_style=finished_style,
                pulse_style=pulse_style,
            ),
            TaskProgressColumn(show_speed=show_speed),
            RichTimeColumn(),
        )
    )
    progress = Progress(
        *columns,
        auto_refresh=auto_refresh,
        console=console,
        transient=transient,
        refresh_per_second=refresh_per_second or 10,
        disable=disable,
    )

    with progress:
        yield from progress.track(sequence, total=total, description=description, update_period=update_period)


class GaussianMixture:
    def __init__(self, means, covariances, weights):
        """
        means: torch tensor of shape (K, n) - 每个高斯成分的均值 (K 个成分, 每个是 n 维)
        covariances: torch tensor of shape (K, n, n) - 每个高斯成分的协方差矩阵 (K 个 n x n 矩阵)
        weights: torch tensor of shape (K,) - 每个高斯成分的权重 (K 个标量)
        """
        self.means = means
        self.covariances = covariances
        self.weights = weights

        self.log_2pi = math.log(2 * math.pi)
        self.log_weights = torch.log(self.weights)

        self.cov_dets = []
        self.cov_invs = []
        for k in range(self.covariances.shape[0]):
            cov_k = self.covariances[k]
            cov_det = torch.det(cov_k)  # 计算行列式
            cov_inv = torch.inverse(cov_k)  # 计算协方差矩阵的逆
            self.cov_dets.append(cov_det)
            self.cov_invs.append(cov_inv)
        self.cov_dets = torch.stack(self.cov_dets)
        self.cov_invs = torch.stack(self.cov_invs)

    def log_prob(self, x):
        """
        x: torch tensor of shape (batch, n) - 输入样本 (batch 个 n 维向量)
        """
        batch, n = x.shape
        K = self.means.shape[0]

        log_probs = []

        for k in range(K):
            mean_k = self.means[k]
            cov_inv_k = self.cov_invs[k]
            log_weight_k = self.log_weights[k]
            cov_det_k = self.cov_dets[k]

            diff = x - mean_k

            exp_term = -0.5 * torch.einsum("bi,ij,bj->b", diff, cov_inv_k, diff)
            # # 使用矩阵乘法计算 (x - mean_k) @ Σ_k^{-1}
            # term1 = torch.matmul(diff, cov_inv_k)  # 形状 (batch, n)
            #
            # # 逐元素点乘 (term1 和 diff)，并对最后一个维度求和
            # exp_term = -0.5 * torch.sum(term1 * diff, dim=-1)  # 形状 (batch,)

            # 计算高斯分布的归一化系数，log(1/sqrt((2*pi)^n * |Σ_k|))
            normalization_log = -0.5 * (n * self.log_2pi + torch.log(cov_det_k))

            # 计算 log p_k
            log_prob_k = log_weight_k + normalization_log + exp_term

            log_probs.append(log_prob_k)

        # 使用 log-sum-exp 技巧，避免数值不稳定
        log_probs = torch.stack(log_probs, dim=0)  # 形状: (K, batch)
        log_probs = torch.logsumexp(log_probs, dim=0)  # 对 K 个成分进行 log-sum-exp

        return log_probs

    def to(self, device):
        """
        将所有的张量移动到指定的设备。
        device: 要移动到的设备 (如 'cuda' 或 'cpu')。
        """
        self.means = self.means.to(device)
        self.covariances = self.covariances.to(device)
        self.weights = self.weights.to(device)
        self.log_weights = self.log_weights.to(device)
        self.cov_dets = self.cov_dets.to(device)
        self.cov_invs = self.cov_invs.to(device)
        return self


class GaussianDiagonalCovariance:
    def __init__(self, mean, covariance):
        """
        mean: torch tensor of shape (n,) - The mean vector of the Gaussian (n-dimensional)
        covariance: torch tensor of shape (n, n) - The covariance matrix of the Gaussian (n x n)
        """
        assert mean.dim() == 1, "Mean should be a 1D tensor"
        assert covariance.dim() == 2, "Covariance should be a 2D tensor"
        assert mean.shape[0] == covariance.shape[0] == covariance.shape[1], (
            "Mean and covariance should have the same dimensionality. "
            f"Got mean.shape={mean.shape}, covariance.shape={covariance.shape}"
        )
        self.mean = mean
        self.covariance = covariance
        self.diag_covariance = torch.diagonal(self.covariance, dim1=0, dim2=1)  # 只保留对角线元素
        self.cov_inv = 1.0 / self.diag_covariance  # 对角协方差的逆矩阵就是对角元素的倒数
        self.cov_det = torch.prod(self.diag_covariance)  # 行列式就是对角线元素的乘积
        self.dim = mean.shape[0]
        self.log_2pi = torch.log(torch.tensor(2.0 * torch.pi))

    def log_prob(self, x):
        """
        x: torch tensor of shape (batch, n) - Input samples (batch size, n-dimensional vectors)
        """
        x = x.reshape(-1, self.dim)
        diff = x - self.mean  # shape (batch, n)
        exp_term = -0.5 * torch.einsum("bi, i -> b", diff**2, self.cov_inv)
        normalization_log = -0.5 * (self.dim * self.log_2pi + torch.log(self.cov_det))
        log_prob = normalization_log + exp_term
        return log_prob.reshape(-1, 1)

    def grad_log_prob(self, x):
        """
        Computes the gradient of log probability with respect to x.
        x: torch tensor of shape (batch, n) - Input samples (batch size, n-dimensional vectors)
        """
        x = x.reshape(-1, self.dim)
        diff = x - self.mean  # shape (batch, n)
        grad_log_p = -diff * self.cov_inv
        return grad_log_p

    def to(self, device):
        self.device = device
        self.mean = self.mean.to(device)
        self.covariance = self.covariance.to(device)
        self.diag_covariance = self.diag_covariance.to(device)
        self.cov_det = self.cov_det.to(device)
        self.cov_inv = self.cov_inv.to(device)
        return self

    def sample(self, num_samples_shape):
        num_samples = num_samples_shape[0]
        z = torch.randn(num_samples, self.dim, device=self.mean.device)
        x = self.mean + torch.sqrt(self.diag_covariance) * z
        return x


class GaussianMixtureDiagonalCovariance:
    def __init__(self, means, covariances, weights):
        """
        means: torch tensor of shape (K, n) - 每个高斯成分的均值 (K 个成分, 每个是 n 维)
        covariances: torch tensor of shape (K, n, n) - 每个高斯成分的协方差矩阵 (K 个 n x n 矩阵)
        weights: torch tensor of shape (K,) - 每个高斯成分的权重 (K 个标量)
        """
        self.device = means.device
        self.dim = means.shape[-1]
        self.K = means.shape[-2]
        self.means = means
        self.covariances = covariances  # 这里 covariances 是全矩阵，形状为 (K, n, n)
        self.weights = weights

        # 预计算 log(2 * pi)
        self.log_2pi = torch.log(torch.tensor(2.0 * torch.pi))

        # 提取协方差矩阵的对角线元素，形状为 (K, n)
        self.diag_covariances = torch.diagonal(self.covariances, dim1=1, dim2=2)  # 只保留对角线元素

        # 预计算权重的对数
        self.log_weights = torch.log(self.weights)

        # 预计算协方差矩阵的行列式和协方差矩阵的逆（对于对角矩阵非常简单）
        self.cov_invs = 1.0 / self.diag_covariances  # 对角协方差的逆矩阵就是对角元素的倒数
        self.cov_dets = torch.prod(self.diag_covariances, dim=1)  # 行列式就是对角线元素的乘积

    def log_prob_for(self, x):
        """
        x: torch tensor of shape (batch, n) - 输入样本 (batch 个 n 维向量)
        """
        log_probs = []

        for k in range(self.K):
            mean_k = self.means[k]
            cov_inv_k = self.cov_invs[k]  # 对角元素的逆，形状应该是 (n,)
            log_weight_k = self.log_weights[k]
            cov_det_k = self.cov_dets[k]

            # 计算 (x - mean_k)
            diff = x - mean_k  # 形状 (batch, n)

            # 因为协方差是对角的，直接逐元素计算 diff^2 / sigma^2
            # exp_term = -0.5 * torch.sum(diff**2 * cov_inv_k, dim=-1)  # slower than einsum (3.45s vs 2.93s in total)
            exp_term = -0.5 * torch.einsum("bi, i -> b", diff**2, cov_inv_k)

            # 计算高斯分布的归一化系数，log(1/sqrt((2*pi)^n * |Σ_k|))
            normalization_log = -0.5 * (self.dim * self.log_2pi + torch.log(cov_det_k))

            # 计算 log 概率值
            log_prob_k = log_weight_k + normalization_log + exp_term

            log_probs.append(log_prob_k)

        # 使用 log-sum-exp 技巧，避免数值不稳定
        log_probs = torch.stack(log_probs, dim=0)  # 形状: (K, batch)
        log_probs = torch.logsumexp(log_probs, dim=0)  # 对 K 个成分进行 log-sum-exp

        return log_probs

    def log_prob(self, x):
        """
        x: torch tensor of shape (batch, n) - 输入样本 (batch 个 n 维向量)
        """
        ori_shape = x.shape
        x = x.reshape(-1, self.dim)

        # 扩展 x 的维度，以便处理 K 个成分
        # diff 的形状为 (batch, K, n)，其中每个成分都对应一个均值
        diff = x.unsqueeze(1) - self.means.unsqueeze(0)  # (batch, K, n)

        # 逐元素平方并与协方差逆进行逐元素乘法
        # 计算 exp_term，形状为 (batch, K)
        exp_term = -0.5 * torch.einsum("bkn,kn->bk", diff**2, self.cov_invs)

        # 计算归一化对数项
        normalization_log = -0.5 * (self.dim * self.log_2pi + torch.log(self.cov_dets))  # (K,)

        # 扩展 normalization_log 的维度为 (1, K)，以便与 batch 对应
        normalization_log = normalization_log.unsqueeze(0)

        # 计算 log 概率值，形状为 (batch, K)
        log_prob = self.log_weights + normalization_log + exp_term

        # 对 K 个分量使用 log-sum-exp 技巧进行稳定的归一化处理，形状为 (batch,)
        log_probs = torch.logsumexp(log_prob, dim=1)

        # return batch*1 shape
        return log_probs.reshape(*ori_shape[:-1], 1)

    def grad_log_prob(self, x):
        """
        计算 grad log p(x)
        x: torch tensor of shape (batch, n) - 输入样本 (batch 个 n 维向量)
        """
        ori_shape = x.shape
        x = x.reshape(-1, self.dim)

        # 扩展 x 的维度，以便处理 K 个成分
        diff = x.unsqueeze(1) - self.means.unsqueeze(0)  # (batch, K, n)

        # 逐元素平方并与协方差逆进行逐元素乘法
        exp_term = -0.5 * torch.einsum("bkn,kn->bk", diff**2, self.cov_invs)

        # 计算归一化对数项
        normalization_log = -0.5 * (self.dim * self.log_2pi + torch.log(self.cov_dets))  # (K,)

        # 扩展 normalization_log 的维度为 (1, K)，以便与 batch 对应
        normalization_log = normalization_log.unsqueeze(0)

        # 计算 log 概率值，形状为 (batch, K)
        log_prob = self.log_weights + normalization_log + exp_term

        # 计算每个高斯分量的概率，并使用 log-sum-exp 技巧确保数值稳定
        log_probs_per_component = log_prob - torch.logsumexp(log_prob, dim=1, keepdim=True)

        # 对于每个分量的梯度：grad log p_k(x) = - Σ_k^{-1} * (x - μ_k)
        grad_log_p_per_component = -torch.einsum("bkn,kn->bkn", diff, self.cov_invs)  # (batch, K, n)

        # 对每个分量加权平均，得到混合分布的梯度
        weighted_grads = grad_log_p_per_component * log_probs_per_component.exp().unsqueeze(-1)  # (batch, K, n)
        grad_log_p = torch.sum(weighted_grads, dim=1)  # (batch, n)

        return grad_log_p.reshape(*ori_shape)

    def to(self, device):
        """
        将所有的张量移动到指定的设备。
        device: 要移动到的设备 (如 'cuda' 或 'cpu')。
        """
        self.device = device
        self.means = self.means.to(device)
        self.covariances = self.covariances.to(device)
        self.diag_covariances = self.diag_covariances.to(device)
        self.weights = self.weights.to(device)
        self.log_weights = self.log_weights.to(device)
        self.cov_dets = self.cov_dets.to(device)
        self.cov_invs = self.cov_invs.to(device)
        return self

    def sample(self, num_samples_shape):
        """
        根据混合高斯模型采样生成数据点。

        num_samples: 需要生成的样本数量
        返回：形状为 (num_samples, n) 的样本张量
        """
        # Step 1: 根据权重从 K 个成分中选择成分
        # 从离散分布中采样，选择哪个高斯成分生成每个样本
        num_samples = num_samples_shape[0]
        component_indices = torch.multinomial(self.weights, num_samples, replacement=True)  # (num_samples,)

        # Step 2: 对于每个样本，从对应成分的高斯分布中采样
        # 创建一个空张量来存储生成的样本
        samples = torch.zeros((num_samples, self.dim), device=self.means.device)

        for i in range(self.K):
            # 对于每个成分，找到从该成分生成样本的索引
            mask = component_indices == i

            # 从该成分的均值和对角协方差矩阵中采样
            num_component_samples = mask.sum().item()
            if num_component_samples > 0:
                mean = self.means[i]  # (n,)
                diag_cov = self.diag_covariances[i]  # (n,)

                # 对角协方差矩阵的采样非常简单：均值 + 标准差 * 标准正态分布采样
                stddev = torch.sqrt(diag_cov)  # (n,)
                component_samples = mean + stddev * torch.randn(
                    num_component_samples, self.dim, device=self.means.device
                )

                # 将生成的样本放到对应的位置
                samples[mask] = component_samples

        return samples
