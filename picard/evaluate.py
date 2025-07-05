import pathlib
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from picard.data import OnlineDataGenerator
from picard.equations import Equation
from picard.utils import compute_at_t


class Evaluator:
    """
    Evaluator for a single Picard solution.
    """

    def __init__(
        self,
        u: nn.Module,
        equation: Equation,
        N: int,
        i: int,
        n_estimate_terminal: int,
        n_estimate_integral: int,
        maximal_batch_size: int = None,
    ):
        """
        :param u: the Picard solution
        :param equation: the equation
        :param N: total number of iterations
        :param i: current iteration
        :param n_estimate_terminal: number of samples to estimate the terminal condition
        :param n_estimate_integral: number of samples to estimate the integral

        Note that the iteration number `i` and total number of iterations `N` must be provided.
        Each u is in principle associated with a unique (i, N) pair.
         Therefore, one prefers to create a new evaluator for each u instead of reusing one evaluator.
        """
        self.u = u
        self.n_terminal = n_estimate_terminal
        self.n_integral = n_estimate_integral
        self.data_generator = OnlineDataGenerator(
            equation,
            self.u,
            N,
            i,
            n_estimate_terminal=self.n_terminal,
            n_estimate_integral=self.n_integral,
        )
        self.maximal_batch_size = round(maximal_batch_size)

    def to(self, device):
        self.u.to(device=device)
        self.data_generator.to(device=device)

    def monte_carlo(
        self,
        x: torch.Tensor,
        t: torch.Tensor = None,
    ):
        """
        :param x:  (n_batch, nx) or (nx,)
        :param t: optional, if provided, must be (n_batch, 1) or (1,)
        :return: (n_batch, nu)
        """
        single = False
        with torch.no_grad():
            if x.dim() == 1:
                single = True
                x = torch.unsqueeze(x, dim=0)
            if t is None:
                t = torch.zeros(x.size(0), 1)
            elif t.dim() == 1:
                t = torch.unsqueeze(t, dim=0)
            assert x.size(0) == t.size(0)
            tx = torch.cat([t, x], dim=-1)
            u = self.data_generator.generate(tx)
        if single:
            u = u.squeeze(dim=0)
        return u

    def evaluate(self, n_batch):
        t = torch.linspace(
            0.0,
            self.data_generator.equation.T,
            n_batch,
            device=self.data_generator.device,
        )
        t.resize_(n_batch, 1)
        return self.evaluate_at_t(t)

    def evaluate_at_t(self, t: torch.Tensor):
        u_value, u_exact = compute_at_t(
            self.u,
            t,
            self.data_generator.equation,
            self.maximal_batch_size,
            return_x=False,
        )
        error = abs(u_value - u_exact)

        l2_error = np.sqrt((error**2).mean()).item()
        l1_relative_error = (error / abs(u_exact)).mean().item()
        return l2_error, l1_relative_error

    def l2_at_t(
        self, t: torch.Tensor
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        :param t: (n_batch, 1)
        :return: (n_batch, nu)
        """
        u_value, u_exact, x = compute_at_t(
            self.u, t, self.data_generator.equation, self.maximal_batch_size
        )
        error = abs(u_value - u_exact)
        return (
            np.sqrt((error**2).mean()).item(),
            error,
            x,
            u_value,
        )

    def l2_at_given_solution(
        self, tx: np.ndarray, u_exact: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        with torch.no_grad():
            tx = torch.from_numpy(tx)
            u_value = self.u(tx).detach().cpu().numpy()
            error = abs(u_value - u_exact)
        return (
            np.sqrt((error**2).mean()).item(),
            error,
            u_value,
        )

    def l2_at_t_grids(
        self, n_batch
    ) -> Tuple[np.ndarray, Tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
        t = torch.linspace(
            0.0,
            self.data_generator.equation.T,
            n_batch,
            device=self.data_generator.device,
        )
        t.resize_(n_batch, 1)
        l2_info = self.l2_at_t(t)
        return t.detach().cpu().numpy(), l2_info

    def l2_at_t0(
        self, n_batch: int
    ) -> Tuple[np.ndarray, Tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
        t = torch.zeros(n_batch, 1, device=self.data_generator.device)
        l2_info = self.l2_at_t(t)
        return t.detach().cpu().numpy(), l2_info

    def l2_at_td(
        self, n_batch: int
    ) -> Tuple[np.ndarray, Tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Calculate the L2 error at t = T[1-U[0,1]**(N-i+1)], i.e., according to the Picard iteration distribution.
        :param n_batch:
        :return:
        """
        t = self.data_generator.sample_t(n_batch)
        l2_info = self.l2_at_t(t)
        return t.detach().cpu().numpy(), l2_info

    @classmethod
    def plot_error_fig(
        cls,
        t: np.ndarray,
        x: np.ndarray,
        error: np.ndarray,
    ):
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        # The first subplot for t vs error
        axs[0].scatter(t, error)
        axs[0].set_title("t vs $|u(t,x) - u_{exact}(t,x)|$")
        axs[0].set_xlabel("t")
        axs[0].set_ylabel("error")

        # The second subplot for x vs error
        # takes magnitude of x if x is not 1D
        x_text = "x"
        if x.ndim == 2 and x.shape[1] > 1:
            x = np.linalg.norm(x, axis=-1)
            x_text = "|x|"
        axs[1].scatter(x, error)
        axs[1].set_title(r"%s vs $|u(t,x) - u_{exact}(t,x)|$" % x_text)
        axs[1].set_xlabel(f"{x_text}")
        axs[1].set_ylabel("error")

        plt.tight_layout()
        return fig

    @classmethod
    def plot_error(
        cls,
        exp_dir: pathlib.Path,
        t: np.ndarray,
        x: np.ndarray,
        error: np.ndarray,
        name: str,
    ) -> pathlib.Path:
        cls.plot_error_fig(t, x, error)
        fig_path = exp_dir / f"{name}_error_distribution.png"
        plt.savefig(fig_path)
        plt.close()
        return fig_path
