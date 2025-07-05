import abc
import random
import time
from typing import Tuple, Dict, Any

import torch

from picard.solution import PicardBaseSolution
from picard.torch_func import jacrev, vmap, hessian
from picard.utils import MetaRegistry, count_cuda_time_wrapper


class LossScaler(metaclass=MetaRegistry):
    @abc.abstractmethod
    def scale(self, v_loss: torch.Tensor, g_loss_multi_dim: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        :param v_loss: (1,)
        :param g_loss_multi_dim: (nx,), it has been already averaged over the batch dimension and squared.
        :return:
            total_loss: (1,)
            info: dict
        """

    def scale_g_h(
        self,
        v_loss: torch.Tensor,
        g_loss_multi_dim: torch.Tensor,
        h_loss_multi_dim: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        :param v_loss: (1,)
        :param g_loss_multi_dim: (nx,), it has been already averaged over the batch dimension and squared.
        :param h_loss_multi_dim: (nx*nx,), it has been already averaged over the batch dimension and squared.
        :return:
            total_loss: (1,)
            info: dict
        """


class SimpleLossScaler(LossScaler):
    def scale(self, v_loss: torch.Tensor, g_loss_multi_dim: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        with torch.no_grad():
            g_loss = torch.sum(g_loss_multi_dim, keepdim=True, dim=-1)
            a = v_loss / g_loss
            a = torch.clamp(a, min=0.0, max=1e3)
        loss = v_loss + a * g_loss
        return loss, {
            "train_gradient_loss(unscaled)": g_loss,
            "train_gradient_loss_scaling_factor": a,
        }


class DimensionLossScaler(LossScaler):
    def scale(self, v_loss: torch.Tensor, g_loss_multi_dim: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        r"""
        scale the gradient loss to be of the same order as the value loss
        """
        with torch.no_grad():
            # scale both losses to be of the same order
            # loss = v_loss + a * gd_loss
            a = v_loss / g_loss_multi_dim
            a = torch.clamp(a, min=0.0, max=1e3)
            mean_a = torch.mean(a, dim=0)
        g_loss = torch.sum(a * g_loss_multi_dim, keepdim=True, dim=-1)
        return v_loss + g_loss, {
            "train_gradient_loss(unscaled)": g_loss,
            "train_gradient_loss_scaling_factor": mean_a,
        }


class FixedLossScaler(LossScaler):
    def __init__(self, fixed_weight: float):
        self.fixed_weight = fixed_weight

    def scale(self, v_loss: torch.Tensor, g_loss_multi_dim: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        g_loss = torch.sum(g_loss_multi_dim, keepdim=True, dim=-1)
        return v_loss + self.fixed_weight * g_loss, {
            "train_gradient_loss(unscaled)": g_loss,
        }

    def __str__(self):
        return f"FixedLossScaler(fixed_weight={self.fixed_weight})"


class FixedHessianLossScaler(LossScaler):
    def __init__(self, fixed_gradient_weight: float, fixed_hessian_weight: float):
        self.fixed_gradient_weight = fixed_gradient_weight
        self.fixed_hessian_weight = fixed_hessian_weight

    def scale_g_h(
        self,
        v_loss: torch.Tensor,
        g_loss_multi_dim: torch.Tensor,
        h_loss_multi_dim: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        g_loss = torch.sum(g_loss_multi_dim, keepdim=True, dim=-1)
        h_loss = torch.sum(h_loss_multi_dim, keepdim=True, dim=-1)
        return (
            v_loss + self.fixed_gradient_weight * g_loss + self.fixed_hessian_weight * h_loss,
            {
                "train_gradient_loss(unscaled)": g_loss,
                "train_hessian_loss(unscaled)": h_loss,
            },
        )

    def __str__(self):
        return (
            f"FixedHessianLossScaler(fixed_gradient_weight={self.fixed_gradient_weight},"
            f" fixed_hessian_weight={self.fixed_hessian_weight})"
        )


class PicardSolutionGradientWrapper(PicardBaseSolution):
    train_loss_key = "train_total_loss"

    @classmethod
    def construct_solution(cls, runner):
        solution = runner.get_solution_plain()
        sol_jac = cls(solution)
        if isinstance(sol_jac.scaler, FixedLossScaler) and sol_jac.scaler.fixed_weight <= 1e-9:
            return solution
        return sol_jac

    def __init__(self, solution: PicardBaseSolution):
        super().__init__(solution.equation, solution.train_cfg)
        self.solution = solution
        self.jac_fn = vmap(jacrev(self.forward_with_aux, has_aux=True))
        self.hessian_fn = vmap(hessian(self.forward))
        self.num_hess_samples = self.solution.train_cfg.NUM_HESS_SAMPLES
        self.beta = self.solution.beta
        loss_cfg = self.solution.train_cfg.LOSS
        self.scaler_cfg = loss_cfg.SCALER
        self.nx = self.solution.equation.nx
        assert self.num_hess_samples <= self.nx**2
        if self.scaler_cfg.cls is None:
            self.scaler = FixedLossScaler(1.0)
        else:
            self.scaler: LossScaler = LossScaler.get_class(self.scaler_cfg.cls)(**self.scaler_cfg.kwargs)
        print(f"Using {self.scaler}")

        self.use_aux_loss = solution.train_cfg.LOSS.use_aux_loss
        self.weight_aux_loss = solution.train_cfg.LOSS.weight_aux_loss
        if self.use_aux_loss:
            self.jac_u_fn = vmap(jacrev(self.solution_only_u, has_aux=False))

    def __getstate__(self):
        # jac_fn depends on `forward_with_aux` which is causing conflicts between:
        #  1. the `forward_with_aux` member function from the class `PicardSolutionGradientWrapper.forward_with_aux`
        #  2. the `forward_with_aux` as a bind method `self.forward_with_aux`
        state = super().__getstate__()
        del state["jac_fn"]
        del state["hessian_fn"]
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.jac_fn = vmap(jacrev(self.forward_with_aux, has_aux=True))
        self.hessian_fn = vmap(hessian(self.forward))

    def forward_with_aux(self, tx):
        return self.solution(tx), self.solution(tx)

    def solution_only_u(self, tx):
        sol = self.solution(tx)
        return torch.narrow(sol, dim=-1, start=0, length=1).view(-1, 1)

    @count_cuda_time_wrapper
    def training_step(self, batch, batch_idx):
        r"""
        loss = v_loss + gd_loss
        v_loss = \mean_{t,x} weight(t,x) * (u_NN(t,x) - u(t,x))^2
        gd_loss = \mean_{t,x} weight(t,x) * ||u_NN_x(t,x) - u_x(t,x)||^2
        ---
        Consider a simple example u(x) = 0.5x^TAx + b^Tx + c
        u_x = Ax + b
        learn u_NN(x) = 0.5x^TA'x + b'x + c' by taking the derivative of the loss terms
            |x^T(A-A')x + (b-b')^Tx + (c-c')|^2, ||(A-A')x + (b-b')||^2
        """
        tx, y = batch
        y_u, y_ux = torch.narrow(y, dim=-1, start=0, length=1), torch.narrow(y, dim=-1, start=1, length=self.nx)
        # weight has shape (batch_size, 1)
        weight = torch.exp(torch.narrow(tx, dim=-1, start=0, length=1) * self.beta)

        if self.solution.output_dim == self.equation.nx:
            u_x = self.solution(tx)
            v_loss = torch.zeros(1, device=tx.device, dtype=tx.dtype)
        else:
            if self.solution.output_dim == (self.equation.nx + 1):
                u_u_x = self.solution(tx)
                u, u_x = u_u_x[:, 0:1], u_u_x[:, 1:]
                if self.use_aux_loss:
                    u_tx_net = self.jac_u_fn(tx)
                    u_x_net = u_tx_net.view(*tx.size())[:, 1:]
                    aux_loss = torch.mean(self.loss_fn(u_x_net - u_x), dim=0)
            elif self.solution.output_dim == 1:
                u_tx, u = self.jac_fn(tx)
                u_x = u_tx.view(*tx.size())[:, 1:]
                u = u.view(u_tx.size(0), 1)  # for enforcing terminal
            else:
                raise ValueError(f"Unknown output_dim: {self.solution.output_dim}")
            v_loss = torch.mean(weight * self.loss_fn(u - y_u), dim=0)

        gd_loss_multi_dim = torch.mean(weight * self.loss_fn(u_x - y_ux), dim=0)
        if self.use_aux_loss:
            gd_loss_multi_dim = gd_loss_multi_dim + self.weight_aux_loss * aux_loss
            self.log("aux_loss", aux_loss.mean(dim=-1), prog_bar=False)
            self.log("gd + aux_loss", gd_loss_multi_dim.mean(dim=-1), prog_bar=False)
        # loss_summer arguments: (1, 1), (1, nx)
        loss, info = self.scaler.scale(v_loss, gd_loss_multi_dim)
        self.log("train_value_loss", v_loss, prog_bar=False)
        self.log(self.train_loss_key, loss, prog_bar=True)
        self.log_dict(info, prog_bar=False)
        return loss

    def forward(self, tx):
        return self.solution(tx)


class PicardSolutionGradientHessianWrapper(PicardSolutionGradientWrapper):
    @count_cuda_time_wrapper
    def training_step(self, batch, batch_idx):
        r"""
        loss = v_loss + g_loss + h_loss
        v_loss = \mean_{t,x} weight(t,x) * (u_NN(t,x) - u(t,x))^2
        g_loss = \mean_{t,x} weight(t,x) * ||u_NN_x(t,x) - u_x(t,x)||^2
        h_loss = \mean_{t,x} weight(t,x) * ||u_NN_xx(t,x) - u_hessian(t,x)||^2
        """
        tx, y = batch
        y_u, y_ux, y_uh = (
            torch.narrow(y, dim=-1, start=0, length=1),
            torch.narrow(y, dim=-1, start=1, length=self.nx),
            torch.narrow(y, dim=-1, start=1 + self.nx, length=self.nx**2),
        )

        # calculate u and its derivatives
        u_tx, u = self.jac_fn(tx)
        u_x = u_tx.view(*tx.size())[:, 1:]

        # weight has shape (batch_size, 1)
        weight = torch.exp(torch.narrow(tx, dim=-1, start=0, length=1) * self.beta)
        v_loss = torch.mean(weight * self.loss_fn(u - y_u), dim=0)
        g_loss_multi_dim = torch.mean(weight * self.loss_fn(u_x - y_ux), dim=0)

        u_tx_hessian = self.hessian_fn(tx)
        u_hessian = u_tx_hessian[:, :, 1:, 1:].reshape(tx.size(0), self.nx * self.nx)
        diff = u_hessian - y_uh
        if self.num_hess_samples > 0:
            random_indices = random.sample(range(self.nx * self.nx), self.num_hess_samples)
            random_indices_tensor = torch.tensor(random_indices, dtype=torch.long, device=y_uh.device)
            diff = torch.index_select(diff, dim=1, index=random_indices_tensor)
        h_loss_multi_dim = torch.mean(weight * self.loss_fn(diff), dim=0)

        # loss_summer arguments: (1, 1), (1, nx), (1, nx*nx)
        loss, info = self.scaler.scale_g_h(v_loss, g_loss_multi_dim, h_loss_multi_dim)
        self.log("train_value_loss", v_loss, prog_bar=False)
        self.log("train_gradient_loss", g_loss_multi_dim.mean(), prog_bar=False)
        self.log("train_hessian_loss", h_loss_multi_dim.mean(), prog_bar=False)
        self.log(self.train_loss_key, loss, prog_bar=True)
        self.log_dict(info, prog_bar=False)
        return loss

    def forward(self, tx):
        return self.solution(tx)
