import abc
import time
from itertools import chain
from typing import Optional

import lightning.pytorch as pl
import numpy as np
import picard.equations as eqs
import torch
from picard.utils import (
    ZeroFunction,
    compute_metrics,
    hutchinson_trace_estimation_batch,
    get_laplacian,
    count_cuda_time_wrapper,
    compute_grad_metrics,
)
from tensorboardX import SummaryWriter
from yacs.config import CfgNode


class LossFnLinearClip(torch.nn.Module):
    def __init__(self, clip: float):
        super().__init__()
        self.clip = torch.scalar_tensor(clip)

    # noinspection PyTypeChecker
    def forward(self, x):
        return torch.where(
            torch.abs(x) < self.clip,
            torch.square(x),
            2 * self.clip * torch.abs(x) - self.clip ** 2,
        )


class PicardBaseSolution(pl.LightningModule):
    """
    It is required that all Picard solutions inherit from this class and implement the following methods:
    - forward:
        the forward method of the network, which takes tx as input and returns value of the solution only.
    - construct_solution:
        a class method that takes a PicardRunner instance and returns an instance of the solution.
    Consider override:
    - training_step:
        if the loss is computed in a different way.
    - get_parameters_to_train:
        if not all parameters are to be trained.
    Should NOT override:
    - configure_optimizers:
        we support build optimizer and scheduler from config.
    """

    train_loss_key = "train_loss"

    def __init__(self, equation: eqs.Equation, train_cfg: CfgNode):
        super().__init__()
        self.save_hyperparameters()
        self.equation = equation
        self.train_cfg = train_cfg
        self.register_buffer("beta", torch.scalar_tensor(train_cfg.LOSS.beta))

        loss_fn_cfg = train_cfg.LOSS.FN
        if loss_fn_cfg.cls is None:
            self.loss_fn = torch.square
        else:
            kwargs = loss_fn_cfg.kwargs
            clip = kwargs["clip"]
            self.loss_fn = LossFnLinearClip(clip)

    @classmethod
    @abc.abstractmethod
    def construct_solution(cls, runner):
        pass

    @count_cuda_time_wrapper
    def training_step(self, batch, batch_idx):
        tx, y = batch
        y_hat = self(tx)
        weight = torch.exp(torch.narrow(tx, dim=-1, start=0, length=1) * self.beta)
        loss = torch.mean(weight * self.loss_fn(y_hat - y[:, :1]), dim=0)
        self.log(self.train_loss_key, loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # a dummy implementation
        pass

    def get_parameters_to_train(self):
        return self.parameters()

    def configure_optimizers(self):
        opt_cfg = self.train_cfg.OPTIMIZER
        optimizer_cls = getattr(torch.optim, opt_cfg.cls)
        optimizer = optimizer_cls(self.get_parameters_to_train(), **opt_cfg.kwargs)
        scheduler_cfg = opt_cfg.SCHEDULER
        if scheduler_cfg.cls is not None:
            scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_cfg.cls)
            # because we use `step` as interval
            # we set a default `patience` to 512
            scheduler_kws = {}
            if scheduler_cls == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler_kws.update({"patience": 512})
            scheduler_kws.update(scheduler_cfg.kwargs)
            scheduler = scheduler_cls(optimizer, **scheduler_kws)
            self.print(f"Using scheduler {scheduler_cls.__name__} with kwargs {scheduler_kws}.")
            scheduler_config = {
                "interval": "step",
                "strict": True,
                "monitor": self.train_loss_key,
            }
            scheduler_config.update(scheduler_cfg.config)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    **scheduler_config,
                },
            }
        return optimizer


def construct_mlp(n_in: int, n_out: int, n_neurons: list, activations: list, bound: Optional[float]):
    assert len(n_neurons) == len(activations)
    layers = []
    n_neurons = [n_in] + n_neurons
    for i in range(len(activations)):
        layers.append(torch.nn.Linear(n_neurons[i], n_neurons[i + 1]))
        layers.append(getattr(torch.nn, activations[i])())
    layers.append(torch.nn.Linear(n_neurons[-1], n_out))
    # clamp the output
    if bound is not None:
        assert bound > 0
        layers.append(torch.nn.Hardtanh(-bound, bound))
    return torch.nn.Sequential(*layers)


class PISGradNet(torch.nn.Module):
    """
    PyTorch implementation of the PIS Grad Network.

    This network mimics the behavior of the original Haiku/JAX version. The ULA gradients
    (which were detached in the original implementation) are now used directly. The network
    includes:
      - A learnable timestep phase parameter.
      - A fixed timestep coefficient computed as a linspace.
      - A time embedding network (t_encoder) for non-gradient parts.
      - A smoothing network (smooth_net) to compute a smooth factor.
      - A main network (nn_module) that takes the concatenation of state and time embedding.

    The forward pass computes:
        output = smooth * dot(net_output, x) + (1 - smooth) * residual
    where smooth is computed from the smoothing function.
    """

    def __init__(self, hidden_shapes: list, dim: int, g0, T=1.0):
        """
        Args:
            hidden_shapes (list): List of hidden layer sizes (e.g., [64, 64]).
            dim (int): Dimension of the state x.
            g0: Terminal func of the problem.
        """
        super().__init__()

        # Save network architecture parameters.
        self.hidden_shapes = hidden_shapes
        self.n_layers = len(hidden_shapes)
        self.dim = dim
        self.act = torch.nn.ELU()

        # In this setting, we fix the channels as 64
        self.channels = 64

        # Learnable parameter for timestep phase, shape [1, channels].
        self.timestep_phase = torch.nn.Parameter(torch.zeros(1, self.channels))

        # Fixed timestep coefficients as a linspace from 0.1 to 1000, shape [1, channels].
        self.register_buffer("timestep_coeff", torch.linspace(0.1, 100.0, steps=self.channels).unsqueeze(0))

        # Time embedding network (t_encoder): from 2*channels -> channels.
        self.t_encoder = torch.nn.Sequential(
            torch.nn.Linear(2 * self.channels, self.channels),
            self.act,
            torch.nn.Linear(self.channels, self.channels),
        )

        # Smoothing network (smooth_net):
        # It consists of an initial Linear layer, then n_layers blocks of (activation + Linear),
        # followed by one activation and a final Linear layer that outputs a vector of size dim.
        # In the smoothing_function, only the first element is used.
        layers_smooth = []
        layers_smooth.append(torch.nn.Linear(2 * self.channels, self.channels))
        for _ in range(self.n_layers):
            layers_smooth.append(self.act)
            layers_smooth.append(torch.nn.Linear(self.channels, self.channels))
        layers_smooth.append(self.act)
        layers_smooth.append(torch.nn.Linear(self.channels, dim))
        self.smooth_net = torch.nn.Sequential(*layers_smooth)

        # Main network (nn_module):
        # It takes as input the concatenation of x (dim) and the time embedding (channels),
        # then passes through a series of layers specified by hidden_shapes, and outputs a vector of size dim.
        net_layers = []
        in_dim = self.dim + self.channels  # Input: state concatenated with time embedding.
        for hs in self.hidden_shapes:
            net_layers.append(torch.nn.Linear(in_dim, hs))
            net_layers.append(self.act)
            in_dim = hs
        net_layers.append(torch.nn.Linear(in_dim, self.dim))
        self.nn_module = torch.nn.Sequential(*net_layers)

        self.g0 = g0
        self.T = T

    def get_pis_timestep_embedding(self, lbd: torch.Tensor) -> torch.Tensor:
        """
        Computes the timestep embedding by concatenating sine and cosine embeddings.

        Args:
            lbd (torch.Tensor): Tensor of shape [batch_size, 1].

        Returns:
            torch.Tensor: Embedding tensor of shape [batch_size, 2*channels].
        """
        squeeze_back = lbd.ndim == 1
        # Compute argument: shape [batch_size, channels].
        arg = self.timestep_coeff * lbd + self.timestep_phase
        sin_embed = torch.sin(arg)
        cos_embed = torch.cos(arg)
        # Concatenate along the last dimension.
        embed = torch.cat([sin_embed, cos_embed], dim=-1)
        if squeeze_back:
            embed = embed.squeeze(0)
        return embed

    def smoothing_function(self, lbd: torch.Tensor) -> torch.Tensor:
        """
        Computes the smoothing factor based on the timestep embedding.

        Args:
            lbd (torch.Tensor): Tensor of shape [batch_size] or [batch_size, 1].

        Returns:
            torch.Tensor: A smoothing scalar for each batch, shape [batch_size].
        """
        if lbd.ndim == 1:
            lbd = lbd.unsqueeze(-1)  # Convert to shape [batch_size, 1].
        lbd_emb = self.get_pis_timestep_embedding(lbd)  # [batch_size, 2*channels]
        zero_val = torch.zeros_like(lbd)  # [batch_size, 1]
        zero_emb = self.get_pis_timestep_embedding(zero_val)
        out_lbd = self.smooth_net(lbd_emb)  # [batch_size, dim]
        out_zero = self.smooth_net(zero_emb)  # [batch_size, dim]
        # Only the first element is used to compute the smoothing factor.
        return out_lbd[..., 0:1] - out_zero[..., 0:1]

    def forward(self, tx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            lbd (torch.Tensor): Timestep values, shape [batch_size] or [batch_size, 1].
            x (torch.Tensor): State tensor, shape [batch_size, dim].

        Returns:
            torch.Tensor: Output tensor, shape [batch_size].
        """
        lbd, x = tx[..., 0:1], tx[..., 1:]

        lbd = self.T - lbd

        # Compute the smoothing factor.
        smooth = self.smoothing_function(lbd)  # Shape [batch_size, 1]

        # Compute the time embedding.
        t_emb = self.get_pis_timestep_embedding(lbd)  # [batch_size, 2*channels]
        t_emb = self.t_encoder(t_emb)  # [batch_size, channels]

        # Concatenate state and time embedding, then pass through the main network.
        net_input = torch.cat([t_emb, x], dim=-1)  # [batch_size, dim + channels]
        net_out = self.nn_module(net_input)  # [batch_size, dim]

        # Compute dot product between network output and x.
        sp_out = torch.sum(net_out * x, dim=-1, keepdim=True)  # [batch_size, 1]

        # Final output: weighted combination of the network output and the residual.
        decay_t = torch.exp(-0.5 * lbd)
        residual = self.g0(decay_t * x)
        out = smooth * sp_out + (1.0 - smooth) * residual
        return out


class PicardSolution(PicardBaseSolution):
    @classmethod
    def construct_solution(cls, runner):
        return cls(
            equation=runner.equation,
            network_cfg=runner.cfg.NETWORK,
            train_cfg=runner.cfg.TRAIN,
        )

    def __init__(self, equation: eqs.Equation, network_cfg: CfgNode, train_cfg: CfgNode):
        super().__init__(equation, train_cfg)
        self.save_hyperparameters(ignore=("loss_cfg", "equation"))
        if network_cfg.TYPE == "Value":
            self.output_dim = self.equation.nu
        elif network_cfg.TYPE == "ValueGradient":
            self.output_dim = self.equation.nu + self.equation.nx
        elif network_cfg.TYPE == "OnlyGradient":
            self.output_dim = self.equation.nx
        else:
            raise ValueError(f"Unknown network type {network_cfg.TYPE}")

        if network_cfg.PISGRADNET:
            self.model = PISGradNet(
                hidden_shapes=network_cfg.NEURONS, dim=self.equation.nx, g0=self.equation.g, T=self.equation.T
            )
        else:
            self.model = construct_mlp(
                1 + self.equation.nx,
                self.output_dim,
                network_cfg.NEURONS,
                network_cfg.ACTIVATIONS,
                network_cfg.BOUND,
            )

    def forward(self, tx):
        return self.model(tx)


class ZeroSolution(torch.nn.Module):
    def __init__(self, output_dim: int = 1):
        super().__init__()
        self.output_dim = output_dim

    # noinspection PyMethodMayBeStatic
    def forward(self, tx):
        return ZeroFunction.apply(tx).expand(-1, self.output_dim)


class GxSolution(torch.nn.Module):
    def __init__(self, equation: eqs.Equation):
        super().__init__()
        self.equation = equation

    def forward(self, tx):
        return self.equation.g(tx[:, 1:])


class PINNSolution:
    def __init__(self, network, equation: eqs.Equation, cfg: CfgNode):
        super().__init__()
        lr = 0.001
        self.optimizer = torch.optim.Adam(
            network.parameters(), lr=lr
        )  # NOTE: optimizer is fixed, which may be different from Picard
        self.num_epochs = cfg.TRAIN.N_EPOCHS
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.num_v_samples = cfg.METHOD.num_v_samples
        self.DEVICE = cfg.DATA.DEVICE
        self.n_eval_points = cfg.EVAL.L2_N_POINTS
        self.equation = equation.to(device=self.DEVICE)
        self.dnn = network.to(device=self.DEVICE)
        self.enforce_terminal = True if ("EnforceTerminal" in cfg.NETWORK.cls) or (cfg.NETWORK.PISGRADNET) else False
        self.log_interval = cfg.EVAL.FREQ
        self.terminal_weight = 0.0 if self.enforce_terminal else cfg.TRAIN.LOSS.beta
        self.eval_gradient = cfg.EVAL.TEST_GRAD
        self.eval_hessian = cfg.EVAL.TEST_HESSIAN

        from datetime import datetime

        now = datetime.now()
        formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
        if cfg.EQUATION.cls == "Cha":
            eq_name = f"{cfg.EQUATION.cls}_alpha={self.equation.alpha}"
        else:
            eq_name = cfg.EQUATION.cls
        self.model_path = f"pinn/{eq_name}_dim={self.equation.nx}_T={self.equation.T}_v={self.num_v_samples}_tw={self.terminal_weight}_lr={lr}_{formatted_now}"
        self.writer = SummaryWriter(log_dir=self.model_path)
        try:
            t_test = torch.zeros(1, 1).to(self.DEVICE)
            x_test = torch.zeros(1, self.equation.nx).to(self.DEVICE)
            self.equation.exact_solution(t_test, x_test)
            print("Exact solution implemented, using exact solution for evaluation")
            self.exact_solution = True
        except NotImplementedError:
            print("Exact solution not implemented, using data for evaluation")
            self.exact_solution = False

        hessian_approximation = cfg.DATA.HESSIAN_APPROXIMATION
        self.hessian_approximation_ctx = (
            {"config": hessian_approximation} if hessian_approximation.method is not None else None
        )
        self.hessian_approximation_config = hessian_approximation
        if self.hessian_approximation_ctx is not None:
            print(f"Hessian approximation is enabled with {hessian_approximation}")
            assert (
                    hessian_approximation.method in self.equation.supported_approximate_methods
            ), f"Current equation does not support the method {hessian_approximation.method}"

    def train(self):
        # if equation has exact solution
        if not self.exact_solution:
            load_file = (
                f"../../data/{type(self.equation).__name__}_nx={self.equation.nx}_T={self.equation.T}_N=7_100.npy"
            )
            print("Loading evaluation data from", load_file)
            data = np.load(load_file)
            t_eval = torch.tensor(data[:, 0:1]).to(self.DEVICE)
            x_eval = torch.tensor(data[:, 1:-1]).to(self.DEVICE)
            u_eval = torch.tensor(data[:, -1:]).to(self.DEVICE)

        total_time = 0
        for epoch in range(self.num_epochs):
            time_0 = time.time()
            t = self.equation.T * torch.rand(self.batch_size, 1).to(self.DEVICE)
            x = self.equation.sample_x(t)  # assume the sampling time in pinn can be omitted.
            t.requires_grad_()
            x.requires_grad_()

            u = self.dnn(torch.cat([t, x], dim=-1))
            u_t = torch.autograd.grad(outputs=u.sum(), inputs=t, create_graph=True, retain_graph=True)[0]
            u_x = torch.autograd.grad(outputs=u.sum(), inputs=x, create_graph=True, retain_graph=True)[0]

            if self.equation.has_hessian_term:
                if self.hessian_approximation_ctx is None:
                    u_xx = torch.zeros(self.batch_size, x.size(1), x.size(1), device=x.device)
                    for i in range(u_x.size(1)):
                        grad_grad = torch.autograd.grad(
                            u_x[:, i],
                            x,
                            grad_outputs=torch.ones_like(u_x[:, i]),
                            create_graph=True,
                            only_inputs=True,
                        )[0]
                        u_xx[:, i, :] = grad_grad
                else:
                    assert self.hessian_approximation_config.method == "SDGD"
                    v = self.hessian_approximation_config.kwargs["v"]
                    indices = torch.multinomial(torch.ones_like(x), v)  # (n_batch, v)
                    indices_effective = indices
                    # indices_effective has shape (N*M, v)
                    u_ii_all = torch.zeros(indices_effective.size(), device=x.device)
                    for i in range(indices_effective.size(1)):
                        active_index = indices_effective[:, i].unsqueeze(dim=-1)  # shape: (n_batch*n_estimate,)
                        # u_ii's shape: (n_batch*n_estimate, nx)
                        active_u_x = torch.gather(u_x, 1, active_index)
                        u_ii = torch.autograd.grad(
                            active_u_x,
                            x,
                            grad_outputs=torch.ones_like(active_u_x),
                            create_graph=True,
                            only_inputs=True,
                        )[0]
                        # get the active u_ii
                        u_ii_all[:, i] = torch.gather(u_ii, 1, active_index).squeeze()
                    u_xx = u_ii_all

            else:
                if self.num_v_samples > 0:
                    u_xx = hutchinson_trace_estimation_batch(self.dnn, t, x, self.num_v_samples)
                else:
                    u_xx = get_laplacian(x, u)

            loss_in = ((self.equation.pinn_function(t, x, u, u_t, u_x, u_xx)) ** 2).mean()

            if self.enforce_terminal:
                loss = loss_in
            else:
                T = self.equation.T * torch.ones_like(t).to(self.DEVICE)
                x_T = self.equation.sample_x(T)
                u_T = self.dnn(torch.cat([T, x_T], dim=-1))
                loss_T = ((u_T - self.equation.g(x_T)) ** 2).mean()
                loss = loss_in + self.terminal_weight * loss_T

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_time += time.time() - time_0
            if epoch % self.log_interval == 0:
                print(epoch, total_time, loss.item())
                if self.exact_solution:
                    t_eval = self.equation.T * torch.rand(self.n_eval_points, 1).to(self.DEVICE)
                    x_eval = self.equation.sample_x(t_eval)
                    u_eval = self.equation.exact_solution(t_eval, x_eval)

                if self.eval_gradient:
                    x_eval.requires_grad_()
                u_pred = self.dnn(torch.cat([t_eval, x_eval], dim=-1))
                _, MArE, rRMSE, rMAE = compute_metrics(u_pred, u_eval)
                self.writer.add_scalar("loss total", loss.item(), global_step=epoch)
                self.writer.add_scalar("Time", total_time, global_step=epoch)
                self.writer.add_scalar("MArE", MArE.item(), global_step=epoch)
                self.writer.add_scalar("rRMSE", rRMSE.item(), global_step=epoch)
                self.writer.add_scalar("rMAE", rMAE.item(), global_step=epoch)
                if self.eval_gradient:
                    retain_graph = self.eval_hessian
                    u_x_pred = torch.autograd.grad(
                        outputs=u_pred.sum(), inputs=x_eval, create_graph=True, retain_graph=retain_graph
                    )[0]
                    u_x_eval = self.equation.u_x(t_eval, x_eval)
                    _, MArEg, rRMSEg, rMAEg = compute_grad_metrics(u_x_pred, u_x_eval)
                    self.writer.add_scalar("MArE gradient", MArEg.item(), global_step=epoch)
                    self.writer.add_scalar("rRMSE gradient", rRMSEg.item(), global_step=epoch)
                    self.writer.add_scalar("rMAE gradient", rMAEg.item(), global_step=epoch)
                    # if self.eval_hessian:
                    #     print(u_x_pred.shape, x_eval.shape)
                    #     u_xx_pred = torch.autograd.grad(outputs=u_x_pred.sum(), inputs=x_eval, create_graph=True)[0]
                    #     u_xx_eval = self.equation.u_hessian(t_eval, x_eval)
                    #     print(u_xx_pred.shape, u_xx_eval.shape)
                    #     print("Hessian error:", torch.norm(u_xx_pred - u_xx_eval) / torch.norm(u_xx_eval))
                    #     _, MArEh, rRMSEh, rMAEh = compute_grad_metrics(u_xx_pred, u_xx_eval)
                    #     self.writer.add_scalar("MArE hessian", MArEh.item(), global_step=epoch)
                    #     self.writer.add_scalar("rRMSE hessian", rRMSEh.item(), global_step=epoch)
                    #     self.writer.add_scalar("rMAE hessian", rMAEh.item(), global_step=epoch)
                # if not self.enforce_terminal:
                #     self.writer.add_scalar("loss in", loss_in.item(), global_step=epoch)
                #     self.writer.add_scalar("loss terminal", loss_T.item(), global_step=epoch)

                # save model
                torch.save(self.dnn.state_dict(), f"{self.model_path}/pinn_{epoch}.pt")


class DiffusionSolution:
    def __init__(self, network, equation: eqs.Equation, cfg: CfgNode):
        super().__init__()
        lr = 0.001
        self.optimizer = torch.optim.Adam(
            network.parameters(), lr=lr
        )  # NOTE: optimizer is fixed, which may be different from Picard
        self.num_epochs = cfg.TRAIN.N_EPOCHS
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.K = cfg.METHOD.K
        self.dt = cfg.METHOD.dt
        self.time_to_go = self.K * self.dt
        self.DEVICE = cfg.DATA.DEVICE
        self.n_eval_points = cfg.EVAL.L2_N_POINTS
        self.equation = equation.to(device=self.DEVICE)
        self.dnn = network.to(device=self.DEVICE)
        self.enforce_terminal = True if "EnforceTerminal" in cfg.NETWORK.cls else False
        self.log_interval = cfg.EVAL.FREQ
        self.terminal_weight = 0.0 if self.enforce_terminal else cfg.TRAIN.LOSS.beta
        self.eval_gradient = cfg.EVAL.TEST_GRAD
        from datetime import datetime

        now = datetime.now()
        formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
        if cfg.EQUATION.cls == "Cha":
            eq_name = f"{cfg.EQUATION.cls}_alpha={self.equation.alpha}"
        else:
            eq_name = cfg.EQUATION.cls
        model_path = f"diffusion/{eq_name}_dim={self.equation.nx}_T={self.equation.T}_K={self.K}_dt={self.dt}_tw={self.terminal_weight}_lr={lr}_bs={self.batch_size}_{formatted_now}"
        self.writer = SummaryWriter(log_dir=model_path)
        try:
            t_test = torch.zeros(1, 1).to(self.DEVICE)
            x_test = torch.zeros(1, self.equation.nx).to(self.DEVICE)
            self.equation.exact_solution(t_test, x_test)
            print("Exact solution implemented, using exact solution for evaluation")
            self.exact_solution = True
        except NotImplementedError:
            print("Exact solution not implemented, using data for evaluation")
            self.exact_solution = False

    def train(self):
        # if equation has exact solution
        if not self.exact_solution:
            load_file = (
                f"../../data/{type(self.equation).__name__}_nx={self.equation.nx}_T={self.equation.T}_N=7_100.npy"
            )
            print("Loading evaluation data from", load_file)
            data = np.load(load_file)
            t_eval = torch.tensor(data[:, 0:1]).to(self.DEVICE)
            x_eval = torch.tensor(data[:, 1:-1]).to(self.DEVICE)
            u_eval = torch.tensor(data[:, -1:]).to(self.DEVICE)

        total_time = 0
        for epoch in range(self.num_epochs):
            time_0 = time.time()
            t0 = self.equation.T * torch.rand(self.batch_size, 1).to(self.DEVICE)
            x0 = self.equation.sample_x(t0)
            ts = torch.zeros(self.batch_size, self.K + 1, 1).to(self.DEVICE)
            xts = torch.zeros(self.batch_size, self.K + 1, self.equation.nx).to(self.DEVICE)
            ts[:, 0, :] = t0
            xts[:, 0, :] = x0

            # replace dts
            mask = ((t0 + self.time_to_go) <= self.equation.T).reshape(-1)
            dts = torch.ones_like(ts)[:, :-1, :] * self.dt
            replace_dts = ((self.equation.T - t0[~mask]) / self.K).unsqueeze(1).repeat(1, self.K, 1)
            dts[~mask] = replace_dts

            for k in range(self.K):
                ts[:, k + 1, :] = ts[:, k, :] + dts[:, k, :]
                xts[:, k + 1, :] = self.equation.sample_x_ts(
                    ts[:, k, :], ts[:, k + 1, :], xts[:, k, :], return_dW=False
                )
            ts_flat = ts.reshape(-1, 1)
            xts_flat = xts.reshape(-1, self.equation.nx)
            ts_flat.requires_grad_()
            xts_flat.requires_grad_()

            # compute diffusion loss
            vts_flat = self.dnn(torch.cat([ts_flat, xts_flat], dim=-1))
            vts_grad_flat = torch.autograd.grad(vts_flat.sum(), xts_flat, retain_graph=True, create_graph=True)[0]
            if self.equation.has_gradient_term:
                fs_flat = self.equation.ff(ts_flat, xts_flat, vts_flat, vts_grad_flat)
            else:
                fs_flat = self.equation.f(ts_flat, xts_flat, vts_flat)

            vts = vts_flat.reshape(self.batch_size, self.K + 1, 1)
            vts_grad = vts_grad_flat.reshape(self.batch_size, self.K + 1, self.equation.nx)
            fs = fs_flat.reshape(self.batch_size, self.K + 1, 1)
            dws = torch.diff(xts, dim=1)
            vts_pred = vts[:, 0, :] - (fs[:, :-1, :] * dts).sum(1) + (vts_grad[:, :-1, :] * dws).sum(-1).sum(-1, True)

            # mask with ts larger than equation.T
            loss_in = ((vts[:, -1, :] - vts_pred) ** 2).mean()

            if self.enforce_terminal:
                loss = loss_in
            else:
                T = self.equation.T * torch.ones_like(t0).to(self.DEVICE)
                x_T = self.equation.sample_x(T)
                # u_T = self.dnn(T, x_T)
                u_T = self.dnn(torch.cat([T, x_T], dim=-1))

                loss_T = ((u_T - self.equation.g(x_T)) ** 2).mean()
                loss = loss_in + self.terminal_weight * loss_T

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_time += time.time() - time_0
            if epoch % self.log_interval == 0:
                print(epoch, total_time, loss.item())
                if self.exact_solution:
                    t_eval = self.equation.T * torch.rand(self.n_eval_points, 1).to(self.DEVICE)
                    x_eval = self.equation.sample_x(t_eval)
                    u_eval = self.equation.exact_solution(t_eval, x_eval)

                if self.eval_gradient:
                    x_eval.requires_grad_()
                u_pred = self.dnn(torch.cat([t_eval, x_eval], dim=-1))
                _, MArE, rRMSE, rMAE = compute_metrics(u_pred, u_eval)
                self.writer.add_scalar("loss total", loss.item(), global_step=epoch)
                self.writer.add_scalar("Time", total_time, global_step=epoch)
                self.writer.add_scalar("MArE", MArE.item(), global_step=epoch)
                self.writer.add_scalar("rRMSE", rRMSE.item(), global_step=epoch)
                self.writer.add_scalar("rMAE", rMAE.item(), global_step=epoch)
                if self.eval_gradient:
                    u_x_pred = torch.autograd.grad(outputs=u_pred.sum(), inputs=x_eval, create_graph=True)[0]
                    u_x_eval = self.equation.u_x(t_eval, x_eval)
                    _, MArEg, rRMSEg, rMAEg = compute_grad_metrics(u_x_pred, u_x_eval)
                    self.writer.add_scalar("MArE gradient", MArEg.item(), global_step=epoch)
                    self.writer.add_scalar("rRMSE gradient", rRMSEg.item(), global_step=epoch)
                    self.writer.add_scalar("rMAE gradient", rMAEg.item(), global_step=epoch)
                if not self.enforce_terminal:
                    self.writer.add_scalar("loss in", loss_in.item(), global_step=epoch)
                    self.writer.add_scalar("loss terminal", loss_T.item(), global_step=epoch)


class MLPEnforceTerminal(torch.nn.Module):
    def __init__(self, equation, cfg, t):
        super().__init__()
        self.equation = equation
        self.T = equation.T
        self.t = t
        self.model = construct_mlp(
            self.equation.nx,
            self.equation.nu,
            cfg.NETWORK.NEURONS,
            cfg.NETWORK.ACTIVATIONS,
            cfg.NETWORK.BOUND,
        ).to(device=cfg.DATA.DEVICE)

    def forward(self, x):
        return self.equation.g(x) + (self.T - self.t) * self.model(x)


class MLPGradEnforceTerminal(torch.nn.Module):
    def __init__(self, equation, cfg, t):
        super().__init__()
        self.equation = equation
        self.T = equation.T
        self.t = t
        self.model = construct_mlp(
            self.equation.nx,
            self.equation.nx,
            cfg.NETWORK.NEURONS,
            cfg.NETWORK.ACTIVATIONS,
            cfg.NETWORK.BOUND,
        ).to(device=cfg.DATA.DEVICE)

    def forward(self, x):
        return self.equation.g_x(x) + (self.T - self.t) * self.model(x)


class FullyNonlinearSolution:
    """
    https://arxiv.org/abs/1908.00412
    """

    def __init__(self, equation: eqs.Equation, cfg: CfgNode):
        super().__init__()
        self.num_epochs = cfg.TRAIN.N_EPOCHS
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.DEVICE = cfg.DATA.DEVICE
        self.n_eval_points = cfg.EVAL.L2_N_POINTS
        self.equation = equation.to(device=self.DEVICE)
        self.K = round(self.equation.T / cfg.METHOD.dt)
        self.dt = self.equation.T / self.K
        self.num_sub_iter = cfg.METHOD.num_sub_iter
        self.u_list = []
        self.grad_u_list = []
        self.optimizer_list = []
        self.lr = 0.001
        t = 0
        for _ in range(self.K + 1):
            u_network = MLPEnforceTerminal(self.equation, cfg, t)
            self.u_list.append(u_network)
            grad_u_network = MLPGradEnforceTerminal(self.equation, cfg, t)
            self.grad_u_list.append(grad_u_network)
            self.optimizer_list.append(
                torch.optim.Adam(
                    chain(
                        self.u_list[-1].model.parameters(),
                        self.grad_u_list[-1].model.parameters(),
                    ),
                    lr=self.lr,
                )
            )
            t += self.dt
        self.enforce_terminal = True if "EnforceTerminal" in cfg.NETWORK.cls else False
        self.log_interval = cfg.EVAL.FREQ
        self.terminal_weight = 0.0 if self.enforce_terminal else cfg.TRAIN.LOSS.beta
        self.eval_gradient = cfg.EVAL.TEST_GRAD
        from datetime import datetime

        now = datetime.now()
        formatted_now = now.strftime("%Y-%m-%d_%H_%M_%S")
        eq_name = f"{cfg.EQUATION.cls}_alpha={self.equation.alpha}"
        model_path = f"fully_nonlinear_solver/{eq_name}_dim={self.equation.nx}_T={self.equation.T}_K={self.K}_dt={self.dt}_tw={self.terminal_weight}_lr={self.lr}_bs={self.batch_size}_{formatted_now}"
        self.writer = SummaryWriter(log_dir=model_path)

    def sample_data(self, num_samples):
        t_all = torch.zeros(num_samples, self.K + 1, 1).to(self.DEVICE)
        x_all = torch.zeros(num_samples, self.K + 1, self.equation.nx).to(self.DEVICE)
        dW_all = torch.zeros(num_samples, self.K, self.equation.nx).to(self.DEVICE)
        t = torch.zeros(num_samples, 1).to(self.DEVICE)
        x = self.equation.sample_x0(len(t))
        for kk in range(self.K):
            t_next = t + self.dt
            x_next, dW = self.equation.sample_x_ts(t, t_next, x, return_dW=True)
            t_all[:, kk, :] = t
            x_all[:, kk, :] = x
            dW_all[:, kk, :] = dW * np.sqrt(self.dt)
            t = t_next
            x = x_next
        t_all[:, -1, :] = t
        x_all[:, -1, :] = x
        return dW_all, t_all, x_all

    def get_loss(self, k, x_all, t_all, dW_all):
        x_next = x_all[:, k, :]
        t = t_all[:, k - 1, :]
        x = x_all[:, k - 1, :]
        dW = dW_all[:, k - 1, :]
        x_next.requires_grad_()
        u = self.u_list[k - 1](x)
        u_x = self.grad_u_list[k - 1](x)

        if self.enforce_terminal and (k == self.K):
            u_next = self.equation.g(x_next)
            u_x_next = self.equation.g_x(x_next)
        else:
            u_next = self.u_list[k](x_next)
            u_x_next = self.grad_u_list[k](x_next)
        u_hessian_next = torch.zeros(x.size(0), x.size(1), x.size(1), device=x.device)
        for i in range(u_x_next.size(1)):
            grad_grad = torch.autograd.grad(
                u_x_next[:, i],
                x_next,
                grad_outputs=torch.ones_like(u_x_next[:, i]),
                create_graph=True,
                only_inputs=True,
            )[0]
            u_hessian_next[:, i, :] = grad_grad

        # NOTE: Assume mu = 0.
        f_hat = self.equation.ffh(t, x, u, u_x, u_hessian_next.detach())
        F = u - f_hat * self.dt + (u_x * self.equation.alpha_sqrt * dW).sum(-1, True)

        loss = ((u_next.detach() - F) ** 2).mean()
        return loss

    def train(self):
        total_time = 0
        for epoch in range(self.num_epochs):
            # assert self.enforce_terminal
            # sample valid data
            t = 0
            t_eval_all = torch.zeros(self.K + 1, 100, 1).to(self.DEVICE)
            x_eval_all = torch.zeros(self.K + 1, 100, self.equation.nx).to(self.DEVICE)
            for kk in range(self.K + 1):
                t_eval = t * torch.ones(100, 1).to(self.DEVICE)
                x_eval = self.equation.sample_x(t_eval)
                t += self.dt
                t_eval_all[kk] = t_eval
                x_eval_all[kk] = x_eval

            if not self.enforce_terminal:
                for idx in range(self.num_sub_iter):
                    dW_all, t_all, x_all = self.sample_data(self.batch_size)
                    optimizer = self.optimizer_list[-1]
                    x = x_all[:, -1, :]
                    x.requires_grad_()
                    u = self.u_list[-1](x)
                    u_x = self.grad_u_list[-1](x)
                    loss_in = ((u - self.equation.g(x)) ** 2).mean()
                    loss_grad = ((u_x - self.equation.g_x(x)) ** 2).mean()
                    loss = loss_in + loss_grad * self.dt
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            for k in range(self.K, 0, -1):
                time_0 = time.time()
                optimizer = self.optimizer_list[k - 1]
                # scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
                if k < self.K:
                    self.u_list[k - 1].model.load_state_dict(self.u_list[k].model.state_dict())
                    self.grad_u_list[k - 1].model.load_state_dict(self.grad_u_list[k].model.state_dict())

                for idx in range(self.num_sub_iter):
                    # sample train data
                    dW_all, t_all, x_all = self.sample_data(self.batch_size)
                    # train
                    loss = self.get_loss(k, x_all, t_all, dW_all)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_time += time.time() - time_0
                # valid
                # loss_val = self.get_loss(k, x_val, t_val, dW_val)
                # scheduler.step(loss_val)
                print("-" * 40)
                print(
                    epoch,
                    k,
                    total_time,
                    loss.item(),
                    # loss_val.item(),
                )

                t = 0
                u_eval_all = torch.zeros(self.K + 1, 100).to(self.DEVICE)
                u_pred_all = torch.zeros(self.K + 1, 100).to(self.DEVICE)
                u_eval_grad_all = torch.zeros(self.K + 1, 100, x_all.shape[-1]).to(self.DEVICE)
                u_pred_grad_all = torch.zeros(self.K + 1, 100, x_all.shape[-1]).to(self.DEVICE)
                for idkk, kk in enumerate(range(self.K + 1)):
                    t_eval = t_eval_all[idkk]
                    x_eval = x_eval_all[idkk]
                    x_eval.requires_grad_()
                    u_eval = self.equation.exact_solution(t_eval, x_eval)
                    u_pred = self.u_list[kk](x_eval)
                    u_eval_all[kk] = u_eval.reshape(-1)
                    u_pred_all[kk] = u_pred.reshape(-1)
                    if self.eval_gradient:
                        u_x_pred = self.grad_u_list[kk](x_eval)
                        u_x_eval = self.equation.u_x(t_eval, x_eval)
                        u_eval_grad_all[kk] = u_x_eval
                        u_pred_grad_all[kk] = u_x_pred
                    if kk == k - 1:
                        print(f"Current subnetwork {kk} with {t}")
                        _, MArE, rRMSE, rMAE = compute_metrics(u_pred, u_eval)
                        print("=" * 40)

                u_pred_all = u_pred_all.reshape(-1, 1)
                u_eval_all = u_eval_all.reshape(-1, 1)
                _, MArE, rRMSE, rMAE = compute_metrics(u_pred_all, u_eval_all)
                self.writer.add_scalar("loss total", loss.item(), global_step=k)
                self.writer.add_scalar("Time", total_time, global_step=k)
                self.writer.add_scalar("MArE", MArE.item(), global_step=k)
                self.writer.add_scalar("rRMSE", rRMSE.item(), global_step=k)
                self.writer.add_scalar("rMAE", rMAE.item(), global_step=k)
                if self.eval_gradient:
                    u_pred_grad_all = u_pred_grad_all.reshape(-1, x.shape[1])
                    u_eval_grad_all = u_eval_grad_all.reshape(-1, x.shape[1])
                    _, MArEg, rRMSEg, rMAEg = compute_grad_metrics(u_pred_grad_all, u_eval_grad_all)
                    self.writer.add_scalar("MArE gradient", MArEg.item(), global_step=k)
                    self.writer.add_scalar("rRMSE gradient", rRMSEg.item(), global_step=k)
                    self.writer.add_scalar("rMAE gradient", rMAEg.item(), global_step=k)


def generate_sx_for_integral(equation, tx: torch.Tensor, n_multiples: int, return_dW=False):
    """
    :param tx:
    :param n_multiples:
    :param return_dW: whether to return dW
    :return:
    """
    tx = torch.repeat_interleave(tx, n_multiples, dim=0)
    t, x = torch.narrow(tx, dim=-1, start=0, length=1), torch.narrow(tx, dim=-1, start=1, length=equation.nx)
    # sample s from uniform distribution on [t, T]
    s = torch.rand_like(t) * (equation.T - t) + t
    Xs = equation.sample_x_ts(t, s, x, return_dW=return_dW)
    if return_dW:
        Xs, dW = Xs
        sXs = torch.cat([s, Xs], dim=-1)
        return t, s, Xs, sXs, dW
    sXs = torch.cat([s, Xs], dim=-1)
    return t, s, Xs, sXs


def estimate_integral_with_gradients(
        equation, solution, n_estimate_integral, hessian_approximation_ctx, tx: torch.Tensor
):
    r"""
    We make the assumptions:
      - Sigma = sqrt(alpha) I,
      - D = I.
    The second assumption can be derived by assuming that `\mu_x` and `\Sigma_x` are zeros.
    Therefore, the assumptions boil down to:
      - Sigma = sqrt(alpha) I,
      - `\mu_x` = 0.
    For now, SimpleDiffusionEquation satisfies the above assumptions (actually, it further requires \mu=0).
    In the future, we shall move the computation of Ys to the equation class.
    ---
    The formula:
        v(t, x) = \int_t^T f(s, Xs, u(s,Xs), Sigma^T u_x(s,Xs)) (1, Ys) ds
        Ys = Sigma^T/(s-t)\int_t^s [Sigma^{-1}D_{s,r,x}]^T dWr
           = 1/(s-t)\int_t^s dWr
           = 1/sqrt(s-t) N(0,1)^d
    :param tx:
    :return:
        v(t, x): (n_batch, 1 + nx)
    """
    assert isinstance(equation, eqs.SimpleDiffusionEquation) or isinstance(equation, eqs.OUProcessEquation)
    n_estimate = n_estimate_integral
    n_batch = tx.size(0)
    # we need to compute the gradient of u with respect to Xs
    t, s, Xs, sXs, dW = generate_sx_for_integral(equation, tx, n_estimate, return_dW=True)
    assert hessian_approximation_ctx is None
    Xs.requires_grad_()

    f = solution(sXs)

    x_baseline = tx[:, 1:]
    x_baseline.requires_grad_()
    t_baseline = tx[:, 0:1]
    f_baseline = solution(torch.cat([t_baseline, x_baseline], dim=-1)).repeat(n_estimate, 1)

    Ys = dW / torch.sqrt(s - t) / equation.alpha_sqrt
    # shape of eYs: (n_batch*n_estimate, 1 + nx)
    eYs = torch.cat([torch.ones_like(s), Ys], dim=-1)
    integral = (equation.T - t) * (f - f_baseline) * eYs
    integral[:, 0:1] = integral[:, 0:1] + f_baseline * (equation.T - t)
    integral = integral.view(n_batch, n_estimate, -1)
    integral = torch.sum(integral, dim=1, keepdim=False) / n_estimate
    return integral


def estimate_terminal_with_gradients(equation, n_estimate_terminal, tx: torch.Tensor):
    r"""
    Assume the same assumption as in `estimate_integral_with_gradients`.
    ---
    The formula:
        E[(g(X_(t,T,x)) - g(x)) (1, Y)] + (g(x), 0)
        Y = 1/sqrt(T-t) N(0,1)^d
    :param tx:
    :return:
    """
    assert isinstance(equation, eqs.SimpleDiffusionEquation) or isinstance(equation, eqs.OUProcessEquation)
    n_estimate = n_estimate_terminal
    n_batch = tx.size(0)
    tx = torch.repeat_interleave(tx, n_estimate, dim=0)
    t, x = torch.narrow(tx, dim=-1, start=0, length=1), torch.narrow(tx, dim=-1, start=1, length=equation.nx)
    T = torch.ones_like(t) * equation.T
    XT, dW = equation.sample_x_ts(t, T, x, return_dW=True)  # (n_batch*n_estimate, 1)
    gT = equation.g(XT)  # (n_batch*n_estimate, 1)
    Y = dW / torch.sqrt(T - t) / equation.alpha_sqrt
    eY = torch.cat([torch.ones_like(gT), Y], dim=-1)  # (n_batch*n_estimate, 1+nx)
    # the faster implementation: we do NOT need
    x_single = x.view(n_batch, n_estimate, -1)[:, 0]
    g_single = equation.g(x_single)
    g = torch.repeat_interleave(g_single, n_estimate, dim=0)
    terminal = (gT - g) * eY
    terminal = terminal.view(n_batch, n_estimate, -1).sum(dim=1, keepdim=False) / n_estimate
    terminal[:, 0:1] = terminal[:, 0:1] + g_single
    return terminal
