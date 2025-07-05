import pathlib
from typing import List

import h5py
import torch
import torch.nn as nn

from picard.config import load_cfg
from picard.evaluate import Evaluator
from picard.picard_iteration import PicardRunner
import picard.solution as sols
import picard.solution_enforce_terminal as sols_enforce_terminal


class ExperimentEvaluator:
    def __init__(
        self,
        exp_dir: pathlib.Path,
        n_estimate_terminal: int = 1_000_000,
        n_estimate_integral: int = 1_000_000,
    ):
        self.exp_dir = exp_dir
        self.cfg = load_cfg(PicardRunner.get_config_path(exp_dir))
        self.equation = PicardRunner.get_equation(self.cfg)

        self._solution_cache: List[nn.Module] = [None] * (self.cfg.PICARD.N + 1)
        self._evaluator: List[Evaluator] = [None] * (self.cfg.PICARD.N + 1)
        self.n_estimate_terminal = n_estimate_terminal
        self.n_estimate_integral = n_estimate_integral

    def get_solution(self, i: int):
        assert 0 <= i <= self.cfg.PICARD.N
        if self._solution_cache[i] is not None:
            return self._solution_cache[i]
        if i == 0:
            output_dim = self.equation.nx + 1 if self.cfg.NETWORK.TYPE == "ValueGradient" else 1
            u = (
                sols.GxSolution(self.equation)
                if hasattr(sols_enforce_terminal, self.cfg.NETWORK.cls)
                else sols.ZeroSolution(output_dim)
            )
        else:
            sol_cls = PicardRunner.get_solution_cls(self.cfg)
            try:
                u = sol_cls.load_from_checkpoint(PicardRunner.get_model_checkpoint_path(self.exp_dir, i))
            except TypeError:
                # for backward compatibility: if the signature of PicardSolution has changed,
                # we can construct it manually then load the state dict.
                if sol_cls != sols.PicardSolution:
                    raise ValueError("Other type of solution unsupported!")
                u = sol_cls(
                    self.equation,
                    network_cfg=self.cfg.NETWORK,
                    loss_cfg=self.cfg.TRAIN.LOSS,
                )
                u.load_state_dict(torch.load(PicardRunner.get_model_checkpoint_path(self.exp_dir, i))["state_dict"])
        self._solution_cache[i] = u
        return u

    def get_evaluator(self, i: int):
        if self._evaluator[i] is not None:
            return self._evaluator[i]
        evaluator = Evaluator(
            self.get_solution(i),
            self.equation,
            N=self.cfg.PICARD.N,
            i=i,
            n_estimate_terminal=self.n_estimate_terminal,
            n_estimate_integral=self.n_estimate_integral,
        )
        self._evaluator[i] = evaluator
        return evaluator

    def monte_carlo_at_zero(self):
        nx = self.equation.nx
        for ii in range(self.cfg.PICARD.N + 1):
            evaluator = self.get_evaluator(ii)
            solution = self.get_solution(ii)
            x0 = torch.zeros(1, nx)
            u_data = evaluator.monte_carlo(x0)
            u0 = solution(torch.zeros(1, 1 + nx))[0].item()
            print(f"{ii}: u(0, 0)={u0:.6f}\t Picard Estimate: {u_data.mean():.6f}+-{u_data.std():.4e}")

    def l2(self, n_batch: int):
        for ii in range(1, self.cfg.PICARD.N + 1):
            evaluator = self.get_evaluator(ii)
            t, (err_l2, error, x, u) = evaluator.l2_at_td(n_batch)
            print(f"Iter {ii} (t random): L2 error = {err_l2:.4e}")
            evaluator.plot_error(self.exp_dir, t, x, error, f"iter_{ii:03d}")

        for ii in range(1, self.cfg.PICARD.N + 1):
            evaluator = self.get_evaluator(ii)
            t, (err_l2, error, x, u) = evaluator.l2_at_t0(n_batch)
            print(f"Iter {ii} (t=0): L2 error = {err_l2:.4e}")
            evaluator.plot_error(self.exp_dir, t, x, error, f"iter_{ii:03d}_t0")

    def l2_file(self, data_file: str):
        with h5py.File(data_file, "r") as h5f:
            tx = h5f["tx"][()]
            u = h5f["u"][()]
        for ii in range(1, self.cfg.PICARD.N + 1):
            evaluator = self.get_evaluator(ii)
            err_l2, error, u_sol = evaluator.l2_at_given_solution(tx, u)
            print(f"Iter {ii}: L2 error = {err_l2:.4e}")
            evaluator.plot_error(
                self.exp_dir,
                tx[:, 0],
                tx[:, 1:],
                error,
                f"iter_{ii:03d}_of_given_file",
            )
