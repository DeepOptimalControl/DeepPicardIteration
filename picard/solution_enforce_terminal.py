from typing import Any

import torch
from picard import equations as eqs
from picard.solution import PicardSolution
from yacs.config import CfgNode


class PicardSolutionEnforceTerminal(PicardSolution):
    def __init__(self, equation: eqs.Equation, network_cfg: CfgNode, train_cfg: CfgNode):
        super().__init__(equation, network_cfg, train_cfg)
        self.register_buffer("T", torch.scalar_tensor(self.equation.T))

        if network_cfg.TYPE == "Value":
            self.forward = self._forward_value
        elif network_cfg.TYPE == "OnlyGradient":
            self.forward = self._forward_with_gradient
        else:
            raise ValueError(f"Unsupported network configuration: {self.network_cfg}")

    def _forward_value(self, tx) -> Any:
        t, x = torch.narrow(tx, dim=-1, start=0, length=1), torch.narrow(tx, dim=-1, start=1, length=self.equation.nx)
        return self.equation.g(x) + (self.T - t) * self.model(tx)

    def _forward_with_gradient(self, tx) -> Any:
        t, x = torch.narrow(tx, dim=-1, start=0, length=1), torch.narrow(tx, dim=-1, start=1, length=self.equation.nx)
        return self.equation.g_x(x) + (self.T - t) * self.model(tx)
