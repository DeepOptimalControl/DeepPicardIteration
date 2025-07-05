import copy
import os
import pathlib
import shutil
from typing import Dict, Union, List, Tuple

import lightning as pl
import picard.equations as eqs
import picard.solution as sols
import picard.solution_enforce_terminal as sols_enforce_terminal
import tensorboardX
import torch
import wandb
import yaml
from lightning.pytorch.callbacks import (
    RichProgressBar,
    RichModelSummary,
    LearningRateMonitor,
)
from lightning.pytorch.callbacks.progress.rich_progress import (
    CustomBarColumn,
    BatchesProcessedColumn,
    ProcessingSpeedColumn,
)
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from picard.data import PicardDataModule
from picard.solution_jac import (
    PicardSolutionGradientWrapper,
    PicardSolutionGradientHessianWrapper,
)
from picard.utils import EvalCallback, RichTimeColumn
from rich.progress import TextColumn
from yacs.config import CfgNode


class CustomRichProgressBar(RichProgressBar):
    def __init__(self, *args, **kwargs):
        self.description_prefix = kwargs.pop("description_prefix", "")
        self.total_epochs = kwargs.pop("total_epochs", 0)
        if self.total_epochs > 1:
            self.n_total_epoch_digits = len(str(self.total_epochs))
        super().__init__(*args, **kwargs)

    def _get_train_description(self, current_epoch: int) -> str:
        desc = self.description_prefix
        if self.total_epochs > 1:
            desc = f"{desc} | Epoch {current_epoch:0{self.n_total_epoch_digits}d}/{self.total_epochs}"

        return desc

    def get_metrics(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> Dict[str, Union[int, str, float, Dict[str, float]]]:
        metrics = super().get_metrics(trainer, pl_module)
        metrics.pop("v_num", None)
        for k, v in metrics.items():
            if k.endswith("loss"):
                metrics[k] = f"{v:.3e}"
        return metrics

    def configure_columns(self, trainer: "pl.Trainer") -> list:
        return [
            TextColumn("[progress.description]{task.description}"),
            CustomBarColumn(
                complete_style=self.theme.progress_bar,
                finished_style=self.theme.progress_bar_finished,
                pulse_style=self.theme.progress_bar_pulse,
            ),
            BatchesProcessedColumn(style=self.theme.batch_progress),
            RichTimeColumn(style=self.theme.time),
            ProcessingSpeedColumn(style=self.theme.processing_speed),
        ]


class PicardRunner:
    @staticmethod
    def get_model_checkpoint_path(exp_dir: pathlib.Path, i: int):
        return exp_dir / f"model_{i}.ckpt"

    @staticmethod
    def get_config_path(exp_dir: pathlib.Path):
        return exp_dir / "config.yaml"

    @staticmethod
    def get_pl_dir(exp_dir: pathlib.Path, i: int):
        return exp_dir / f"pl_iter_{i}"

    @staticmethod
    def get_equation(cfg: CfgNode):
        eqs_cls = getattr(eqs, cfg.EQUATION.cls)
        return eqs_cls(**cfg.EQUATION.kwargs)

    @staticmethod
    def get_solution_cls(cfg: CfgNode):
        if cfg.NETWORK.cls is None:
            return sols.PicardSolution
        if hasattr(sols, cfg.NETWORK.cls):
            return getattr(sols, cfg.NETWORK.cls)
        if hasattr(sols_enforce_terminal, cfg.NETWORK.cls):
            return getattr(sols_enforce_terminal, cfg.NETWORK.cls)
        raise ValueError(f"Unknown solution class {cfg.NETWORK.cls}")

    def get_solution_plain(self):
        """
        Gradient is not considered.
        :return:
        """
        solution_cls = self.get_solution_cls(self.cfg)
        solution = solution_cls.construct_solution(self)
        return solution

    def get_solution(self):
        if self.supervise_hessian:
            return PicardSolutionGradientHessianWrapper.construct_solution(self)
        elif self.supervise_gradient:
            return PicardSolutionGradientWrapper.construct_solution(self)
        return self.get_solution_plain()

    def __init__(self, cfg: CfgNode):
        self.cfg = cfg
        self.exp_dir = pathlib.Path(cfg.NAME)
        config_file = self.get_config_path(self.exp_dir)
        if self.exp_dir.exists():
            # if self.exp_dir is empty, we are good to go, else raise error
            under_exp_dir = list(self.exp_dir.iterdir())
            if len(under_exp_dir) > 0:
                if len(under_exp_dir) == 1 and under_exp_dir[0].name == config_file.name:
                    config_file.unlink()
                else:
                    if not self.cfg.FORCE:
                        raise FileExistsError(
                            f"Experiment directory {self.exp_dir} already exists."
                            f" Please change the name in the config file to avoid overwriting."
                        )
                    else:
                        print(f"Force overwriting the existing directory {self.exp_dir.absolute()}")
                        shutil.rmtree(self.exp_dir)
                        self.exp_dir.mkdir()
        else:
            self.exp_dir.mkdir(parents=True)

        # dump full cfg
        with open(config_file, "w") as f:
            f.write(cfg.dump())
        print(f"Experiment directory: {self.exp_dir.absolute()}")

        self.equation = self.get_equation(cfg)
        self.supervise_gradient = False
        if self.cfg.TRAIN.SUPERVISE_GRADIENT or self.equation.has_gradient_term:
            print(
                "Supervising gradient...since",
                (
                    "the equation has gradient term."
                    if self.equation.has_gradient_term
                    else "SUPERVISE_GRADIENT is set to True."
                ),
            )
            self.supervise_gradient = True
        self.supervise_hessian = self.cfg.TRAIN.SUPERVISE_HESSIAN

        self.i = 0
        self.N = cfg.PICARD.N
        if cfg.DATA.ONLINE:
            self.total_data = [cfg.DATA.DATA_SIZE] * self.N
        else:
            self.total_data = [cfg.DATA.DATA_SIZE]
            assert cfg.DATA.N_BUFFER == 0  # to use

        if cfg.NETWORK.TYPE == "ValueGradient":
            self.output_dim = self.equation.nx + 1
        elif cfg.NETWORK.TYPE == "OnlyGradient":
            self.output_dim = self.equation.nx
        else:
            self.output_dim = 1

        # self.u_current = (
        #     sols.GxSolution(self.equation)
        #     if hasattr(sols_enforce_terminal, cfg.NETWORK.cls)
        #     else sols.ZeroSolution(self.output_dim)
        # )
        self.u_current = sols.ZeroSolution(self.output_dim)
        self.u_history = [self.u_current]

        self.n_total_iteration_digits = len(str(self.N))
        self.previous_dataset_size_args = None

        # add a writer of tensorboard
        self.writer = tensorboardX.SummaryWriter(log_dir=os.path.join(cfg.LOGGING.TENSORBOARD_DIR, cfg.NAME))

    def get_reporting_callbacks(self) -> Tuple[List[pl.Callback], dict]:
        description = f"Iter {self.i:0{self.n_total_iteration_digits}d}/{self.N:0{self.n_total_iteration_digits}d}"
        pg_bar = CustomRichProgressBar(description_prefix=description, total_epochs=self.cfg.TRAIN.N_EPOCHS)
        model_summary = RichModelSummary(max_depth=2)
        cbs = [pg_bar, model_summary]
        trainer_kwargs = {}
        if self.cfg.TRAIN.OPTIMIZER.SCHEDULER.cls is not None:
            cbs.append(LearningRateMonitor())
        if self.cfg.EVAL.FREQ is not None:
            # predicted_maximal_batch = (
            #     n_buffer_per_worker
            #     * self.cfg.DATA.N_WORKERS
            #     * self.cfg.TRAIN.BATCH_SIZE
            #     * self.cfg.DATA.kwargs.get("n_estimate_integral", 1)
            # )
            cbs.append(
                EvalCallback(self.cfg.EVAL, self.equation, writer=self.writer, iter=self.i, output_dim=self.output_dim)
            )
            trainer_kwargs.update(
                {
                    "val_check_interval": self.cfg.EVAL.FREQ,
                    "check_val_every_n_epoch": 1,
                }
            )
        return cbs, trainer_kwargs

    def get_data_module(self) -> PicardDataModule:
        module_kws = dict(
            equation=self.equation,
            solution=self.u_current,
            N=self.N,
            i=self.i,
            data_cfg=self.cfg.DATA,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            exp_dir=self.exp_dir,
            do_multi_epochs=self.cfg.TRAIN.N_EPOCHS > 1,
            generate_gradients=self.supervise_gradient,
            generate_hessians=self.supervise_hessian,
        )
        if self.i > self.cfg.DATA.MEMORY.REUSE:
            module_kws["dataset_size_info_args"] = self.previous_dataset_size_args
        if self.cfg.PICARD.FORMULA == "TwoLayer" and len(self.u_history) >= 2:
            data_module = PicardDataModule(solution_m2=self.u_history[-2], **module_kws)
        else:
            data_module = PicardDataModule(**module_kws)
        return data_module

    def run_one(self):
        self.i += 1
        this_iter_dir = self.get_pl_dir(self.exp_dir, self.i)
        this_iter_dir.mkdir(parents=True)
        u = self.get_solution()

        if (self.cfg.NETWORK.PRETRAIN_PATH is not None) and (self.i == 1):
            u.load_state_dict(torch.load(self.cfg.NETWORK.PRETRAIN_PATH)["state_dict"])
            print(f"Using the pretrained network {self.cfg.NETWORK.PRETRAIN_PATH}")  # NOTE: 好像初始还是用0 solution搞的
            self.u_current = copy.deepcopy(u)

        if self.cfg.NETWORK.RELOAD and self.i > 1:
            print("Reloading the network...")
            u.load_state_dict(torch.load(self.get_model_checkpoint_path(self.exp_dir, self.i - 1))["state_dict"])

        if self.cfg.METHOD.cls == "PINN":
            solver = sols.PINNSolution(network=u, equation=self.equation, cfg=self.cfg)
            solver.train()
            return True
        elif self.cfg.METHOD.cls == "Diffusion":
            solver = sols.DiffusionSolution(network=u, equation=self.equation, cfg=self.cfg)
            solver.train()
            return True
        elif self.cfg.METHOD.cls == "FullyNonlinearSolver":
            solver = sols.FullyNonlinearSolution(equation=self.equation, cfg=self.cfg)
            solver.train()
            return True
        if self.cfg.LOGGING.LOGGER == "wandb":
            logger = WandbLogger(
                name=f"{self.cfg.NAME}_picard_iter_{self.i}",
                save_dir=this_iter_dir,
                group=f"{self.cfg.NAME}",
                **self.cfg.LOGGING.kwargs,
            )
        elif self.cfg.LOGGING.LOGGER == "tensorboard":
            logger = TensorBoardLogger(save_dir=this_iter_dir)
        else:
            raise ValueError(f"LOGGER configuration not recognized: '{self.cfg.LOGGING.LOGGER}'")
        logger.log_hyperparams(yaml.safe_load(self.cfg.dump()))

        data_module = self.get_data_module()
        cbs, trainer_kwargs = self.get_reporting_callbacks()
        trainer = pl.Trainer(
            max_epochs=self.cfg.TRAIN.N_EPOCHS,
            default_root_dir=this_iter_dir,
            logger=logger,
            callbacks=cbs,
            **trainer_kwargs,
        )
        # inside data_module, it has already been moved to the correct device.
        trainer.fit(u, datamodule=data_module)
        self.previous_dataset_size_args = data_module.dataset_size_info_args
        if trainer.interrupted:
            # The trainer will catch KeyboardInterrupt for graceful exit.
            # In a custom callback, one can override its on_keyboard_interrupt method to do some clean-up.
            wandb.finish(exit_code=1)
            return False
        trainer.save_checkpoint(self.get_model_checkpoint_path(self.exp_dir, self.i))
        self.u_current = u
        logger.finalize("success")
        wandb.finish()
        return True

    def run(self):
        for _ in range(self.N):
            if not self.run_one():
                # in case of exception
                print("Interrupted...Other processes is terminating...  ")
                break
            self.u_history.append(self.u_current)
