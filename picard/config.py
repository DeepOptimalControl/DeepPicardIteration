import logging
import os
from typing import List

import numpy as np
import torch
from yacs.config import CfgNode

_C = CfgNode()

_C.BASE = None
_C.FORCE = False  # whether to force overwrite the existing directory
_C.NAME = "exp"
# equation to solve
_C.EQUATION = CfgNode()
_C.EQUATION.cls = "AllenCahnEquation"
_C.EQUATION.kwargs = CfgNode(new_allowed=True)

_C.METHOD = CfgNode()
_C.METHOD.cls = "Picard"  # "Picard" or "PINN" or "Diffusion"
_C.METHOD.num_v_samples = 16  # -1 means to use full laplacian
_C.METHOD.K = 20
_C.METHOD.dt = 0.005
_C.METHOD.num_sub_iter = 100

# picard iteration
_C.PICARD = CfgNode()
_C.PICARD.N = 1  # number of iterations
_C.PICARD.FORMULA = None  # set `TwoLayer` to use the two-layer formula

# training
_C.TRAIN = CfgNode()
_C.TRAIN.BATCH_SIZE = 2048
_C.TRAIN.N_EPOCHS = 1
_C.TRAIN.SUPERVISE_GRADIENT = None  # whether to supervise the gradient term
_C.TRAIN.SUPERVISE_HESSIAN = None  # whether to supervise the hessian term
_C.TRAIN.NUM_HESS_SAMPLES = -1  # number of hessian elems sampled to supervise, -1 means full hessian
_C.TRAIN.LOSS = CfgNode()
_C.TRAIN.LOSS.beta = 0.0
_C.TRAIN.LOSS.SCALER = CfgNode()
_C.TRAIN.LOSS.SCALER.cls = None
_C.TRAIN.LOSS.SCALER.kwargs = CfgNode(new_allowed=True)
_C.TRAIN.LOSS.FN = CfgNode()
_C.TRAIN.LOSS.FN.cls = None
_C.TRAIN.LOSS.FN.kwargs = CfgNode(new_allowed=True)
_C.TRAIN.LOSS.use_aux_loss = False  # whether to use auxiliary loss for auto-diff u and u_x
_C.TRAIN.LOSS.weight_aux_loss = 0.1
_C.TRAIN.OPTIMIZER = CfgNode()
_C.TRAIN.OPTIMIZER.cls = "Adam"
_C.TRAIN.OPTIMIZER.kwargs = CfgNode(new_allowed=True)
_C.TRAIN.OPTIMIZER.SCHEDULER = CfgNode()
_C.TRAIN.OPTIMIZER.SCHEDULER.cls = None
_C.TRAIN.OPTIMIZER.SCHEDULER.kwargs = CfgNode(new_allowed=True)
_C.TRAIN.OPTIMIZER.SCHEDULER.config = CfgNode(new_allowed=True)

# network structure
_C.NETWORK = CfgNode()
_C.NETWORK.cls = None
_C.NETWORK.TYPE = "Value"  # "Value" or "ValueGradient" or "OnlyGradient"
_C.NETWORK.NEURONS = [10, 10]
_C.NETWORK.ACTIVATIONS = ["Tanh", "Tanh"]
_C.NETWORK.BOUND = None
_C.NETWORK.RELOAD = False
_C.NETWORK.USE_T_EMBEDDING = False
_C.NETWORK.PISGRADNET = False
_C.NETWORK.PRETRAIN_PATH = None
_C.NETWORK.kwargs = CfgNode(new_allowed=True)

_C.DATA = CfgNode()
_C.DATA.kwargs = CfgNode(new_allowed=True)
_C.DATA.SAVE = False
_C.DATA.ONLINE = True
_C.DATA.TRAIN_FILE = ""
# number of workers to do data fetching and transformation;
_C.DATA.N_WORKERS = 1
_C.DATA.DATA_SIZE = 2048 * 5000
_C.DATA.NEW_SAMPLING = False
# None: will automatically estimate the buffer to use; 0: buffer all data at once; >0: number of batches to buffer
_C.DATA.N_BUFFER = None  # will automatically estimate the maximal buffer to use
# Keep for compatibility
_C.DATA.RESERVED_MEMORY = None  # in MB  # when automatically estimating the buffer size, reserve this amount of memory
_C.DATA.MEMORY = CfgNode()
_C.DATA.MEMORY.RESERVED = None  # in MB
_C.DATA.MEMORY.REDUCE_FACTOR = 1.0  # reduce the buffer size by this factor
_C.DATA.MEMORY.REUSE = 9999999  # results after REUSE iter will use the same config as the REUSE-th iter(1,...)
_C.DATA.PREFETCH_FACTOR = None  # prefetch batches
_C.DATA.DEVICE = None  # device to generate data
_C.DATA.FLOAT = "float"  # will also set the dtype of the network
_C.DATA.EXACT = False  # whether to use exact solution as data
_C.DATA.SHUFFLE = None  # whether to shuffle the data; only support if cached in memory
# _C.DATA.CACHE_TO_FILE = False  # whether to cache the data and then load for training
_C.DATA.PRELOAD = False  # whether to preload the data
# number of workers to preload the data (doing the actual data generation job); None: use the same number as N_WORKERS
# currently, PRELOAD_N_WORKERS != N_WORKERS is not supported
_C.DATA.PRELOAD_N_WORKERS = None  # number of workers to preload the data
_C.DATA.HESSIAN_APPROXIMATION = CfgNode()
_C.DATA.HESSIAN_APPROXIMATION.method = None
_C.DATA.HESSIAN_APPROXIMATION.kwargs = CfgNode(new_allowed=True)
_C.DATA.SAMPLE_BOUND = None
_C.DATA.ESTIMATE_TERMINAL = "OU_ByGx"  # "OU_ByGx" or "OU_Simple"
_C.DATA.ESTIMATE_INTEGRAL = "OU_Simple"  # "OU_Simple" or "OU_Joint"
_C.DATA.ESTIMATE_DELTA_T = 0.0

_C.LOGGING = CfgNode()
_C.LOGGING.LOGGER = "wandb"  # tensorboard
_C.LOGGING.kwargs = CfgNode(new_allowed=True)
_C.LOGGING.kwargs.project = "picard"
_C.LOGGING.kwargs.offline = False
_C.LOGGING.TENSORBOARD_DIR = "tensorboard"  # tensorboard directory

_C.EVAL = CfgNode()
_C.EVAL.L2_N_POINTS = 10_000
_C.EVAL.FREQ = None  # integer, evaluate every `freq` steps.
_C.EVAL.BATCH_SIZE = None  # set this if whole data cannot fit into memory
_C.EVAL.TEST_GRAD = False  # whether to test the gradient error
_C.EVAL.TEST_HESSIAN = False  # whether to test the hessian error


def compatibility_check(cfg: CfgNode):
    if cfg.DATA.RESERVED_MEMORY is not None:
        logging.warning("DATA.RESERVED_MEMORY is deprecated. Please use DATA.MEMORY.RESERVED instead.")
        if cfg.DATA.MEMORY.RESERVED is None:
            cfg.DATA.MEMORY.RESERVED = cfg.DATA.RESERVED_MEMORY
        else:
            raise ValueError("Both RESERVED_MEMORY and MEMORY.RESERVED are set.")


def get_default_cfg():
    return _C.clone()


def get_standard_float_dtype(float_type):
    dtype_str_mapping = {
        torch.float32: {"float", "float32", "f32", "single", "32"},
        torch.float64: {"double", "float64", "f64", "64"},
    }
    if isinstance(float_type, int):
        float_type = str(float_type)
    if isinstance(float_type, str):
        float_type = float_type.lower()
        for dtype, dtype_strs in dtype_str_mapping.items():
            if float_type in dtype_strs:
                return dtype
    return float_type


def get_standard_float_type_np(float_type):
    return {
        torch.float32: np.float32,
        torch.float64: np.float64,
    }[get_standard_float_dtype(float_type)]


def get_float_type_bytes(float_type):
    return {
        torch.float32: 4,
        torch.float64: 8,
    }[get_standard_float_dtype(float_type)]


def get_tensor_type(dtype):
    if dtype == torch.float32:
        return torch.FloatTensor
    elif dtype == torch.float64:
        return torch.DoubleTensor
    else:
        raise ValueError(f"Unknown dtype {dtype}")


def set_default_dtype(dtype):
    torch.set_default_dtype(get_standard_float_dtype(dtype))


def override_should_not_contain_base(override: List[str]):
    for item in override:
        if item.startswith("--BASE") or item.startswith("BASE"):
            raise ValueError("override should not contain BASE")


def set_start_method_to_spawn(ignore_context_has_set: bool):
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except RuntimeError as e:
        if not ignore_context_has_set:
            raise RuntimeError(
                "Cannot set multiprocessing context twice. "
                "Please set `ignore_context_has_set` to True if you are fine with this."
            ) from e
        assert mp.get_start_method() == "spawn"


def apply_cfg(cfg: CfgNode, ignore_context_has_set: bool):
    set_default_dtype(cfg.DATA.FLOAT)
    if cfg.DATA.DEVICE != "cpu" and torch.cuda.is_available():
        cfg.DATA.DEVICE = "cuda"
    use_cuda_with_multiprocessing = (cfg.DATA.DEVICE == "cuda") and (cfg.DATA.N_WORKERS > 0)
    if use_cuda_with_multiprocessing:
        set_start_method_to_spawn(ignore_context_has_set)


def _read_cfg_from_file(cfg_file: str):
    """
    This function is meant to get a complete cfg from a file; i.e., missing keys are filled with default values.
    This is needed since using `CfgNode`'s `merge_from_other_cfg` will require the cfg to be complete.
    :param cfg_file:
    :return:
    """
    cfg = get_default_cfg()
    cfg.merge_from_file(cfg_file)
    return cfg


def _read_cfg_from_file_only(cfg_file: str):
    """
    This function only reads key-value pairs from a file; missing values are not filled with default values.
    This is useful when merging a cfg with a file:
     one should not fill the missing values in the file with default,
     otherwise the default values will overwrite the values in the cfg.
    :param cfg_file:
    :return:
    """
    with open(cfg_file, "r") as f:
        cfg = CfgNode.load_cfg(f)
    return cfg


def get_nested_base(cfg: CfgNode):
    """
    Get the all the base cfg, and return then in the order from deep to shallow.
    :param cfg:
    :return:
    """
    all_base = []
    while hasattr(cfg, "BASE") and cfg.BASE is not None:
        base, cfg = cfg.BASE, _read_cfg_from_file_only(cfg.BASE)
        all_base.append((base, cfg))
    return reversed(all_base)


def load_cfg(cfg_file: str, override: List[str] = None, ignore_context_has_set: bool = False):
    top_cfg = _read_cfg_from_file_only(cfg_file)
    all_base = get_nested_base(top_cfg)

    cfg = get_default_cfg()
    all_names = []  # name is somewhat special, we shall join them
    for base, base_cfg in all_base:
        cfg.merge_from_other_cfg(base_cfg)
        if hasattr(base_cfg, "NAME"):
            all_names.append(base_cfg.NAME)
    cfg.merge_from_other_cfg(top_cfg)
    # top_cfg must have a name
    cfg.NAME = "_".join(all_names + [top_cfg.NAME])

    if hasattr(cfg, "BASE"):
        cfg.pop("BASE")

    # value from base can be overwritten by value from cfg_file
    if override is not None:
        override_should_not_contain_base(override)
        cfg.merge_from_list(override)
    apply_cfg(cfg, ignore_context_has_set)
    compatibility_check(cfg)
    cfg.freeze()
    return cfg
