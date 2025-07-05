# Introduction
This is a repo for [Deep Picard Iteration for High-Dimensional Nonlinear PDEs](https://arxiv.org/abs/2409.08526).

# Environment

Ensure you are in the `python` folder, then run:
```
conda env create -f environment.yaml
pip install -e .
```

# Scripts

We provide several example scripts for running the experiments presented in the paper. The scripts are located in the `scripts` folder. Each YAML file includes comments explaining the configuration for each method, and we offer multiple examples for each approach.

## In `scripts/burgers`

To run experiments for the Burgers equation, execute:
```
# DPI
picard train base_100d_T1.0_w0.0_0.yaml
picard train base_100d_T1.0_w1.0_0.yaml

# PINN-HTE
picard train pinn_100d_T1.0_v16_beta10.0.yaml

# D-DBSDE
picard train diffusion_100d_T1.0_beta10.0.yaml
```

## In `scripts/hjb`

For HJB, run:
```
# DPI
picard train base_100d_T1.0_w0.1_0.yaml

# PINN-HTE
picard train pinn_100d_T1.0_v16.yaml

# D-DBSDE
picard train diffusion_100d_T1.0.yaml
```

## In `scripts/fully_nonlinear/case_1`

To reproduce results for fully nonlinear problems, run:
```
# DPI
picard train base_100d_T1.0_w0.0_nov_0.yaml

# PINN
picard train pinn_100d_T1.0_v16_beta10.0.yaml

# DBDP
picard train fn_100d_T1.0.yaml
```