from pathlib import Path

import typer

from picard import load_cfg, ExperimentEvaluator, PicardRunner

# the picard.__init__ does not import the picard.main, so it is ok to import directly form picard here.

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    }
)
def train(configfile: str, ctx: typer.Context):
    assert Path(configfile).exists(), f"config file {configfile} does not exist"
    cfg = load_cfg(configfile, [arg.lstrip("-") for arg in ctx.args])
    cfg.freeze()
    runner = PicardRunner(cfg)
    runner.run()


@app.command()
def evaluate(exp_dir: str, do_l2: bool = False, sol_file: str = None):
    exp_dir = Path(exp_dir)
    evaluator = ExperimentEvaluator(
        exp_dir,
        n_estimate_terminal=1000_000,
        n_estimate_integral=1000_000,
    )
    if not do_l2:
        evaluator.monte_carlo_at_zero()
    else:
        if sol_file is not None:
            evaluator.l2_file(sol_file)
        else:
            n_points = 1000
            evaluator.l2(n_points)
