import logging
from dataclasses import dataclass

import tyro
import wandb

from create_sweep import (
    METHOD_LEADING_ARGUMENTS,
    METHOD_MODULES,
    METHOD_PARAMETERS,
    METHOD_TRAILING_ARGUMENTS,
    MethodLabel,
)
from evaluation_labels import RANKING_METRIC

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(name=__name__)


@dataclass(frozen=True)
class Config:
    dataset_artifact: str
    method: MethodLabel

    sweep_id: str

    # Both name a project rather than defaulting to one: a stale default would
    # read an old sweep and print its winners without saying so.
    project_prefix: str
    final_project: str

    top_k: int = 3
    seeds: tuple[int, ...] = (0, 1, 2)


def main() -> None:
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])

    sweep_project = f"{config.project_prefix}-{config.method}"
    sweep = wandb.Api().sweep(f"{sweep_project}/{config.sweep_id}")

    ranked = sorted(
        (run for run in sweep.runs if RANKING_METRIC in run.summary),
        key=lambda run: run.summary[RANKING_METRIC],
        reverse=True,
    )
    if len(ranked) < config.top_k:
        logger.warning(
            "Sweep %s has only %d scored runs, fewer than the requested top %d.",
            sweep_project,
            len(ranked),
            config.top_k,
        )

    module = METHOD_MODULES[config.method]

    for run in ranked[: config.top_k]:
        swept_arguments = [
            f"--{key}={run.config[key]}" for key in METHOD_PARAMETERS[config.method]
        ]
        for seed in config.seeds:
            command = [
                "python",
                "-m",
                module,
                f"--wandb-project={config.final_project}",
                f"--expert-dataset-artifact={config.dataset_artifact}",
                f"--seed={seed}",
                # The swept flags sit where the sweep put them, between what
                # has to precede them and what has to follow.
                *METHOD_LEADING_ARGUMENTS[config.method],
                *swept_arguments,
                *METHOD_TRAILING_ARGUMENTS[config.method],
            ]
            print(" ".join(command))


if __name__ == "__main__":
    main()
