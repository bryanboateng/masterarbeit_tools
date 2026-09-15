import logging
from dataclasses import dataclass
from typing import Any, Literal

import tyro
import wandb
from rich.pretty import pretty_repr

from evaluation_labels import RANKING_METRIC

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(name=__name__)

SearchLabel = Literal["bayes", "random", "grid"]

MethodLabel = Literal[
    "tacil-dp", "tacil-bc", "ccil", "gpi", "mopo", "noise-bc", "noise-dp"
]

METHOD_MODULES: dict[MethodLabel, str] = {
    "tacil-dp": "scripts.policies.dp.run",
    "tacil-bc": "scripts.policies.bc.run",
    "ccil": "scripts.policies.bc.run",
    "gpi": "scripts.policies.gpi.run",
    "mopo": "scripts.policies.mopo.run",
    "noise-bc": "scripts.policies.bc.run",
    "noise-dp": "scripts.policies.dp.run",
}

# tyro only reads a flag between its own subcommand and the next one, and a
# sweep agent puts every swept flag in one place. So each method says what has
# to stand before that place and what after it.
METHOD_LEADING_ARGUMENTS: dict[MethodLabel, list[str]] = {
    "tacil-dp": ["augmentation:tacil"],
    "tacil-bc": ["augmentation:tacil"],
    "ccil": [
        "augmentation:ccil",
        # The follow-up paper commits to spectral normalisation and names the
        # two parameters below it as the ones to tune.
        "--augmentation.dynamics.lipschitz-type=spectral_normalization",
        # A quantile carries over between datasets of different size.
        "--augmentation.labels.rejection.rule=quantile",
    ],
    "gpi": [],
    "mopo": [],
    "noise-bc": [],
    "noise-dp": [],
}

METHOD_TRAILING_ARGUMENTS: dict[MethodLabel, list[str]] = {
    "tacil-dp": [],
    "tacil-bc": ["policy.validation:validation-config"],
    "ccil": [
        "policy.validation:validation-config",
        "augmentation.labels.generation:backward-euler",
    ],
    "gpi": ["--augmentation=None"],
    "mopo": ["--augmentation=None"],
    "noise-bc": ["augmentation:none", "policy.validation:validation-config"],
    "noise-dp": ["augmentation:none"],
}

_TACIL_PARAMETERS: dict[str, Any] = {
    "augmentation.dynamics.window_length": {
        "distribution": "int_uniform",
        "min": 1,
        "max": 15,
    },
    "augmentation.data.action_noise": {
        "distribution": "log_uniform_values",
        "min": 0.001,
        "max": 10,
    },
    "augmentation.data.error_limit_in_steps": {
        "distribution": "log_uniform_values",
        "min": 0.05,
        "max": 120.0,
    },
    "augmentation.dynamics.lipschitz_limit": {
        "distribution": "log_uniform_values",
        "min": 1.0,
        "max": 120.0,
    },
}

# The bounds are the two magnitudes CCIL itself uses: 1e-4 on most tasks and
# 1.0 on ant. Behavior cloning normalises observations to standard deviations
# and the diffusion policy to a range, so one grid would mean two different
# things; a distribution covers both without picking stand-in values.
_NOISE_PARAMETERS: dict[str, Any] = {
    "policy.observation_noise": {
        "distribution": "log_uniform_values",
        "min": 1e-4,
        "max": 1.0,
    }
}


METHOD_SEARCHES: dict[MethodLabel, SearchLabel] = {
    "tacil-dp": "bayes",
    "tacil-bc": "bayes",
    "ccil": "bayes",
    "gpi": "bayes",
    "mopo": "grid",
    "noise-bc": "random",
    "noise-dp": "random",
}


METHOD_PARAMETERS: dict[MethodLabel, dict[str, Any]] = {
    "tacil-dp": _TACIL_PARAMETERS,
    "tacil-bc": _TACIL_PARAMETERS,
    # The two the follow-up paper names as the ones to tune. The Lipschitz
    # type and the label generator are fixed, and a sweep could not reach them
    # anyway: they are subcommands, not values.
    "ccil": {
        "augmentation.dynamics.lipschitz_constraint": {
            "distribution": "log_uniform_values",
            "min": 0.5,
            "max": 5.0,
        },
        "augmentation.labels.rejection.limit": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0,
        },
    },
    "gpi": {
        "policy.progression_weight": {
            "distribution": "log_uniform_values",
            "min": 0.1,
            "max": 10.0,
        },
        "policy.attraction_weight": {
            "distribution": "log_uniform_values",
            "min": 0.1,
            "max": 10.0,
        },
        "policy.steps_per_search": {"distribution": "int_uniform", "min": 2, "max": 32},
    },
    # The grid the MOPO authors themselves searched, with the penalties the
    # reference implementation settled on.
    "mopo": {
        "policy.rollout.length": {"values": [1, 5]},
        "policy.rollout.penalty_coefficient": {"values": [0.5, 2.5, 5.0]},
    },
    "noise-bc": _NOISE_PARAMETERS,
    "noise-dp": _NOISE_PARAMETERS,
}


@dataclass(frozen=True)
class Config:
    project_prefix: str
    dataset_artifact: str
    method: MethodLabel

    seed: int

    use_replay_criterion: bool = False

    # None picks the search that suits the method's parameters. A grid over a
    # distribution never finishes, and Bayes over a handful of values wastes
    # its model on them.
    sweep_method: SearchLabel | None = None

    def __post_init__(self) -> None:
        if self.use_replay_criterion and not self.method.startswith("tacil"):
            raise ValueError(
                f"A trajectory error rule only applies to the TACIL methods, which "
                f"are the ones that walk a trajectory; the method is {self.method}."
            )


def main() -> None:
    config = tyro.cli(f=Config, config=[tyro.conf.UsePythonSyntaxForLiteralCollections])
    logger.info(pretty_repr(config))

    # The data size is deliberately not in the name: one project per method
    # holds one sweep per size, which is what makes them comparable. Every run
    # carries its size in expert_dataset_artifact.
    project = f"{config.project_prefix}-{config.method}"
    if config.use_replay_criterion:
        project += "-replay"

    sweep_configuration = _build_sweep_configuration(config=config, project=project)
    logger.info(pretty_repr(sweep_configuration))

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)

    logger.info("Sweep ID: %s", sweep_id)


def _build_sweep_configuration(*, config: Config, project: str) -> dict[str, Any]:
    module = METHOD_MODULES[config.method]
    parameters = METHOD_PARAMETERS[config.method]

    method_arguments = [
        *METHOD_LEADING_ARGUMENTS[config.method],
        *(
            [
                "--augmentation.data.stopping-rule="
                + ("replay" if config.use_replay_criterion else "account")
            ]
            if config.method.startswith("tacil")
            else []
        ),
        "${args_no_boolean_flags}",
        *METHOD_TRAILING_ARGUMENTS[config.method],
    ]

    return {
        "program": module.replace(".", "/") + ".py",
        "method": config.sweep_method or METHOD_SEARCHES[config.method],
        "metric": {"name": RANKING_METRIC, "goal": "maximize"},
        "command": [
            "${env}",
            "${interpreter}",
            "-m",
            module,
            f"--wandb-project={project}",
            f"--expert-dataset-artifact={config.dataset_artifact}",
            f"--seed={config.seed}",
            *method_arguments,
        ],
        "parameters": parameters,
    }


if __name__ == "__main__":
    main()
