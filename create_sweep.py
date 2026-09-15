import logging
from dataclasses import dataclass
from typing import Any, Literal

import tyro
import wandb
from rich.pretty import pretty_repr

from dataset_names import build_artifact_name
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

METHOD_FIXED_ARGUMENTS: dict[MethodLabel, list[str]] = {
    "tacil-dp": ["augmentation:tacil"],
    "tacil-bc": ["augmentation:tacil"],
    "ccil": ["augmentation:ccil"],
    "gpi": ["--augmentation=None"],
    "mopo": ["--augmentation=None"],
    "noise-bc": ["augmentation:none"],
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

# CCIL sets this per task, from 1e-4 on most of them to 1.0 on ant, so the
# useful magnitude is not known in advance. One parameter over five decades is
# a grid, not a search: Bayes would spend its first trials rediscovering it.
_NOISE_PARAMETERS: dict[str, Any] = {
    "policy.observation_noise": {"values": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]}
}


METHOD_PARAMETERS: dict[MethodLabel, dict[str, Any]] = {
    "tacil-dp": _TACIL_PARAMETERS,
    "tacil-bc": _TACIL_PARAMETERS,
    "ccil": {
        "augmentation.dynamics.lipschitz_type": {
            "values": ["soft_sampling", "spectral_normalization", "none"]
        },
        "augmentation.dynamics.lipschitz_constraint": {
            "values": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        },
        "augmentation.labels.rejection_quantile": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0,
        },
        "augmentation.labels.type": {"values": ["noisy_action", "backward_euler_fast"]},
        "augmentation.labels.action_noise_std": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 10.0,
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
    dataset_percentage: int
    method: MethodLabel

    seed: int

    use_replay_criterion: bool = False
    sweep_method: SearchLabel = "bayes"

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

    leading_arguments = [
        *METHOD_FIXED_ARGUMENTS[config.method],
        *(
            [
                "--augmentation.data.stopping-rule="
                + ("replay" if config.use_replay_criterion else "account")
            ]
            if config.method.startswith("tacil")
            else []
        ),
    ]
    swept_arguments = "${args_no_boolean_flags}"

    method_arguments = (
        [swept_arguments, *leading_arguments]
        if all(key.startswith("policy.") for key in parameters)
        else [*leading_arguments, swept_arguments]
    )

    return {
        "program": module.replace(".", "/") + ".py",
        "method": config.sweep_method,
        "metric": {"name": RANKING_METRIC, "goal": "maximize"},
        "command": [
            "${env}",
            "${interpreter}",
            "-m",
            module,
            f"--wandb-project={project}",
            "--expert-dataset-artifact="
            + build_artifact_name(dataset_percentage=config.dataset_percentage)
            + ":latest",
            f"--seed={config.seed}",
            *method_arguments,
        ],
        "parameters": parameters,
    }


if __name__ == "__main__":
    main()
