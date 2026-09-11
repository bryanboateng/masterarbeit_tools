"""Score policies exported by the experiment repo, as CCIL scores its own.

Runs under the CCIL venv (Python 3.8), not the experiment repo's:

    set -x LD_LIBRARY_PATH $HOME/.mujoco/mujoco210/bin /usr/lib/nvidia
    <ccil>/.venv/bin/python evaluate_ccil_policy.py \
      --ccil-directory <ccil> --policy-directory <repo>/output/replication

The policy directory is the one replication/run.py writes, laid out as
<task>/<variant>-seed<n>.pt. Every policy without a .pkl next to it is scored;
the rest is skipped. So a second call does no work and only prints the table,
and an interrupted run picks up where it stopped.

A single policy can be scored on its own instead:

    ... --ccil-directory <ccil> --task hopper --policy <file>.pt --seed 40
"""

import argparse
import collections
import glob
import os
import pickle
import random
import re
import sys

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

import numpy as np
import torch
import yaml

EPISODE_COUNT = 100

# The Mujoco and Metaworld table of the CCIL paper, as mean and standard
# deviation over ten seeds.
PAPER_RESULTS = {
    "hopper": {
        "none": (1983.98, 672.66),
        "noise": (1563.56, 1012.02),
        "ccil": (2631.25, 303.86),
    },
    "walker2d": {
        "none": (1922.55, 1410.09),
        "noise": (2893.21, 1076.89),
        "ccil": (3538.48, 573.23),
    },
    "ant": {
        "none": (2965.20, 202.71),
        "noise": (3776.65, 442.13),
        "ccil": (3338.35, 474.17),
    },
    "halfcheetah": {
        "none": (8309.31, 795.30),
        "noise": (8468.98, 738.83),
        "ccil": (8757.38, 379.12),
    },
    "coffee_pull": {
        "none": (3552.59, 233.41),
        "noise": (3072.86, 785.91),
        "ccil": (4168.46, 192.98),
    },
    "button": {
        "none": (3693.02, 104.99),
        "noise": (3663.44, 63.10),
        "ccil": (3775.22, 91.24),
    },
    "coffee_push": {
        "none": (1288.19, 746.37),
        "noise": (2551.11, 857.79),
        "ccil": (2484.19, 976.03),
    },
    "drawer_close": {
        "none": (3247.06, 468.73),
        "noise": (4226.71, 18.90),
        "ccil": (4145.45, 76.23),
    },
}

VARIANT_LABELS = (("none", "BC"), ("noise", "NoiseBC"), ("ccil", "CCIL"))

POLICY_NAME_PATTERN = re.compile(r"^(?P<variant>[a-z]+)-seed(?P<seed>\d+)\.pt$")


class _EnvironmentConfig(object):
    """load_env only reads one field off the config, so this fakes it."""

    def __init__(self, env):
        self.env = env


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccil-directory", required=True)
    parser.add_argument(
        "--policy-directory", default=None, help="a tree of <task>/<variant>-seed<n>.pt"
    )
    parser.add_argument("--policy", default=None, help="a single TorchScript .pt file")
    parser.add_argument("--task", default=None, help="a config name, e.g. hopper")
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=EPISODE_COUNT)
    arguments = parser.parse_args()

    if arguments.policy_directory is None and arguments.policy is None:
        parser.error("pass --policy-directory or --policy")
    if arguments.policy_directory is not None and arguments.policy is not None:
        parser.error("pass --policy-directory or --policy, not both")
    if arguments.policy is not None and arguments.task is None:
        parser.error("--policy needs --task")
    return arguments


def main():
    arguments = parse_arguments()
    ccil_directory = os.path.expanduser(arguments.ccil_directory)
    sys.path.insert(0, os.path.join(ccil_directory, "correct_il"))

    if arguments.policy is not None:
        score_policy(
            ccil_directory=ccil_directory,
            task=arguments.task,
            policy_path=os.path.expanduser(arguments.policy),
            seed=arguments.seed,
            episode_count=arguments.episodes,
        )
        return

    policy_directory = os.path.expanduser(arguments.policy_directory)
    for task, _, seed, policy_path in _find_policies(policy_directory):
        rewards_path = policy_path[: -len(".pt")] + ".pkl"
        if os.path.isfile(rewards_path):
            continue
        rewards = score_policy(
            ccil_directory=ccil_directory,
            task=task,
            policy_path=policy_path,
            seed=seed,
            episode_count=arguments.episodes,
        )
        with open(rewards_path, "wb") as file:
            pickle.dump(rewards, file)

    print_summary(policy_directory=policy_directory)


def score_policy(*, ccil_directory, task, policy_path, seed, episode_count):
    from utils import D3Agent, evaluate_on_environment, load_env

    with open(os.path.join(ccil_directory, "config", task + ".yml")) as file:
        task_config = yaml.safe_load(file)
    noise = (task_config["eval"]["noise"] or [0])[0]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    environment, is_metaworld = load_env(_EnvironmentConfig(task_config["env"]))
    environment.seed(seed)
    environment.action_space.seed(seed)
    environment.observation_space.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = torch.jit.load(policy_path)
    policy.to(device)

    print(
        "task {}, environment {}, seed {}, evaluation noise {}".format(
            task, task_config["env"], seed, noise
        )
    )
    rewards, successes = evaluate_on_environment(
        environment,
        D3Agent(policy, device),
        n_trials=episode_count,
        metaworld=is_metaworld,
        sensor_noise_size=noise,
        actuator_noise_size=noise,
    )
    environment.close()

    print(
        "reward {:.2f} +- {:.2f} over {} episodes, success rate {:.2f}".format(
            float(np.mean(rewards)),
            float(np.std(rewards)),
            len(rewards),
            successes / float(episode_count),
        )
    )
    return rewards


def print_summary(*, policy_directory):
    # Mean over the episodes of a seed first, then mean and standard deviation
    # over the seeds. That is what tabulate_results.py in the CCIL repo does,
    # and what the error bars of the paper are.
    seed_means = collections.defaultdict(list)
    for task, variant, _, policy_path in _find_policies(policy_directory):
        rewards_path = policy_path[: -len(".pt")] + ".pkl"
        if not os.path.isfile(rewards_path):
            continue
        with open(rewards_path, "rb") as file:
            seed_means[(task, variant)].append(float(np.mean(pickle.load(file))))

    if not seed_means:
        print("No scored policies under {}".format(policy_directory))
        return

    print(
        "\n{:<14}{:<10}{:>24}{:>24}{:>7}".format(
            "task", "variant", "ours", "paper", "seeds"
        )
    )
    for task in sorted({task for task, _ in seed_means}):
        for variant, label in VARIANT_LABELS:
            means = seed_means.get((task, variant), [])
            if not means:
                continue
            paper = PAPER_RESULTS.get(task, {}).get(variant)
            print(
                "{:<14}{:<10}{:>24}{:>24}{:>7}".format(
                    task,
                    label,
                    "{:.2f} +- {:.2f}".format(np.mean(means), np.std(means)),
                    "-" if paper is None else "{:.2f} +- {:.2f}".format(*paper),
                    len(means),
                )
            )


def _find_policies(policy_directory):
    """Return (task, variant, seed, path) for every policy in the tree."""
    found = []
    for policy_path in glob.glob(os.path.join(policy_directory, "*", "*.pt")):
        match = POLICY_NAME_PATTERN.match(os.path.basename(policy_path))
        if match is None:
            continue
        found.append(
            (
                os.path.basename(os.path.dirname(policy_path)),
                match.group("variant"),
                int(match.group("seed")),
                policy_path,
            )
        )
    return sorted(found)


if __name__ == "__main__":
    main()
