"""Score a policy exported by the experiment repo, as CCIL scores its own.

Runs under the CCIL venv (Python 3.8), not the experiment repo's:

    set -x LD_LIBRARY_PATH $HOME/.mujoco/mujoco210/bin /usr/lib/nvidia
    <ccil>/.venv/bin/python evaluate_ccil_policy.py \
      --ccil-directory <ccil> --task hopper \
      --policy <repo>/output/replication/hopper-ccil-seed40.pt --seed 40
"""

import argparse
import os
import pickle
import random
import sys

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

import numpy as np
import torch
import yaml


class _EnvironmentConfig(object):
    """load_env only reads one field off the config, so this fakes it."""

    def __init__(self, env):
        self.env = env


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccil-directory", required=True)
    parser.add_argument("--task", required=True, help="a config name, e.g. hopper")
    parser.add_argument("--policy", required=True, help="a TorchScript .pt file")
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--rewards-out", default=None, help="write the rewards here")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    ccil_directory = os.path.expanduser(arguments.ccil_directory)
    sys.path.insert(0, os.path.join(ccil_directory, "correct_il"))

    from utils import D3Agent, evaluate_on_environment, load_env

    with open(os.path.join(ccil_directory, "config", arguments.task + ".yml")) as file:
        task_config = yaml.safe_load(file)

    noises = task_config["eval"]["noise"] or [0]
    noise = noises[0]

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)

    environment, is_metaworld = load_env(_EnvironmentConfig(task_config["env"]))
    environment.seed(arguments.seed)
    environment.action_space.seed(arguments.seed)
    environment.observation_space.seed(arguments.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = torch.jit.load(os.path.expanduser(arguments.policy))
    policy.to(device)
    agent = D3Agent(policy, device)

    print(
        "task {}, environment {}, seed {}, evaluation noise {}".format(
            arguments.task, task_config["env"], arguments.seed, noise
        )
    )

    rewards, successes = evaluate_on_environment(
        environment,
        agent,
        n_trials=arguments.episodes,
        metaworld=is_metaworld,
        sensor_noise_size=noise,
        actuator_noise_size=noise,
    )

    print(
        "reward {:.2f} +- {:.2f} over {} episodes, success rate {:.2f}".format(
            float(np.mean(rewards)),
            float(np.std(rewards)),
            len(rewards),
            successes / float(arguments.episodes),
        )
    )

    if arguments.rewards_out:
        with open(os.path.expanduser(arguments.rewards_out), "wb") as file:
            pickle.dump(rewards, file)
        print("wrote {}".format(arguments.rewards_out))


if __name__ == "__main__":
    main()
