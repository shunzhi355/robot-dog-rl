import argparse

import isaacgym  # noqa: F401

parser = argparse.ArgumentParser(description="List StackForce Legged Gym environments.")
parser.add_argument("--keyword", type=str, default=None, help="Keyword to filter tasks.")
args_cli = parser.parse_args()

import legged_gym.envs  # noqa: F401
from stackforce_simready_train081801_gym import register_tasks
from legged_gym.utils.task_registry import task_registry


def main():
    register_tasks()
    task_names = sorted(task_registry.task_classes.keys())
    for name in task_names:
        if args_cli.keyword is None or args_cli.keyword in name:
            print(name)


if __name__ == "__main__":
    main()
