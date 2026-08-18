from .train081801_env import Train081801
from .train081801_config import Train081801Cfg, Train081801CfgPPO


def register():
    import legged_gym.envs  # noqa: F401
    from legged_gym.utils.task_registry import task_registry

    if "stackforce_train081801" in task_registry.task_classes:
        return
    task_registry.register("stackforce_train081801", Train081801, Train081801Cfg(), Train081801CfgPPO())
