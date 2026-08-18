from .081801_env import 081801
from .081801_config import 081801Cfg, 081801CfgPPO


def register():
    import legged_gym.envs  # noqa: F401
    from legged_gym.utils.task_registry import task_registry

    if "stackforce_081801" in task_registry.task_classes:
        return
    task_registry.register("stackforce_081801", 081801, 081801Cfg(), 081801CfgPPO())
