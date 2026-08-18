import torch


def compute_custom_reward(env):
    """Return one custom reward term per environment.

    Edit this function after export to add your own task-specific reward logic.
    The returned tensor should have shape [num_envs].

    To activate it, also set:
        rewards.scales.custom_reward = <non_zero_value>
    in the generated *_config.py file.
    """

    # Example:
    # root_height = env.root_states[:, 2]
    # return torch.clamp(root_height - 0.25, min=0.0)

    return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
