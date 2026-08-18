import os

import isaacgym  # noqa: F401

import legged_gym.envs  # noqa: F401
from stackforce_simready_081801_gym import register_tasks
from legged_gym.utils import get_args, task_registry

import os
import torch


class _StackForceOnnxPolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        actions = self.policy(obs)
        if isinstance(actions, (tuple, list)):
            return actions[0]
        if isinstance(actions, dict):
            if "actions" in actions:
                return actions["actions"]
            if "action" in actions:
                return actions["action"]
            return next(iter(actions.values()))
        return actions


def _stackforce_policy_obs_tensor(obs):
    if isinstance(obs, dict):
        obs = obs["policy"] if "policy" in obs else next(iter(obs.values()))
    elif not isinstance(obs, torch.Tensor) and hasattr(obs, "get"):
        try:
            candidate = obs.get("policy")
        except Exception:
            candidate = None
        if candidate is not None:
            obs = candidate
    if not isinstance(obs, torch.Tensor):
        raise TypeError(f"ONNX export requires a tensor policy observation, got {type(obs)!r}")
    if hasattr(obs, "detach"):
        obs = obs.detach()
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)
    elif obs.shape[0] > 1:
        obs = obs[:1]
    return obs.contiguous()


def stackforce_export_policy_as_onnx(policy, obs, output_dir, file_name="policy.onnx", opset=17):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    sample_obs = _stackforce_policy_obs_tensor(obs)
    module = _StackForceOnnxPolicy(policy).to(sample_obs.device).eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            sample_obs,
            output_path,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
            opset_version=opset,
        )
    print(f"Exported ONNX policy to: {output_path}")
    return output_path


def train(args):
    register_tasks()
    env, _ = task_registry.make_env(name=args.task, args=args)
    _, base_train_cfg = task_registry.get_cfgs(args.task)
    experiment_name = args.experiment_name if getattr(args, "experiment_name", None) else base_train_cfg.runner.experiment_name
    log_root = os.path.abspath(os.path.join("logs", experiment_name))
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, log_root=log_root)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    if ppo_runner.log_dir:
        final_path = os.path.join(ppo_runner.log_dir, "model_final.pt")
        ppo_runner.save(final_path)
        print(f"Saved final checkpoint to: {final_path}")
        try:
            stackforce_export_policy_as_onnx(
                ppo_runner.get_inference_policy(device=env.device),
                env.get_observations(),
                os.path.join(ppo_runner.log_dir, "exported", "policies"),
            )
        except Exception as exc:
            print(f"ONNX export skipped: {exc}")


if __name__ == "__main__":
    args = get_args()
    train(args)
