import argparse
import os
import sys

import isaacgym  # noqa: F401
import legged_gym.envs  # noqa: F401
from stackforce_simready_train081801_gym import register_tasks
from legged_gym.utils import Logger, export_policy_as_jit, get_args, task_registry

import numpy as np
import torch

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


def parse_custom_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Direct path to a model checkpoint.")
    parser.add_argument("--num_steps", type=int, default=-1, help="Number of inference steps to simulate.")
    parser.add_argument("--export_policy", action="store_true", default=False, help="Export policy as TorchScript.")
    parser.add_argument("--export_onnx", action="store_true", default=False, help="Export policy.onnx.")
    custom_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return custom_args


def play(args, custom_args):
    register_tasks()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, args.num_envs if args.num_envs is not None else 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.ref_env = 0
        env_cfg.viewer.pos = [2.0, -2.2, 1.25]
        env_cfg.viewer.lookat = [0.0, 0.0, 0.55]
        env_cfg.viewer.rendered_envs_idx = [0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    experiment_name = args.experiment_name if getattr(args, "experiment_name", None) else train_cfg.runner.experiment_name
    log_root = os.path.abspath(os.path.join("logs", experiment_name))
    if custom_args.checkpoint_path:
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None)
        ppo_runner.load(os.path.abspath(custom_args.checkpoint_path), load_optimizer=False)
    else:
        train_cfg.runner.resume = True
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=log_root)

    policy = ppo_runner.get_inference_policy(device=env.device)

    if custom_args.export_policy:
        export_path = os.path.join(log_root, "exported", "policies")
        export_policy_as_jit(ppo_runner.alg.actor_critic, export_path)
        print("Exported policy as jit script to:", export_path)

    if custom_args.export_onnx:
        stackforce_export_policy_as_onnx(policy, obs, os.path.join(log_root, "exported", "policies"))

    logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1 if env.num_actions > 1 else 0
    stop_state_log = min(100, custom_args.num_steps if custom_args.num_steps > 0 else 100)
    stop_rew_log = env.max_episode_length + 1

    if hasattr(env_cfg, "viewer"):
        camera_offset = np.array(env_cfg.viewer.pos, dtype=np.float64)
        camera_target_offset = np.array(env_cfg.viewer.lookat, dtype=np.float64)
    else:
        camera_offset = None
        camera_target_offset = None

    i = 0
    while custom_args.num_steps <= 0 or i < custom_args.num_steps:
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        base_pos = getattr(getattr(env, "simulator", None), "base_pos", None)
        if camera_offset is not None and camera_target_offset is not None and base_pos is not None and hasattr(env, "set_viewer_camera"):
            robot_pos = base_pos[robot_index].detach().cpu().numpy()
            env.set_viewer_camera(robot_pos + camera_offset, robot_pos + camera_target_offset)
        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_x": env.commands[robot_index, 0].item(),
                    "command_y": env.commands[robot_index, 1].item(),
                    "command_yaw": env.commands[robot_index, 2].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "contact_forces_z": env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log and infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes > 0:
                logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()
        i += 1


if __name__ == "__main__":
    custom_args = parse_custom_args()
    args = get_args()
    play(args, custom_args)
