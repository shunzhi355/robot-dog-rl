import torch

from legged_gym.envs.base.legged_robot import LeggedRobot

from .custom_rewards import compute_custom_reward


class 081801(LeggedRobot):
    def step(self, actions):
        obs, privileged_obs, rewards, dones, infos = super().step(actions)
        infos.setdefault("episode", {})
        infos["episode"]["rew_step_mean"] = torch.mean(rewards.detach())
        infos["episode"]["rew_step_abs_mean"] = torch.mean(torch.abs(rewards.detach()))
        return obs, privileged_obs, rewards, dones, infos

    def check_termination(self):
        contact_threshold = getattr(self.cfg.env, "termination_contact_force_threshold", 20.0)
        grace_steps = int(getattr(self.cfg.env, "termination_grace_time_s", 2.0) / self.dt)
        fail_steps = max(1, int(getattr(self.cfg.env, "fail_to_terminal_time_s", 0.5) / self.dt))
        contact_died = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > contact_threshold,
            dim=1,
        )
        fallen_died = self.projected_gravity[:, 2] > getattr(self.cfg.env, "fallen_projected_gravity_z", -0.35)
        failed = (contact_died | fallen_died) & (self.episode_length_buf > grace_steps)
        if not hasattr(self, "termination_fail_buf"):
            self.termination_fail_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.termination_fail_buf *= failed.float()
        self.termination_fail_buf += failed.float()
        self.reset_buf = self.termination_fail_buf > fail_steps
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, "termination_fail_buf"):
            self.termination_fail_buf[env_ids] = 0

    def _reward_custom_reward(self):
        return compute_custom_reward(self)
