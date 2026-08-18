from pathlib import Path

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "robots" / "generated_robot"
URDF_PATH = ASSET_DIR / "urdf" / "generated_robot.urdf"


class Train081801Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 48
        num_actions = 12
        termination_grace_time_s = 2.0
        fail_to_terminal_time_s = 0.5
        termination_contact_force_threshold = 20.0
        fallen_projected_gravity_z = -0.35

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        curriculum = False

    class init_state(LeggedRobotCfg.init_state):
        # The calibrated stance has all four foot collision meshes at z=-0.14 m.
        # Start with 0.02 m ground clearance and let the robot settle onto the plane.
        pos = [0.0, 0.0, 0.16]
        rot = [0.0, 0.0, 0.0, 1.0]
        default_joint_angles = {
            "hip_joint": 0,
            "knee_joint": -0.5765,
            "foot_joint": 0.6092,
            "hip3_joint": 0,
            "knee3_joint": 0.7132,
            "foot3_joint": 0.6817,
            "hip2_joint": 0,
            "knee2_joint": 0.5767,
            "foot2_joint": 0.6110,
            "hip4_joint": 0,
            "knee4_joint": -0.7137,
            "foot4_joint": 0.6799,
        }

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [2.0, -2.2, 1.25]
        lookat = [0.0, 0.0, 0.55]
        rendered_envs_idx = [0]

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {
            "hip_joint": 20,
            "knee_joint": 20,
            "foot_joint": 20,
            "hip3_joint": 20,
            "knee3_joint": 20,
            "foot3_joint": 20,
            "hip2_joint": 20,
            "knee2_joint": 20,
            "foot2_joint": 20,
            "hip4_joint": 20,
            "knee4_joint": 20,
            "foot4_joint": 20,
        }
        damping = {
            "hip_joint": 2.0,
            "knee_joint": 2.0,
            "foot_joint": 2.0,
            "hip3_joint": 2.0,
            "knee3_joint": 2.0,
            "foot3_joint": 2.0,
            "hip2_joint": 2.0,
            "knee2_joint": 2.0,
            "foot2_joint": 2.0,
            "hip4_joint": 2.0,
            "knee4_joint": 2.0,
            "foot4_joint": 2.0,
        }
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = str(URDF_PATH)
        name = "train081801"
        foot_name = "foot"
        penalize_contacts_on = ["base_link"]
        terminate_after_contacts_on = ["base_link"]
        dof_names = ["hip_joint", "knee_joint", "foot_joint", "hip3_joint", "knee3_joint", "foot3_joint", "hip2_joint", "knee2_joint", "foot2_joint", "hip4_joint", "knee4_joint", "foot4_joint"]
        self_collisions = 0
        fix_base_link = False
        collapse_fixed_joints = True
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.0
        num_commands = 3
        resampling_time = 10.0
        heading_command = False

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]

    class domain_rand(LeggedRobotCfg.domain_rand):
        # Keep stage-0 deterministic until the nominal stance is stable.
        randomize_friction = False
        friction_range = [0.6, 1.2]
        randomize_base_mass = False
        push_robots = False

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.14

        class scales(LeggedRobotCfg.rewards.scales):
            termination = 0
            tracking_lin_vel = 1
            tracking_ang_vel = 0.5
            lin_vel_z = -2
            ang_vel_xy = -0.2
            orientation = -2
            torques = -5e-5
            dof_vel = -1e-3
            dof_acc = -1e-6
            base_height = -1
            feet_air_time = 0
            collision = -1
            feet_stumble = 0
            action_rate = -0.08
            stand_still = -0.5
            custom_reward = 0.0

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

    class noise(LeggedRobotCfg.noise):
        # Observation noise is introduced in later robustness stages.
        add_noise = False
        noise_level = 1.0

    class sim(LeggedRobotCfg.sim):
        dt = 1.0 / 60.0
        substeps = 2

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1
            num_position_iterations = 8
            num_velocity_iterations = 1
            contact_offset = 0.02
            rest_offset = 0.0
            max_depenetration_velocity = 5.0


class Train081801CfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1e-3

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "train081801"
        max_iterations = 1500
