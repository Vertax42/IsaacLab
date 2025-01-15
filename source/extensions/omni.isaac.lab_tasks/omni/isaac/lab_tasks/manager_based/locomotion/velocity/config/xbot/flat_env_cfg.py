# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass


from .rough_env_cfg import XbotRoughEnvCfg


@configclass
class XbotFlatEnvCfg(XbotRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # actions
        # self.actions.joint_pos.scale = 0.5
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.6


class XbotFlatEnvCfg_PLAY(XbotFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # disable command randomization
        # self.commands.base_velocity.resampling_time_range = (1e9, 1e9)
        # self.commands.base_velocity.rel_standing_envs = 0.0  # 或保持原值
        # self.commands.base_velocity.rel_heading_envs = 0.0  # 如果不想随机 heading
        # self.commands.base_velocity.ranges.lin_vel_x = (-0.0, 0.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        # self.commands.base_velocity.ranges.heading = (-0.0, 0.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
