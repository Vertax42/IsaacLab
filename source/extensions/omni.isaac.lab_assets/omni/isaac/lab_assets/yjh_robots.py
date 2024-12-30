# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Author: Vertax
# Email: yaphetys@gmail.com
# Date: 2024-12-16
# Description: IsaacLab configuration for yijiahe robots.

"""Configuration for yijiahe robots.

The following configurations are available:

* :obj:`X100_CFG`: yjh X100 humanoid robot

Reference: https://github.com/isaac-sim/IsaacLab
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg


XBot_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/vertax/usd_file/XBot-L/XBot-L.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            # "left_shoulder_pitch_joint": 0.0,
            # "left_shoulder_roll_joint": 0.0,
            # "left_arm_yaw_joint": 0.0,
            # "left_elbow_pitch_joint": 0.9,
            # "left_elbow_yaw_joint": 0.0,
            # 'left_wrist_roll_joint': 0.,
            # 'left_wrist_yaw_joint': 0.,
            # 'left_hand_thumb_bend_joint': 0.,
            # 'left_hand_thumb_rota_joint1': 0.,
            # 'left_hand_thumb_rota_joint2': 0.,
            # 'left_hand_index_joint1': 0.,
            # 'left_hand_index_joint2': 0.,
            # 'left_hand_mid_joint1': 0.,
            # 'left_hand_mid_joint2': 0.,
            # 'left_hand_ring_joint1': 0.,
            # 'left_hand_ring_joint2': 0.,
            # 'left_hand_pinky_joint1': 0.,
            # 'left_hand_pinky_joint2': 0.,
            # "right_shoulder_pitch_joint": 0.0,
            # "right_shoulder_roll_joint": 0.0,
            # "right_arm_yaw_joint": 0.0,
            # "right_elbow_pitch_joint": 0.9,
            # "right_elbow_yaw_joint": 0.0,
            # 'right_wrist_roll_joint': 0.,
            # 'right_wrist_yaw_joint': 0.,
            # 'right_hand_thumb_bend_joint': 0.,
            # 'right_hand_thumb_rota_joint1': 0.,
            # 'right_hand_thumb_rota_joint2': 0.,
            # 'right_hand_mid_joint1': 0.,
            # 'right_hand_mid_joint2': 0.,
            # 'right_hand_ring_joint1': 0.,
            # 'right_hand_ring_joint2': 0.,
            # 'right_hand_pinky_joint1': 0.,
            # 'right_hand_pinky_joint2': 0.,
            # 'right_hand_index_joint1': 0.,
            # 'right_hand_index_joint2': 0.,
            # "waist_yaw_joint": 0.0,
            # 'waist_roll_joint': 0.,
            "left_leg_roll_joint": 0.0,
            "left_leg_yaw_joint": 0.0,
            "left_leg_pitch_joint": -0.0,
            "left_knee_joint": 0.0,
            "left_ankle_pitch_joint": -0.0,
            "left_ankle_roll_joint": -0.0,
            "right_leg_roll_joint": 0.0,
            "right_leg_yaw_joint": 0.0,
            "right_leg_pitch_joint": -0.0,
            "right_knee_joint": 0.0,
            "right_ankle_pitch_joint": -0.0,
            "right_ankle_roll_joint": -0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_leg_roll_joint",
                "left_leg_yaw_joint",
                "left_leg_pitch_joint",
                "left_knee_joint",  # left_leg_joints
                "right_leg_roll_joint",
                "right_leg_yaw_joint",
                "right_leg_pitch_joint",
                "right_knee_joint",  # right_leg_joints
            ],
            effort_limit=250,
            velocity_limit=120,
            stiffness={
                ".*_leg_pitch_joint": 350.0,
                ".*_leg_roll_joint": 200.0,
                ".*_leg_yaw_joint": 200.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_leg_pitch_joint": 40.0,
                ".*_leg_roll_joint": 40.0,
                ".*_leg_yaw_joint": 40.0,
                ".*_knee_joint": 40.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint", ".*_ankle_pitch_joint"],
            effort_limit=20,
            stiffness=20.0,
            damping=4.0,
        ),
    },
)
"""Configuration for the Yijiahe X100 Humanoid robot."""


XBot_All_Joints_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/vertax/usd_file/XBot-L/XBot-L-all-joints.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            # left arm # 7
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_arm_yaw_joint": 0.0,
            "left_elbow_pitch_joint": 0.0,
            "left_elbow_yaw_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            # left hand #
            # 'left_hand_thumb_bend_joint': 0.,
            # 'left_hand_thumb_rota_joint1': 0.,
            # 'left_hand_thumb_rota_joint2': 0.,
            # 'left_hand_index_joint1': 0.,
            # 'left_hand_index_joint2': 0.,
            # 'left_hand_mid_joint1': 0.,
            # 'left_hand_mid_joint2': 0.,
            # 'left_hand_ring_joint1': 0.,
            # 'left_hand_ring_joint2': 0.,
            # 'left_hand_pinky_joint1': 0.,
            # 'left_hand_pinky_joint2': 0.,
            # right arm # 7
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_arm_yaw_joint": 0.0,
            "right_elbow_pitch_joint": 0.0,
            "right_elbow_yaw_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            # right hand #
            # 'right_hand_thumb_bend_joint': 0.,
            # 'right_hand_thumb_rota_joint1': 0.,
            # 'right_hand_thumb_rota_joint2': 0.,
            # 'right_hand_mid_joint1': 0.,
            # 'right_hand_mid_joint2': 0.,
            # 'right_hand_ring_joint1': 0.,
            # 'right_hand_ring_joint2': 0.,
            # 'right_hand_pinky_joint1': 0.,
            # 'right_hand_pinky_joint2': 0.,
            # 'right_hand_index_joint1': 0.,
            # 'right_hand_index_joint2': 0.,
            # waist # 2
            "waist_yaw_joint": 0.0,
            # 'waist_roll_joint': 0.,
            # left leg # 6
            "left_leg_roll_joint": 0.0,
            "left_leg_yaw_joint": 0.0,
            "left_leg_pitch_joint": -0.0,
            "left_knee_joint": 0.0,
            "left_ankle_pitch_joint": -0.0,
            "left_ankle_roll_joint": -0.0,
            # right leg # 6
            "right_leg_roll_joint": 0.0,
            "right_leg_yaw_joint": 0.0,
            "right_leg_pitch_joint": -0.0,
            "right_knee_joint": 0.0,
            "right_ankle_pitch_joint": -0.0,
            "right_ankle_roll_joint": -0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_leg_roll_joint",
                "left_leg_yaw_joint",
                "left_leg_pitch_joint",
                "left_knee_joint",  # left_leg_joints
                "right_leg_roll_joint",
                "right_leg_yaw_joint",
                "right_leg_pitch_joint",
                "right_knee_joint",  # right_leg_joints
                "waist_yaw_joint",  # waist_joints 4+4+1=9 DOF
            ],
            effort_limit=250,
            velocity_limit=120,
            stiffness={
                ".*_leg_pitch_joint": 350.0,
                ".*_leg_roll_joint": 200.0,
                ".*_leg_yaw_joint": 200.0,
                ".*_knee_joint": 350.0,
                "waist_yaw_joint": 200.0,
            },
            damping={
                ".*_leg_pitch_joint": 10.0,
                ".*_leg_roll_joint": 10.0,
                ".*_leg_yaw_joint": 10.0,
                ".*_knee_joint": 10.0,
                "waist_yaw_joint": 40.0,
            },
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint", ".*_ankle_pitch_joint"],
            effort_limit=20,
            stiffness=15.0,
            damping=10.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_arm_yaw_joint",
                ".*_elbow_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_wrist_.*joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=300.0,
            damping=20.0,
            armature=0.01,
        ),
    },
)
"""Configuration for the Yijiahe X100 Humanoid robot."""
