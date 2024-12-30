# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(3) control."""

import numpy as np
import weakref
from collections.abc import Callable
from scipy.spatial.transform import Rotation

import carb
import omni

from ..device_base import DeviceBase


class Se3Keyboard(DeviceBase):
    """A keyboard controller for sending SE(3) commands as delta poses and binary command (open/close).

    This class is designed to provide a keyboard controller for a robotic arm with a gripper.
    It uses the Omniverse keyboard interface to listen to keyboard events and map them to robot's
    task-space commands.

    The command comprises of two parts:

    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Toggle gripper (open/close)    K
        Move along x-axis              W                 S
        Move along y-axis              A                 D
        Move along z-axis              Q                 E
        Rotate along x-axis            Z                 X
        Rotate along y-axis            T                 G
        Rotate along z-axis            C                 V
        ============================== ================= =================

    .. seealso::

        The official documentation for the keyboard interface: `Carb Keyboard Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

    """

    def __init__(self, pos_sensitivity: float = 0.4, rot_sensitivity: float = 0.8):
        """Initialize the keyboard layer.

        Args:
            pos_sensitivity: Magnitude of input position command scaling. Defaults to 0.05.
            rot_sensitivity: Magnitude of scale input rotation commands scaling. Defaults to 0.5.
        """
        # store inputs
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(
                event, *args
            ),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # command buffers
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for SE(3): {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tToggle gripper (open/close): K\n"
        msg += "\tMove arm along x-axis: W/S\n"
        msg += "\tMove arm along y-axis: A/D\n"
        msg += "\tMove arm along z-axis: Q/E\n"
        msg += "\tRotate arm along x-axis: Z/X\n"
        msg += "\tRotate arm along y-axis: T/G\n"
        msg += "\tRotate arm along z-axis: C/V"
        return msg

    """
    Operations
    """

    def reset(self):
        # default flags
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        A list of available keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> tuple[np.ndarray, bool]:
        """Provides the result from keyboard event state.

        Returns:
            A tuple containing the delta pose command and gripper commands.
        """
        # convert to rotation vector
        rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()
        # return the command and gripper state
        return np.concatenate([self._delta_pos, rot_vec]), self._close_gripper

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "L":
                self.reset()
            if event.input.name == "K":
                self._close_gripper = not self._close_gripper
            elif event.input.name in ["W", "S", "A", "D", "Q", "E"]:
                self._delta_pos += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["Z", "X", "T", "G", "C", "V"]:
                self._delta_rot += self._INPUT_KEY_MAPPING[event.input.name]
        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in ["W", "S", "A", "D", "Q", "E"]:
                self._delta_pos -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["Z", "X", "T", "G", "C", "V"]:
                self._delta_rot -= self._INPUT_KEY_MAPPING[event.input.name]
        # additional callbacks
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # toggle: gripper command
            "K": True,
            # x-axis (forward)
            "W": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            # y-axis (left-right)
            "A": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            # z-axis (up-down)
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,
            # roll (around x-axis)
            "Z": np.asarray([1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "X": np.asarray([-1.0, 0.0, 0.0]) * self.rot_sensitivity,
            # pitch (around y-axis)
            "T": np.asarray([0.0, 1.0, 0.0]) * self.rot_sensitivity,
            "G": np.asarray([0.0, -1.0, 0.0]) * self.rot_sensitivity,
            # yaw (around z-axis)
            "C": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            "V": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,
        }


"""Keyboard controller for velocity control (delta lin_vel_x, lin_vel_y, angle_vel_z, heading)."""


class Se3KeyboardVelocity(DeviceBase):
    """A keyboard controller for sending velocity-like commands:
    (delta_lin_vel_x, delta_lin_vel_y, delta_ang_vel_z, delta_heading).

    This class is designed to provide a keyboard controller for a robot that accepts velocity
    commands in the form of:
      - linear velocity in x, y
      - angular velocity (yaw rate) around z
      - heading (orientation around z, e.g. target yaw)

    Key bindings (example):
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Increase lin_vel_x             W                 S
        Increase lin_vel_y             A                 D
        Increase ang_vel_z (yaw rate)  Q                 E
        Increase heading angle (yaw)   T                 G
        ============================== ================= =================

    You can adjust the sensitivity for how quickly the velocity and heading change.
    """

    def __init__(self, lin_sensitivity: float = 0.2, ang_sensitivity: float = 0.2):
        """
        Args:
            lin_sensitivity: Scaling for linear velocity increments.
            ang_sensitivity: Scaling for angular velocity and heading increments.
        """
        # store inputs
        self.lin_sensitivity = lin_sensitivity
        self.ang_sensitivity = ang_sensitivity

        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()

        # note: Use weakref on callbacks to ensure that this object can be deleted properly
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(
                event, *args
            ),
        )

        # set up default key bindings
        self._create_key_bindings()

        # command buffers:
        # we'll accumulate increments in these variables
        self._delta_lin_vel = np.zeros(2)  # (delta vx, delta vy)
        self._delta_ang_vel_heading = np.zeros(
            2
        )  # (delta angular velocity, delta heading)

        # dictionary for user-defined callbacks (e.g., reset env)
        self._additional_callbacks = {}

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string describing this keyboard interface."""
        msg = f"Keyboard Controller for Velocity: {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tPress W/S to increment +/- linear velocity in X.\n"
        msg += "\tPress A/D to increment +/- linear velocity in Y.\n"
        msg += "\tPress Q/E to increment +/- angular velocity (yaw rate) around Z.\n"
        msg += "\tPress T/G to increment +/- heading (target yaw).\n"
        msg += "\tPress R to reset all increments to zero.\n"
        msg += "\nYou can further customize or add callbacks using 'add_callback'.\n"
        return msg

    def reset(self):
        """Resets the stored increments to zero."""
        self._delta_lin_vel = np.zeros(2)
        self._delta_ang_vel_heading = np.zeros(2)

    def add_callback(self, key: str, func: Callable):
        """Bind an additional function to a specific key press.

        Args:
            key: The keyboard key name, e.g. "R".
            func: The function to call upon KEY_PRESS of that key.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> np.ndarray:
        """Provides the current increments in velocity and heading.

        Returns:
            A tuple (delta_lin_vel_x, delta_lin_vel_y, delta_ang_vel_z, delta_heading).
        """
        return np.concatenate([self._delta_lin_vel, self._delta_ang_vel_heading])

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Callback each time a keyboard event triggers."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self.reset()
            elif event.input.name in ["W", "S", "A", "D"]:
                self._delta_lin_vel += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["Q", "E", "T", "G"]:
                self._delta_ang_vel_heading += self._INPUT_KEY_MAPPING[event.input.name]
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in ["W", "S", "A", "D"]:
                self._delta_lin_vel -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["Q", "E", "T", "G"]:
                self._delta_ang_vel_heading -= self._INPUT_KEY_MAPPING[event.input.name]
        # check additional callbacks
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()
        # print(
        #    "Current command: vx: {self._delta_lin_vel[0]}, vy: {self._delta_lin_vel[1]}, wz: {self._delta_ang_vel_heading[0]}, hd: {self._delta_ang_vel_heading[1]}"
        # )
        return True  # no error

    def _create_key_bindings(self):
        """Sets default mapping for velocity control."""
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # x-axis (forward)
            "W": np.asarray([1.0, 0.0]) * self.lin_sensitivity,
            "S": np.asarray([-1.0, 0.0]) * self.lin_sensitivity,
            # y-axis (left-right)
            "A": np.asarray([0.0, 1.0]) * self.lin_sensitivity,
            "D": np.asarray([0.0, -1.0]) * self.lin_sensitivity,
            # angular velocity (yaw rate)
            "Q": np.asarray([0.5, 0.0]) * self.ang_sensitivity,
            "E": np.asarray([-0.5, 0.0]) * self.ang_sensitivity,
            # heading (yaw)
            "T": np.asarray([0.0, 0.3]) * self.ang_sensitivity,
            "G": np.asarray([0.0, -0.3]) * self.ang_sensitivity,
        }
