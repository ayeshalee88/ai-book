#!/usr/bin/env python3
"""
Kinematics Example for Humanoid Robotics
This script demonstrates forward and inverse kinematics for a simple robotic arm.
"""

import numpy as np
import math

class SimpleKinematics:
    """Simple kinematics implementation for educational purposes"""

    def __init__(self, link_lengths):
        """
        Initialize with link lengths [L1, L2, ...]
        For this example, we'll use a 2-DOF planar arm
        """
        self.link_lengths = link_lengths

    def forward_kinematics(self, joint_angles):
        """
        Calculate end-effector position from joint angles
        joint_angles: [theta1, theta2] in radians
        """
        theta1, theta2 = joint_angles
        L1, L2 = self.link_lengths

        # Calculate end-effector position
        x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
        y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)

        return np.array([x, y])

    def inverse_kinematics(self, target_pos):
        """
        Calculate joint angles from end-effector position
        target_pos: [x, y] desired end-effector position
        """
        x, y = target_pos
        L1, L2 = self.link_lengths

        # Calculate distance from base to target
        r = np.sqrt(x**2 + y**2)

        # Check if position is reachable
        if r > L1 + L2:
            raise ValueError("Position is outside workspace")
        if r < abs(L1 - L2):
            raise ValueError("Position is inside workspace but unreachable")

        # Calculate joint angles
        cos_theta2 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
        theta2 = np.arccos(np.clip(cos_theta2, -1, 1))  # Clip to avoid numerical errors

        k1 = L1 + L2 * np.cos(theta2)
        k2 = L2 * np.sin(theta2)

        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

        return np.array([theta1, theta2])

def main():
    """Example usage of the kinematics functions"""
    # Create a 2-DOF arm with link lengths of 1.0m and 0.8m
    arm = SimpleKinematics([1.0, 0.8])

    print("=== Kinematics Example ===")
    print("Robot: 2-DOF Planar Arm")
    print(f"Link lengths: L1={arm.link_lengths[0]}m, L2={arm.link_lengths[1]}m")
    print()

    # Example 1: Forward kinematics
    print("1. Forward Kinematics Example:")
    joint_angles = [np.pi/4, np.pi/6]  # 45° and 30°
    end_pos = arm.forward_kinematics(joint_angles)
    print(f"   Joint angles: [{joint_angles[0]:.3f}, {joint_angles[1]:.3f}] rad")
    print(f"   End-effector position: [{end_pos[0]:.3f}, {end_pos[1]:.3f}] m")
    print()

    # Example 2: Inverse kinematics
    print("2. Inverse Kinematics Example:")
    target_pos = [1.2, 0.8]
    joint_angles = arm.inverse_kinematics(target_pos)
    print(f"   Target position: [{target_pos[0]}, {target_pos[1]}] m")
    print(f"   Required joint angles: [{joint_angles[0]:.3f}, {joint_angles[1]:.3f}] rad")

    # Verify by running forward kinematics on the result
    verification_pos = arm.forward_kinematics(joint_angles)
    print(f"   Verification (FK on result): [{verification_pos[0]:.3f}, {verification_pos[1]:.3f}] m")
    print(f"   Error: {np.linalg.norm(target_pos - verification_pos):.6f} m")
    print()

    # Example 3: Trajectory generation
    print("3. Simple Trajectory Example:")
    start_pos = arm.forward_kinematics([0, 0])
    end_pos = [1.0, 0.5]

    print(f"   Moving from: [{start_pos[0]:.3f}, {start_pos[1]:.3f}] m")
    print(f"   Moving to:   [{end_pos[0]:.3f}, {end_pos[1]:.3f}] m")

    # Generate intermediate points
    steps = 5
    for i in range(steps + 1):
        t = i / steps  # Interpolation parameter (0 to 1)
        current_pos = start_pos + t * (end_pos - start_pos)

        try:
            angles = arm.inverse_kinematics(current_pos)
            print(f"   Step {i}: pos=[{current_pos[0]:.3f}, {current_pos[1]:.3f}], "
                  f"angles=[{angles[0]:.3f}, {angles[1]:.3f}]")
        except ValueError as e:
            print(f"   Step {i}: pos=[{current_pos[0]:.3f}, {current_pos[1]:.3f}], ERROR: {e}")

if __name__ == "__main__":
    main()