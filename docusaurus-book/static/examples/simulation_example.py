#!/usr/bin/env python3
"""
Complete simulation example for Physical AI & Humanoid Robotics
This example demonstrates a simple bipedal walking simulation using PyBullet
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import math

class SimpleBiped:
    """Simple bipedal robot for simulation"""

    def __init__(self, physics_client, position=[0, 0, 1]):
        self.physics_client = physics_client
        self.position = position

        # Create the robot using primitive shapes
        self.create_robot()

        # Walking parameters
        self.step_frequency = 1.0  # steps per second
        self.step_length = 0.2     # meters
        self.step_height = 0.1     # meters

        # Initialize walking state
        self.walk_phase = 0.0
        self.left_support = True

    def create_robot(self):
        """Create a simple bipedal robot using primitive shapes"""

        # Create collision and visual shapes for main body
        body_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.15, 0.1, 0.2])
        body_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.15, 0.1, 0.2],
                                         rgbaColor=[0.8, 0.2, 0.2, 1])

        # Create the main body (torso)
        self.body_id = p.createMultiBody(
            baseMass=10,
            baseCollisionShapeIndex=body_collision,
            baseVisualShapeIndex=body_visual,
            basePosition=self.position
        )

        # Create head
        head_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
        head_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[0.2, 0.2, 0.8, 1])

        p.createMultiBody(
            baseMass=2,
            baseCollisionShapeIndex=head_collision,
            baseVisualShapeIndex=head_visual,
            basePosition=[self.position[0], self.position[1], self.position[2] + 0.3]
        )

        # Create legs using multiple links
        self.create_leg("left")
        self.create_leg("right")

        # Store joint IDs for control
        self.joint_ids = []
        for i in range(p.getNumJoints(self.body_id)):
            joint_info = p.getJointInfo(self.body_id, i)
            if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.joint_ids.append(i)

    def create_leg(self, side):
        """Create a leg for the robot"""
        # Hip joint (to torso)
        hip_pos = [self.position[0] + (-0.1 if side == "left" else 0.1),
                   self.position[1],
                   self.position[2] - 0.1]

        # Thigh link
        thigh_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.15])
        thigh_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.15],
                                          rgbaColor=[0.2, 0.8, 0.2, 1])

        thigh_id = p.createMultiBody(
            baseMass=3,
            baseCollisionShapeIndex=thigh_collision,
            baseVisualShapeIndex=thigh_visual,
            basePosition=[hip_pos[0], hip_pos[1], hip_pos[2] - 0.15]
        )

        # Knee joint (between thigh and shin)
        knee_pos = [hip_pos[0], hip_pos[1], hip_pos[2] - 0.3]

        # Shin link
        shin_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.15])
        shin_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.15],
                                         rgbaColor=[0.8, 0.8, 0.2, 1])

        shin_id = p.createMultiBody(
            baseMass=2,
            baseCollisionShapeIndex=shin_collision,
            baseVisualShapeIndex=shin_visual,
            basePosition=[knee_pos[0], knee_pos[1], knee_pos[2] - 0.15]
        )

        # Foot link
        foot_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.08, 0.04, 0.02])
        foot_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.08, 0.04, 0.02],
                                         rgbaColor=[0.4, 0.4, 0.4, 1])

        foot_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=foot_collision,
            baseVisualShapeIndex=foot_visual,
            basePosition=[knee_pos[0], knee_pos[1], knee_pos[2] - 0.3]
        )

        # Connect torso to thigh (hip joint)
        p.createConstraint(
            parentBodyUniqueId=self.body_id,
            parentLinkIndex=-1,
            childBodyUniqueId=thigh_id,
            childLinkIndex=-1,
            jointType=p.JOINT_REVOLUTE,
            jointAxis=[1, 0, 0],  # Rotate around x-axis (hip pitch)
            parentFramePosition=[-0.1 if side == "left" else 0.1, 0, -0.1],
            childFramePosition=[0, 0, 0.15]
        )

        # Connect thigh to shin (knee joint)
        p.createConstraint(
            parentBodyUniqueId=thigh_id,
            parentLinkIndex=-1,
            childBodyUniqueId=shin_id,
            childLinkIndex=-1,
            jointType=p.JOINT_REVOLUTE,
            jointAxis=[1, 0, 0],  # Rotate around x-axis (knee)
            parentFramePosition=[0, 0, -0.15],
            childFramePosition=[0, 0, 0.15]
        )

        # Connect shin to foot (ankle joint)
        p.createConstraint(
            parentBodyUniqueId=shin_id,
            parentLinkIndex=-1,
            childBodyUniqueId=foot_id,
            childLinkIndex=-1,
            jointType=p.JOINT_REVOLUTE,
            jointAxis=[1, 0, 0],  # Rotate around x-axis (ankle)
            parentFramePosition=[0, 0, -0.15],
            childFramePosition=[0, 0, 0.02]
        )

    def update_walking(self, time_step):
        """Update walking motion based on time"""
        self.walk_phase += time_step * self.step_frequency * 2 * math.pi

        # Determine which leg is swing leg vs support leg
        left_swing = math.sin(self.walk_phase) > 0
        right_swing = not left_swing

        # Calculate joint angles for walking
        # This is a simplified walking pattern
        hip_angle = 0.2 * math.sin(self.walk_phase) if left_swing else -0.2 * math.sin(self.walk_phase)
        knee_angle = 0.3 * math.sin(self.walk_phase * 2) if left_swing else -0.3 * math.sin(self.walk_phase * 2)
        ankle_angle = -hip_angle  # Keep foot level

        # Apply joint control (in a real implementation, you would control specific joints)
        # For this example, we'll just print the intended motion
        print(f"Time: {time_step:.2f}, Left swing: {left_swing}, Hip: {hip_angle:.2f}, Knee: {knee_angle:.2f}")

    def get_position(self):
        """Get the robot's position"""
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        return pos

def main():
    """Main simulation function"""
    # Connect to physics server
    physics_client = p.connect(p.GUI)  # Use p.DIRECT for non-graphical version
    p.setGravity(0, 0, -9.81)

    # Load plane
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf")

    # Create the biped robot
    robot = SimpleBiped(physics_client)

    # Simulation parameters
    dt = 1.0/240.0  # Time step
    current_time = 0

    print("Starting biped walking simulation...")
    print("Press Ctrl+C to stop the simulation")

    try:
        for step in range(2400):  # Run for 10 seconds (2400 steps at 240Hz)
            # Update walking motion
            robot.update_walking(current_time)

            # Step the simulation
            p.stepSimulation()

            # Update time
            current_time += dt

            # Print position occasionally
            if step % 240 == 0:  # Every second
                pos = robot.get_position()
                print(f"Robot position at t={current_time:.1f}s: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

            # Small delay to control real-time speed
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user")

    # Clean up
    p.disconnect(physics_client)
    print("Simulation ended")

if __name__ == "__main__":
    main()