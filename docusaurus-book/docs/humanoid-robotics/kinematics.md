---
id: kinematics
title: Kinematics in Humanoid Robotics
sidebar_label: Kinematics
---

# Kinematics in Humanoid Robotics

## Introduction to Robot Kinematics

Kinematics is the study of motion without considering the forces that cause it. In humanoid robotics, kinematics provides the mathematical foundation for describing and controlling the robot's movements. Understanding kinematics is crucial for tasks such as walking, reaching, manipulation, and maintaining balance.

### Key Concepts in Robot Kinematics

#### Degrees of Freedom (DOF)
The number of independent parameters that define the configuration of a mechanical system. For humanoid robots:
- **Spatial Motion**: 6 DOF (3 translational, 3 rotational) for position and orientation
- **Planar Motion**: 3 DOF (2 translational, 1 rotational) for 2D movement
- **Joint DOF**: Each joint contributes 1-3 DOF depending on joint type

#### Joint Types
- **Revolute Joint**: 1 DOF rotation around an axis
- **Prismatic Joint**: 1 DOF linear motion along an axis
- **Spherical Joint**: 3 DOF rotation (like human ball-and-socket joints)
- **Cylindrical Joint**: 2 DOF (rotation + translation along same axis)

### Forward Kinematics

Forward kinematics calculates the end-effector position and orientation given joint angles. This is essential for understanding where robot limbs are positioned in space.

#### Mathematical Representation

The pose (position and orientation) of each link in a robot chain is represented using homogeneous transformation matrices:

```
T = [R  p]
    [0  1]
```

Where R is a 3×3 rotation matrix and p is a 3×1 position vector.

For a chain of n joints:
```
T_total = T_0_1 * T_1_2 * ... * T_(n-1)_n
```

#### Denavit-Hartenberg (DH) Convention

The DH convention provides a systematic method for defining coordinate frames on robot links:

1. **z_i axis**: Along the axis of actuation of joint i+1
2. **x_i axis**: Along the common normal from z_i to z_(i+1)
3. **y_i axis**: Completes the right-handed coordinate system

DH parameters:
- **a_i**: Distance along x_i from z_i to z_(i+1)
- **α_i**: Angle about x_i from z_i to z_(i+1)
- **d_i**: Distance along z_i from x_(i-1) to x_i
- **θ_i**: Angle about z_i from x_(i-1) to x_i

#### Example: 2-DOF Planar Arm

Consider a simple 2-DOF planar arm with link lengths L1 and L2:

```python
import numpy as np

def forward_kinematics_2dof(joint_angles, link_lengths):
    """
    Calculate end-effector position for 2-DOF planar arm
    joint_angles: [theta1, theta2] in radians
    link_lengths: [L1, L2]
    """
    theta1, theta2 = joint_angles
    L1, L2 = link_lengths

    # Calculate end-effector position
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)

    return np.array([x, y])

# Example usage
joint_angles = [np.pi/4, np.pi/6]  # 45° and 30°
link_lengths = [1.0, 0.8]  # 1m and 0.8m
end_pos = forward_kinematics_2dof(joint_angles, link_lengths)
print(f"End-effector position: {end_pos}")
```

### Inverse Kinematics

Inverse kinematics (IK) solves the reverse problem: given a desired end-effector position and orientation, find the joint angles required to achieve it.

#### Analytical Solutions

For simple robot structures, closed-form solutions exist:

**2-DOF Planar Arm Inverse Kinematics:**

Given desired end-effector position (x, y) and link lengths L1, L2:

```python
def inverse_kinematics_2dof(end_pos, link_lengths):
    """
    Calculate joint angles for 2-DOF planar arm
    end_pos: [x, y] desired end-effector position
    link_lengths: [L1, L2]
    """
    x, y = end_pos
    L1, L2 = link_lengths

    # Calculate distance from base to end-effector
    r = np.sqrt(x**2 + y**2)

    # Check if position is reachable
    if r > L1 + L2:
        raise ValueError("Position is outside workspace")
    if r < abs(L1 - L2):
        raise ValueError("Position is inside workspace but unreachable")

    # Calculate joint angles
    cos_theta2 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
    theta2 = np.arccos(cos_theta2)

    k1 = L1 + L2 * np.cos(theta2)
    k2 = L2 * np.sin(theta2)

    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

    return np.array([theta1, theta2])
```

#### Numerical Methods

For complex robots, numerical methods are often used:

**Jacobian-Based Methods:**

The Jacobian matrix relates joint velocities to end-effector velocities:

```
v = J(θ) * θ̇
```

Where:
- v is the end-effector velocity vector
- J(θ) is the Jacobian matrix
- θ̇ is the joint velocity vector

For inverse kinematics:
```
θ̇ = J^(-1) * v
```

However, the Jacobian may be singular (non-invertible), requiring techniques like:
- **Pseudoinverse**: θ̇ = J^T * (J * J^T)^(-1) * v
- **Damped Least Squares**: θ̇ = J^T * (J * J^T + λ²I)^(-1) * v

```python
def jacobian_2dof(joint_angles, link_lengths):
    """
    Calculate Jacobian for 2-DOF planar arm
    """
    theta1, theta2 = joint_angles
    L1, L2 = link_lengths

    # Calculate Jacobian matrix
    J = np.array([
        [-L1*np.sin(theta1) - L2*np.sin(theta1+theta2), -L2*np.sin(theta1+theta2)],
        [L1*np.cos(theta1) + L2*np.cos(theta1+theta2), L2*np.cos(theta1+theta2)]
    ])

    return J

def ik_jacobian_inverse(end_pos_desired, current_angles, link_lengths,
                       step_size=0.01, max_iterations=1000, tolerance=1e-6):
    """
    Solve inverse kinematics using Jacobian inverse method
    """
    current_angles = np.array(current_angles)

    for i in range(max_iterations):
        # Calculate current end-effector position
        current_pos = forward_kinematics_2dof(current_angles, link_lengths)

        # Calculate error
        error = end_pos_desired - current_pos

        if np.linalg.norm(error) < tolerance:
            break

        # Calculate Jacobian
        J = jacobian_2dof(current_angles, link_lengths)

        # Calculate joint angle adjustment
        try:
            # Use pseudoinverse to handle near-singularities
            J_inv = np.linalg.pinv(J)
            angle_adjustment = J_inv @ error * step_size
        except:
            # If pseudoinverse fails, use transpose method
            J_inv = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(2))
            angle_adjustment = J_inv @ error * step_size

        # Update joint angles
        current_angles += angle_adjustment

    return current_angles
```

### Kinematics for Humanoid Systems

#### Bipedal Locomotion Kinematics

Humanoid walking involves complex kinematic patterns:

**Stride Parameters:**
- **Step Length**: Distance between consecutive foot placements
- **Step Width**: Lateral distance between feet
- **Stride Length**: Distance between consecutive placements of the same foot
- **Cadence**: Steps per minute

**Gait Cycle Phases:**
1. **Double Support**: Both feet on ground (typically 20% of cycle)
2. **Single Support**: One foot on ground (typically 80% of cycle)
3. **Swing Phase**: Foot in air moving forward

#### Center of Mass (CoM) Control

For stable bipedal locomotion, the CoM trajectory must be carefully controlled:

**Zero Moment Point (ZMP):**
The ZMP is the point on the ground where the net moment of the ground reaction forces is zero. For stability, the ZMP must remain within the support polygon (area of ground contact).

**Linear Inverted Pendulum Model (LIPM):**
A simplified model for CoM motion:
```
ẍ = g/h * (x - z)
```
Where:
- x is CoM position
- z is ZMP position
- h is CoM height
- g is gravitational acceleration

#### Whole-Body Kinematics

Humanoid robots often have 20+ degrees of freedom, requiring coordinated control:

**Task-Space Control:**
- Multiple end-effectors (hands, feet, head) controlled simultaneously
- Coordination of upper and lower body movements
- Balance maintenance during manipulation

**Redundancy Resolution:**
With more DOF than minimally required for tasks:
- Null-space optimization for secondary objectives
- Joint limit avoidance
- Singularity avoidance
- Energy minimization

### Kinematic Constraints and Limitations

#### Joint Limits
Physical constraints on joint angles affect workspace and motion planning:
- **Hard Limits**: Mechanical stops preventing damage
- **Soft Limits**: Recommended operating ranges for safety/comfort
- **Dynamic Limits**: Speed and acceleration constraints

#### Workspace Analysis
- **Dexterous Workspace**: Positions/orientations reachable with full DOF
- **Reachable Workspace**: Positions reachable with any orientation
- **Orientation Workspace**: Orientations achievable at fixed position

#### Singularities
Configurations where the robot loses one or more DOF:
- **Boundary Singularities**: At workspace limits
- **Interior Singularities**: Within workspace (e.g., fully extended arm)
- **Wrist Singularities**: When multiple joint axes align

### Practical Implementation Considerations

#### Computational Efficiency
For real-time control:
- **Analytical Solutions**: When available, preferred for speed
- **Pre-computed Lookups**: For frequently used poses
- **Approximation Methods**: When exact solutions are too slow

#### Numerical Stability
- **Singular Value Decomposition (SVD)**: For robust matrix inversion
- **Regularization**: Adding small values to prevent division by zero
- **Convergence Criteria**: Proper termination conditions

#### Integration with Dynamics
Kinematic solutions must consider dynamic constraints:
- **Joint Torque Limits**: Achievable motions within actuator capabilities
- **Inertia Effects**: Motion planning considering robot mass distribution
- **Contact Forces**: Maintaining balance during interaction

### Advanced Topics

#### Redundant Manipulation
Using extra DOF for optimization:
- **Null-Space Motion**: Joint motion that doesn't affect end-effector
- **Task Prioritization**: Multiple tasks with different priorities
- **Obstacle Avoidance**: Using redundancy for collision avoidance

#### Kinematic Calibration
Improving kinematic accuracy through measurement:
- **Parameter Identification**: Determining actual DH parameters
- **Vision-Based Calibration**: Using cameras to measure poses
- **Self-Calibration**: Using internal sensors for parameter estimation

### Learning Objectives

After studying this chapter, you should be able to:

1. Calculate forward kinematics for simple robot mechanisms
2. Solve inverse kinematics using analytical and numerical methods
3. Apply kinematic principles to humanoid robot systems
4. Understand the relationship between kinematics and balance control
5. Implement basic kinematic algorithms for robot control
6. Analyze workspace and singularity properties of robot mechanisms

### Hands-On Exercise

Implement a kinematic solver for a simple humanoid arm with 7 DOF:

1. Define the kinematic structure using DH parameters
2. Implement forward kinematics function
3. Implement inverse kinematics using Jacobian-based methods
4. Test with various end-effector positions
5. Analyze workspace and singularity properties

### Related Topics

For deeper exploration of concepts covered in this chapter, see:
- [Fundamentals of Physical AI](../embodied-ai/introduction) - Core principles of embodied AI
- [Sensorimotor Loops in Embodied Systems](../embodied-ai/sensorimotor-loops) - Understanding sensorimotor coupling
- [Control Systems for Humanoid Robots](./control-systems) - Control theory applied to robotic systems
- [Machine Learning for Locomotion](../ai-integration/ml-locomotion) - ML approaches to kinematic control
- [Safety Considerations in Physical AI Systems](../challenges-ethics/safety-considerations) - Safety aspects in kinematic control

## Interactive Kinematics Visualization

Try out the kinematics concepts with our interactive robot arm visualizer:

import RobotKinematicsVisualizer from '@site/src/components/RobotKinematicsVisualizer';

<RobotKinematicsVisualizer
  jointAngles={[0.5, 0.3]}
  showWorkspace={true}
  interactive={true}
/>

## Next Steps

[Next: Control Systems for Humanoid Robots](./control-systems)

This exercise will help you understand the practical challenges of kinematic control in humanoid robots.