---
id: control-systems
title: Control Systems for Humanoid Robots
sidebar_label: Control Systems
---

# Control Systems for Humanoid Robots

## Introduction to Humanoid Control Systems

Controlling humanoid robots presents unique challenges due to their complex dynamics, multiple degrees of freedom, and need for stable interaction with the environment. Unlike simpler robots, humanoid systems must maintain balance while performing tasks, handle complex contact interactions, and operate safely around humans.

### Control System Architecture

Humanoid control systems typically employ a hierarchical architecture with multiple control layers operating at different timescales:

```
High-Level Planning (seconds) - Task planning, path planning
    ↓
Behavior Control (hundreds of ms) - Gait generation, motion selection
    ↓
Motion Control (tens of ms) - Trajectory tracking, balance control
    ↓
Low-Level Control (ms) - Joint control, motor control
```

### Low-Level Joint Control

#### PID Control for Joints

Proportional-Integral-Derivative (PID) controllers form the foundation of joint control:

```python
class PIDController:
    def __init__(self, kp, ki, kd, dt=0.001):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0
        self.prev_error = 0

    def update(self, setpoint, measurement):
        error = setpoint - measurement

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * self.dt
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.prev_error) / self.dt
        d_term = self.kd * derivative

        self.prev_error = error

        output = p_term + i_term + d_term
        return output

# Example usage for a single joint
joint_controller = PIDController(kp=100, ki=1, kd=10, dt=0.001)
commanded_position = 1.5  # radians
current_position = 1.2    # radians
torque_command = joint_controller.update(commanded_position, current_position)
```

#### Advanced Joint Control

**Impedance Control:**
Implements virtual spring-damper systems for safe interaction:

```python
class ImpedanceController:
    def __init__(self, stiffness, damping, dt=0.001):
        self.stiffness = stiffness  # N*m/rad or N/rad
        self.damping = damping      # N*m*s/rad or N*s/rad
        self.dt = dt

    def update(self, desired_pos, actual_pos, desired_vel=0, actual_vel=0):
        pos_error = desired_pos - actual_pos
        vel_error = desired_vel - actual_vel

        # Impedance control law: F = K(x_d - x) + B(v_d - v)
        force = self.stiffness * pos_error + self.damping * vel_error
        return force

# Example: Compliant joint control
compliant_controller = ImpedanceController(stiffness=50, damping=10)
compliant_torque = compliant_controller.update(
    desired_pos=0.0,
    actual_pos=0.1,
    desired_vel=0.0,
    actual_vel=0.05
)
```

### Balance and Posture Control

#### Zero Moment Point (ZMP) Control

The ZMP is crucial for stable bipedal locomotion:

```python
import numpy as np

class ZMPController:
    def __init__(self, robot_height, gravity=9.81):
        self.h = robot_height  # CoM height
        self.g = gravity       # gravitational acceleration

    def compute_zmp(self, com_pos, com_acc):
        """
        Compute ZMP from CoM position and acceleration
        zmp_x = com_x - h/g * com_acc_x
        zmp_y = com_y - h/g * com_acc_y
        """
        zmp = com_pos - (self.h / self.g) * com_acc
        return zmp

    def track_zmp(self, desired_zmp, current_com_pos, current_com_acc, kp=1.0):
        """
        Control CoM to track desired ZMP
        """
        current_zmp = self.compute_zmp(current_com_pos, current_com_acc)
        zmp_error = desired_zmp - current_zmp

        # Adjust CoM acceleration to reduce ZMP error
        com_acc_correction = kp * zmp_error
        desired_com_acc = current_com_acc + com_acc_correction

        return desired_com_acc
```

#### Linear Inverted Pendulum Model (LIPM)

A simplified model for balance control:

```python
class LIPMController:
    def __init__(self, com_height, gravity=9.81):
        self.h = com_height
        self.g = gravity
        self.omega = np.sqrt(self.g / self.h)

    def compute_next_com_state(self, current_com_pos, current_com_vel,
                              zmp_pos, dt):
        """
        Compute next CoM state using LIPM
        """
        # State transition equations for LIPM
        A = np.array([
            [np.cosh(self.omega * dt), (1/self.omega) * np.sinh(self.omega * dt)],
            [self.omega * np.sinh(self.omega * dt), np.cosh(self.omega * dt)]
        ])

        state = np.array([current_com_pos, current_com_vel])
        zmp_influence = np.array([
            -zmp_pos * (1 - np.cosh(self.omega * dt)),
            -zmp_pos * self.omega * np.sinh(self.omega * dt)
        ])

        next_state = A @ state + zmp_influence
        return next_state[0], next_state[1]  # next_pos, next_vel
```

### Walking Pattern Generation

#### Preview Control for Walking

Preview control uses future reference trajectories to improve tracking:

```python
class PreviewController:
    def __init__(self, dt, preview_steps, com_height, gravity=9.81):
        self.dt = dt
        self.preview_steps = preview_steps
        self.h = com_height
        self.g = gravity
        self.omega = np.sqrt(self.g / self.h)

        # Compute preview gains
        self.compute_preview_gains()

    def compute_preview_gains(self):
        """Compute gains for preview control"""
        # This is a simplified version - full implementation requires
        # solving Riccati equations for optimal control
        self.k_pos = 1.0
        self.k_vel = 1.0
        self.preview_gains = np.zeros(self.preview_steps)

        # Compute preview gains based on system dynamics
        for i in range(self.preview_steps):
            time_ahead = (i + 1) * self.dt
            self.preview_gains[i] = np.exp(-self.omega * time_ahead)

    def compute_control(self, current_com_pos, current_com_vel,
                       reference_trajectory):
        """
        Compute control using current state and preview of future references
        """
        # Current state error feedback
        control = self.k_pos * (reference_trajectory[0] - current_com_pos) + \
                  self.k_vel * (-current_com_vel)

        # Preview compensation
        for i in range(min(len(reference_trajectory)-1, self.preview_steps)):
            control += self.preview_gains[i] * reference_trajectory[i+1]

        return control
```

### Whole-Body Control

#### Operational Space Control

Controls end-effectors while maintaining balance:

```python
class OperationalSpaceController:
    def __init__(self, robot_model):
        self.model = robot_model  # Robot model with kinematics/dynamics

    def compute_torques(self, task_desired_acc, base_desired_acc,
                       joint_pos, joint_vel):
        """
        Compute joint torques using operational space control
        """
        # Get Jacobians and mass matrix from robot model
        J_task = self.model.jacobian_task(joint_pos)  # Task space Jacobian
        J_base = self.model.jacobian_base(joint_pos)  # Base (CoM) Jacobian
        M = self.model.mass_matrix(joint_pos)         # Joint space mass matrix

        # Compute operational space mass matrices
        Lambda_task = np.linalg.inv(J_task @ np.linalg.inv(M) @ J_task.T)
        Lambda_base = np.linalg.inv(J_base @ np.linalg.inv(M) @ J_base.T)

        # Compute bias forces
        h = self.model.bias_forces(joint_pos, joint_vel)

        # Compute task and base torques
        tau_task = J_task.T @ Lambda_task @ (task_desired_acc +
                   self.model.task_bias_acc(joint_pos, joint_vel))
        tau_base = J_base.T @ Lambda_base @ (base_desired_acc +
                   self.model.base_bias_acc(joint_pos, joint_vel))

        # Combine torques with null-space projection
        I = np.eye(M.shape[0])
        N_task = I - np.linalg.inv(M) @ J_task.T @ Lambda_task @ J_task
        tau = tau_task + N_task @ tau_base + h

        return tau
```

### Model-Based Control

#### Computed Torque Control

Linearizes the robot dynamics for easier control:

```python
class ComputedTorqueController:
    def __init__(self, robot_model, kp_diag, kd_diag):
        self.model = robot_model
        self.kp = np.diag(kp_diag)  # Proportional gains
        self.kd = np.diag(kd_diag)  # Derivative gains

    def compute_control(self, q_desired, qd_desired, qdd_desired,
                       q_current, qd_current):
        """
        Compute control torques using computed torque method
        """
        # Get robot dynamics
        M = self.model.mass_matrix(q_current)
        C = self.model.coriolis_matrix(q_current, qd_current)
        G = self.model.gravity_vector(q_current)

        # Compute desired acceleration in joint space
        q_error = q_desired - q_current
        qd_error = qd_desired - qd_current

        # Feedforward + feedback control
        qdd_cmd = qdd_desired + self.kp @ q_error + self.kd @ qd_error

        # Computed torque control law
        tau = M @ qdd_cmd + C @ qd_current + G

        return tau
```

### Adaptive and Learning-Based Control

#### Model Reference Adaptive Control (MRAC)

Adapts to model uncertainties:

```python
class MRACController:
    def __init__(self, reference_model_params, adaptation_rate=0.01):
        self.ref_params = reference_model_params
        self.gamma = adaptation_rate
        self.theta = np.zeros(len(reference_model_params))  # Adaptive parameters

    def update(self, state_error, reference_state, current_state):
        """
        Update adaptive parameters based on tracking error
        """
        # Update adaptive parameters
        self.theta += self.gamma * state_error * reference_state

        # Apply parameter adaptation to control law
        adapted_control = self.compute_adapted_control(current_state, self.theta)

        return adapted_control

    def compute_adapted_control(self, state, params):
        """
        Compute control with adapted parameters
        """
        # Simplified example - actual implementation depends on system
        return -params @ state
```

### Safety and Compliance Control

#### Active Compliance for Human Safety

Ensures safe interaction with humans:

```python
class SafetyController:
    def __init__(self, max_force_threshold=50.0,  # Newtons
                 max_torque_threshold=10.0):      # N*m
        self.max_force = max_force_threshold
        self.max_torque = max_torque_threshold

    def limit_forces(self, commanded_force, sensed_force=None):
        """
        Limit forces to safe levels
        """
        if sensed_force is not None and np.linalg.norm(sensed_force) > self.max_force:
            # Emergency stop or compliant behavior
            return np.zeros_like(commanded_force)

        # Limit commanded forces
        force_norm = np.linalg.norm(commanded_force)
        if force_norm > self.max_force:
            commanded_force = (commanded_force / force_norm) * self.max_force

        return commanded_force

    def emergency_stop(self, sensor_data):
        """
        Check for emergency conditions
        """
        # Check for collision, excessive force, etc.
        if (sensor_data.get('collision_detected', False) or
            sensor_data.get('excessive_force', False)):
            return True
        return False
```

### Control Implementation Considerations

#### Real-Time Performance

For stable control, timing is critical:

```python
import time
import threading

class RealTimeController:
    def __init__(self, control_frequency=1000):  # Hz
        self.dt = 1.0 / control_frequency
        self.next_update = time.time()
        self.is_running = False

    def run_control_loop(self, control_callback, sensor_callback):
        """
        Run control loop with precise timing
        """
        self.is_running = True

        while self.is_running:
            start_time = time.time()

            # Read sensors
            sensor_data = sensor_callback()

            # Execute control
            control_commands = control_callback(sensor_data)

            # Apply commands
            self.apply_commands(control_commands)

            # Wait for next cycle
            elapsed = time.time() - start_time
            sleep_time = self.dt - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"Control loop exceeded timing budget by {abs(sleep_time)*1000:.1f}ms")

    def apply_commands(self, commands):
        """
        Apply control commands to hardware
        """
        # Interface with robot hardware
        pass
```

### Practical Control Strategies

#### Hierarchical Task Control

Manages multiple simultaneous objectives:

```python
class HierarchicalController:
    def __init__(self):
        self.tasks = []  # List of tasks with priorities

    def add_task(self, task_function, priority, weight=1.0):
        """
        Add a control task with priority and weight
        """
        self.tasks.append({
            'function': task_function,
            'priority': priority,
            'weight': weight
        })

        # Sort by priority (lower number = higher priority)
        self.tasks.sort(key=lambda x: x['priority'])

    def compute_control(self, robot_state):
        """
        Compute control using hierarchical task prioritization
        """
        total_control = np.zeros(robot_state.joint_count)

        for task in self.tasks:
            # Compute task-specific control
            task_control = task['function'](robot_state)

            # Apply task weight
            task_control *= task['weight']

            # Add to total control (with priority-based blending)
            total_control += task_control

        return total_control
```

### Learning Objectives

After studying this chapter, you should be able to:

1. Implement PID and advanced joint control algorithms
2. Design balance and posture controllers for humanoid robots
3. Generate stable walking patterns using ZMP and LIPM
4. Apply operational space control for whole-body coordination
5. Implement safety and compliance control strategies
6. Design hierarchical control architectures
7. Consider real-time implementation constraints

### Hands-On Exercise

Implement a simple balance controller for a 2D inverted pendulum model:

1. Create a simulation of an inverted pendulum
2. Implement a ZMP-based balance controller
3. Test with external disturbances
4. Analyze stability and performance
5. Extend to 3D for more realistic humanoid balance

This exercise will help you understand the practical challenges of balance control in humanoid robots.