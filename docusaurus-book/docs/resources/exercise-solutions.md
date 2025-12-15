---
id: exercise-solutions
title: Exercise Solutions and Learning Resources
sidebar_label: Exercise Solutions
---

# Exercise Solutions and Learning Resources

This guide provides solutions to exercises throughout the Physical AI & Humanoid Robotics book and additional learning resources to deepen your understanding.

## Chapter 1: Introduction to Embodied AI

### Exercise: Understanding Sensorimotor Loops
**Problem**: Create a simple simulation of a sensorimotor loop where a robot adjusts its movement based on sensor feedback.

**Solution**:
```python
import numpy as np
import matplotlib.pyplot as plt

class SimpleSensorimotorLoop:
    def __init__(self):
        self.position = 0
        self.velocity = 0
        self.target = 10  # Target position
        self.sensor_noise = 0.1
        self.dt = 0.1

    def sense(self):
        """Sense the environment with some noise"""
        true_distance = self.target - self.position
        sensed_distance = true_distance + np.random.normal(0, self.sensor_noise)
        return sensed_distance

    def act(self, sensed_distance):
        """Actuator response based on sensed input"""
        # Simple proportional controller
        k_p = 0.5
        force = k_p * sensed_distance
        return np.clip(force, -2, 2)  # Limit force

    def update_dynamics(self, force):
        """Update robot dynamics"""
        # Simple mass-spring-damper model
        acceleration = force - 0.1 * self.velocity  # Damping
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt

    def run_simulation(self, steps=100):
        """Run the sensorimotor loop simulation"""
        positions = []
        velocities = []
        forces = []

        for step in range(steps):
            sensed_distance = self.sense()
            force = self.act(sensed_distance)
            self.update_dynamics(force)

            positions.append(self.position)
            velocities.append(self.velocity)
            forces.append(force)

        return positions, velocities, forces

# Run simulation
robot = SimpleSensorimotorLoop()
positions, velocities, forces = robot.run_simulation()

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(positions)
ax1.set_title('Robot Position Over Time')
ax1.set_ylabel('Position')

ax2.plot(forces)
ax2.set_title('Applied Force Over Time')
ax2.set_ylabel('Force')
ax2.set_xlabel('Time Step')

plt.tight_layout()
plt.show()
```

## Chapter 2: Kinematics in Humanoid Robotics

### Exercise: 2-DOF Inverse Kinematics Solution
**Problem**: Implement inverse kinematics for a 2-DOF planar arm.

**Solution**:
```python
import numpy as np
import matplotlib.pyplot as plt

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

    # Calculate joint angles using law of cosines
    cos_theta2 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
    theta2 = np.arccos(np.clip(cos_theta2, -1, 1))  # Clip to avoid numerical errors

    k1 = L1 + L2 * np.cos(theta2)
    k2 = L2 * np.sin(theta2)

    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

    return np.array([theta1, theta2])

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

# Test the solution
link_lengths = [1.0, 0.8]  # Link lengths
desired_pos = [1.2, 0.5]   # Desired end-effector position

try:
    joint_angles = inverse_kinematics_2dof(desired_pos, link_lengths)
    print(f"Joint angles: {np.degrees(joint_angles)} degrees")

    # Verify with forward kinematics
    calculated_pos = forward_kinematics_2dof(joint_angles, link_lengths)
    print(f"Desired position: {desired_pos}")
    print(f"Calculated position: {calculated_pos}")
    print(f"Error: {np.linalg.norm(np.array(desired_pos) - calculated_pos):.6f}")

except ValueError as e:
    print(f"Error: {e}")

# Visualization
def plot_robot(joint_angles, link_lengths):
    theta1, theta2 = joint_angles
    L1, L2 = link_lengths

    # Calculate joint positions
    elbow_x = L1 * np.cos(theta1)
    elbow_y = L1 * np.sin(theta1)

    end_x = elbow_x + L2 * np.cos(theta1 + theta2)
    end_y = elbow_y + L2 * np.sin(theta1 + theta2)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.plot([0, elbow_x, end_x], [0, elbow_y, end_y], 'o-', linewidth=3, markersize=10)
    plt.plot(0, 0, 'ro', markersize=15, label='Base')
    plt.plot(elbow_x, elbow_y, 'go', markersize=12, label='Elbow')
    plt.plot(end_x, end_y, 'bo', markersize=12, label='End Effector')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.title('2-DOF Robot Arm Configuration')
    plt.show()

plot_robot(joint_angles, link_lengths)
```

## Chapter 3: Machine Learning for Locomotion

### Exercise: Simple DDPG Implementation Solution
**Problem**: Implement a basic DDPG agent for a simple control task.

**Solution**:
```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class Actor(nn.Module):
    """Actor network: maps state to action"""
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    """Critic network: maps (state, action) to Q-value"""
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = torch.relu(self.l4(sa))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = deque(maxlen=1000000)
        self.batch_size = 100
        self.gamma = 0.99  # discount factor
        self.tau = 0.005   # target network update rate
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2

        self.total_it = 0  # Initialize iteration counter

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).cpu().data.numpy().flatten()

        if add_noise:
            noise = np.random.normal(0, 0.1, size=len(action))
            action = action + noise
            action = np.clip(action, -1, 1)

        return action

    def train(self, batch_size=100):
        if len(self.replay_buffer) < batch_size:
            return

        batch = random.sample(self.replay_buffer, batch_size)
        state, action, next_state, reward, done = map(np.stack, zip(*batch))

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        done = torch.BoolTensor(done).unsqueeze(1)

        # Compute target Q-value
        with torch.no_grad():
            noise = torch.FloatTensor(action).data.normal_(0, self.policy_noise)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = reward + done.float() * self.gamma * torch.min(target_Q1, target_Q2)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Increment iteration counter
        self.total_it += 1

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, state, action, next_state, reward, done):
        self.replay_buffer.append((state, action, next_state, reward, done))

# Simple environment for testing
class SimpleEnv:
    def __init__(self):
        self.state = np.random.uniform(-1, 1, 2)  # 2D state
        self.action_dim = 2
        self.state_dim = 2
        self.max_action = 1.0

    def reset(self):
        self.state = np.random.uniform(-1, 1, 2)
        return self.state

    def step(self, action):
        # Simple dynamics: move in the direction of action
        self.state += action * 0.1
        # Add some noise
        self.state += np.random.normal(0, 0.01, 2)

        # Reward based on distance to origin (encourage staying near origin)
        distance = np.linalg.norm(self.state)
        reward = -distance  # Negative reward for being far from origin

        # Simple termination condition
        done = distance > 5.0

        return self.state, reward, done, {}

# Test the DDPG implementation
def test_ddpg():
    env = SimpleEnv()
    agent = DDPGAgent(env.state_dim, env.action_dim, env.max_action)

    # Training loop
    for episode in range(1000):
        state = env.reset()
        episode_reward = 0

        for step in range(50):  # 50 steps per episode
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store_transition(state, action, next_state, reward, done)
            agent.train()

            state = next_state
            episode_reward += reward

            if done:
                break

        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")

    print("DDPG training completed!")

# Uncomment to run test
# test_ddpg()
```

## Chapter 4: Simulation Environments

### Exercise: Simple PyBullet Robot Solution
**Problem**: Create a simple robot in PyBullet and implement basic control.

**Solution**:
```python
import pybullet as p
import pybullet_data
import numpy as np
import time

class SimplePyBulletRobot:
    def __init__(self, gui=True):
        # Connect to physics server
        if gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        # Set gravity
        p.setGravity(0, 0, -9.81)

        # Load plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")

        # Create a simple robot (a base with 2 wheels)
        self.robot_id = self._create_simple_robot()

        # Define wheel joints
        self.left_wheel_joint = 0
        self.right_wheel_joint = 1

    def _create_simple_robot(self):
        """Create a simple robot with 2 wheels"""
        # Create collision shape for base
        base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.15, 0.1])

        # Create visual shape for base
        base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.15, 0.1],
                                      rgbaColor=[0.8, 0.4, 0.2, 1])

        # Create base link
        base_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=base_vis,
            basePosition=[0, 0, 0.2]
        )

        # Create wheels
        wheel_radius = 0.1
        wheel_width = 0.05
        wheel_mass = 0.2

        # Left wheel collision and visual shapes
        wheel_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=wheel_radius, length=wheel_width)
        wheel_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=wheel_radius, length=wheel_width,
                                       rgbaColor=[0.3, 0.3, 0.3, 1])

        # Add left wheel (fixed joint)
        p.createConstraint(
            parentBodyUniqueId=base_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0.2, -0.2, 0],
            childFramePosition=[0.2, -0.2, 0]
        )

        # Add right wheel (fixed joint)
        p.createConstraint(
            parentBodyUniqueId=base_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0.2, 0.2, 0],
            childFramePosition=[0.2, 0.2, 0]
        )

        # For simplicity in this example, we'll use a pre-made URDF or create a more complex one
        # Actually, let's create a simpler version using a single body with wheels

        # Remove the first attempt and create a proper robot with joints
        p.removeBody(base_id)

        # Create a proper robot with joints
        # For this example, we'll create a simple differential drive robot
        robotStartPos = [0, 0, 0.2]
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        # Load a simple robot URDF (using a simple one from pybullet data or creating our own)
        # For this example, let's just create a basic model using multiple bodies connected with joints
        return self._create_differential_drive_robot()

    def _create_differential_drive_robot(self):
        """Create a differential drive robot with proper joints"""
        # Base body
        base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.2, 0.1])
        base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.2, 0.1],
                                      rgbaColor=[0.8, 0.4, 0.2, 1])

        base_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=base_vis,
            basePosition=[0, 0, 0.2]
        )

        # Wheel parameters
        wheel_radius = 0.1
        wheel_width = 0.05
        wheel_mass = 0.1

        # Left wheel collision and visual shapes
        wheel_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=wheel_radius, length=wheel_width)
        wheel_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=wheel_radius, length=wheel_width,
                                       rgbaColor=[0.3, 0.3, 0.3, 1])

        # Left wheel
        left_wheel_id = p.createMultiBody(
            baseMass=wheel_mass,
            baseCollisionShapeIndex=wheel_col,
            baseVisualShapeIndex=wheel_vis,
            basePosition=[0.2, -0.2, 0.1]
        )

        # Right wheel
        right_wheel_id = p.createMultiBody(
            baseMass=wheel_mass,
            baseCollisionShapeIndex=wheel_col,
            baseVisualShapeIndex=wheel_vis,
            basePosition=[0.2, 0.2, 0.1]
        )

        # Create revolute joints to connect wheels to base
        p.createConstraint(
            base_id, -1, left_wheel_id, -1, p.JOINT_REVOLUTE,
            [0, 0, 1], [0, -0.2, 0.1], [0, -0.2, 0.1]
        )

        p.createConstraint(
            base_id, -1, right_wheel_id, -1, p.JOINT_REVOLUTE,
            [0, 0, 1], [0, 0.2, 0.1], [0, 0.2, 0.1]
        )

        # Return base ID as the main robot ID
        return base_id

    def set_wheel_velocities(self, left_vel, right_vel):
        """Set target velocities for the wheels"""
        # In a real implementation, we would control the wheel joints
        # For this simplified example, we'll just print the intended velocities
        print(f"Setting wheel velocities - Left: {left_vel}, Right: {right_vel}")

    def step_simulation(self):
        """Step the simulation"""
        p.stepSimulation()
        time.sleep(1./240.)  # Real-time simulation at 240 Hz

    def get_robot_position(self):
        """Get the robot's position and orientation"""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        return pos, orn

    def disconnect(self):
        """Disconnect from physics server"""
        p.disconnect(self.physics_client)

# Example usage
def run_simple_robot_example():
    robot = SimplePyBulletRobot(gui=True)

    # Simple control loop
    for i in range(1000):
        # Simple movement pattern
        t = i / 240.0  # Time in seconds
        left_vel = 2.0 + 0.5 * np.sin(t)  # Varying velocity
        right_vel = 2.0 + 0.5 * np.cos(t)

        robot.set_wheel_velocities(left_vel, right_vel)
        robot.step_simulation()

        if i % 100 == 0:
            pos, orn = robot.get_robot_position()
            print(f"Step {i}, Position: {pos[:2]}")  # Just x,y coordinates

    robot.disconnect()

# Uncomment to run the example
# run_simple_robot_example()
```

## Additional Learning Resources

### Books and Textbooks
1. **"Robotics: Modelling, Planning and Control"** by Siciliano et al. - Comprehensive coverage of robot kinematics, dynamics, and control
2. **"Probabilistic Robotics"** by Thrun, Burgard, and Fox - Focuses on uncertainty in robotics
3. **"Introduction to Autonomous Mobile Robots"** by Siegwart et al. - Good introduction to mobile robotics
4. **"Reinforcement Learning: An Introduction"** by Sutton and Barto - Foundational text for RL

### Online Courses and Tutorials
1. **MIT 6.832 (Underactuated Robotics)** - Advanced robotics course available online
2. **Columbia University Robotics Course** - Available on edX
3. **ETH Zurich Robot Dynamics and Control** - Comprehensive online materials
4. **OpenAI Spinning Up** - Excellent resource for learning RL

### Software Libraries and Frameworks
1. **PyBullet** - Physics simulation with Python API
2. **Gazebo** - ROS-integrated simulation environment
3. **Mujoco** - High-fidelity physics engine
4. **ROS/ROS2** - Robot operating system
5. **OpenAI Gym** - Reinforcement learning environments
6. **Stable Baselines3** - RL algorithms implementation
7. **PyTorch/TensorFlow** - Deep learning frameworks

### Research Papers
1. "Continuous control with deep reinforcement learning" (DDPG) - Lillicrap et al.
2. "Addressing Function Approximation Error in Actor-Critic Methods" (TD3) - Fujimoto et al.
3. "Human-level control through deep reinforcement learning" (DQN) - Mnih et al.
4. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor" (SAC) - Haarnoja et al.

### Communities and Forums
1. **Reddit r/robotics** - Active community for robotics discussions
2. **ROS Answers** - Q&A for ROS users
3. **PyBullet Forum** - Community support for PyBullet
4. **AI Stack Exchange** - General AI and robotics questions

### Hardware Platforms for Practice
1. **TurtleBot3** - Affordable mobile robot platform
2. **JetBot** - NVIDIA's educational robot
3. **Robotis OP3** - Humanoid robot for research
4. **Unitree Go1/A1** - Quadruped robots for research
5. **Arduino/Raspberry Pi** - For building custom robots

## Problem Sets and Challenges

### Beginner Level
1. Implement forward and inverse kinematics for a 2-DOF arm
2. Create a simple PID controller for motor position control
3. Implement a basic path following algorithm
4. Simulate a sensorimotor loop with noise

### Intermediate Level
1. Implement a walking gait for a simple biped
2. Train a DQN agent to balance an inverted pendulum
3. Create a SLAM system using simulated sensors
4. Implement impedance control for safe interaction

### Advanced Level
1. Develop a whole-body controller for a humanoid robot
2. Train a locomotion policy that generalizes across terrains
3. Implement sim-to-real transfer for a manipulation task
4. Create a multi-robot coordination system

## Solutions to Common Implementation Challenges

### Sim-to-Real Transfer
Problem: Policies trained in simulation don't work on real robots.

Solution approaches:
1. Domain randomization - Randomize simulation parameters
2. System identification - Learn real-world dynamics
3. Robust control design - Account for model uncertainty
4. Online adaptation - Update policy based on real experience

### Real-time Control
Problem: Control loops missing timing deadlines.

Solutions:
1. Use real-time operating systems (RT Linux, Xenomai)
2. Optimize computational complexity
3. Use fixed-step integration
4. Implement priority-based scheduling

### Sensor Fusion
Problem: Combining data from multiple sensors reliably.

Solutions:
1. Extended Kalman Filter (EKF) for nonlinear systems
2. Unscented Kalman Filter (UKF) for better nonlinear approximation
3. Particle filters for multimodal distributions
4. Complementary filters for simple fusion tasks

## Assessment Rubrics

### Implementation Projects
- **Functionality (40%)**: Does the system perform as intended?
- **Code Quality (25%)**: Is the code well-structured and documented?
- **Performance (20%)**: How efficiently does the solution work?
- **Robustness (15%)**: How well does it handle edge cases and errors?

### Theoretical Understanding
- **Conceptual Understanding (50%)**: Grasp of fundamental principles
- **Application (30%)**: Ability to apply concepts to new situations
- **Analysis (20%)**: Critical thinking and problem-solving skills

## Recommended Study Path

### Month 1: Foundations
- Week 1-2: Mathematics review (linear algebra, calculus, probability)
- Week 3-4: Introduction to robotics and kinematics

### Month 2: Control and Dynamics
- Week 5-6: Control theory and implementation
- Week 7-8: Dynamics and system modeling

### Month 3: Perception and Learning
- Week 9-10: Sensors and perception
- Week 11-12: Machine learning for robotics

### Month 4: Integration and Application
- Week 13-14: Simulation environments
- Week 15-16: Real-world implementation and testing

This comprehensive set of solutions and resources should provide students with the tools they need to understand and implement the concepts covered in the Physical AI & Humanoid Robotics book.