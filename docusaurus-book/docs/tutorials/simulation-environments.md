---
id: simulation-environments
title: Simulation Environments for Physical AI
sidebar_label: Simulation Environments
---

# Simulation Environments for Physical AI

## Introduction to Robotics Simulation

Simulation environments are crucial tools in physical AI development, allowing researchers and developers to test algorithms, train controllers, and validate designs before deploying on real hardware. This tutorial covers the most popular simulation environments used in humanoid robotics and embodied AI research.

### Why Simulation is Important

Simulation provides several key benefits for physical AI development:

1. **Safety**: Test dangerous behaviors without risk to hardware or humans
2. **Cost-Effectiveness**: Reduce the need for expensive physical prototypes
3. **Speed**: Run experiments faster than real-time
4. **Reproducibility**: Create consistent testing conditions
5. **Iteration**: Rapidly test and refine algorithms

### Simulation Fidelity Trade-offs

When choosing a simulation environment, consider the trade-offs between:

- **Accuracy vs. Speed**: More accurate physics requires more computation
- **Realism vs. Simplification**: Complex environments may slow development
- **Development Time vs. Quality**: Higher fidelity simulations take longer to set up

## Gazebo: The ROS-Integrated Simulator

Gazebo is the most popular simulation environment for ROS-based robotics projects, offering high-fidelity physics simulation and extensive sensor modeling.

### Installation and Setup

```bash
# Install Gazebo on Ubuntu with ROS
sudo apt update
sudo apt install gazebo11 libgazebo11-dev

# If using ROS Noetic
sudo apt install ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control
```

### Basic Gazebo Concepts

Gazebo operates on several core concepts:

- **Worlds**: Environments containing models and physics properties
- **Models**: Physical objects with URDF/SDF definitions
- **Plugins**: Custom code that extends simulation capabilities
- **Sensors**: Simulated sensor devices (cameras, IMUs, LIDAR, etc.)

### Creating a Simple World

```xml
<!-- simple_world.world -->
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="simple_world">
    <!-- Physics engine -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Sun light -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Your robot model would be included here -->
    <include>
      <uri>model://your_robot</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>

    <!-- Optional: Additional objects -->
    <model name="box">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.1667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1667</iyy>
            <iyz>0</iyz>
            <izz>0.1667</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

### URDF Model Definition

URDF (Unified Robot Description Format) is commonly used with Gazebo:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base/Body link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.4"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.004"/>
    </inertial>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="3"/>
    <dynamics damping="1" friction="0.1"/>
  </joint>

  <!-- Left leg -->
  <link name="left_hip">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.027" ixy="0" ixz="0" iyy="0.025" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="left_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_hip"/>
    <origin xyz="-0.1 0 -0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="200" velocity="2"/>
    <dynamics damping="2" friction="0.2"/>
  </joint>

  <!-- Additional joints and links would continue similarly -->

  <!-- Gazebo-specific tags for simulation -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="head">
    <material>Gazebo/White</material>
  </gazebo>

  <gazebo reference="left_hip">
    <material>Gazebo/Gray</material>
  </gazebo>

  <!-- Transmission for ROS control -->
  <transmission name="left_hip_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_hip_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_hip_motor">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>
</robot>
```

### Gazebo Controller Integration

```cpp
// Example Gazebo plugin for joint control
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <stdio.h>

namespace gazebo
{
  class JointControlPlugin : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
    {
      // Store the model pointer for convenience
      this->model = _parent;

      // Get the joint
      this->joint = this->model->GetJoint("left_hip_joint");

      // Listen to the update event. This event is broadcast every
      // simulation iteration.
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&JointControlPlugin::OnUpdate, this));
    }

    // Called by the world update start event
    public: void OnUpdate()
    {
      // Apply a small sine wave torque to the joint
      double torque = 0.1 * sin(gazebo::common::Time::GetWallTime().Double());
      this->joint->SetForce(0, torque);
    }

    // Pointer to the model
    private: physics::ModelPtr model;

    // Pointer to the joint
    private: physics::JointPtr joint;

    // Event connection
    private: event::ConnectionPtr updateConnection;
  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(JointControlPlugin)
}
```

### Python Control Interface

```python
#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import math

class GazeboController:
    def __init__(self):
        rospy.init_node('gazebo_controller', anonymous=True)

        # Publishers for joint commands
        self.joint_publishers = {
            'left_hip_joint': rospy.Publisher('/simple_humanoid/left_hip_joint_position_controller/command',
                                            Float64, queue_size=10),
            'right_hip_joint': rospy.Publisher('/simple_humanoid/right_hip_joint_position_controller/command',
                                             Float64, queue_size=10),
            # Add more joints as needed
        }

        # Subscriber for joint states
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)

        self.current_joint_states = JointState()

        # Timer for control loop
        self.control_timer = rospy.Timer(rospy.Duration(0.01), self.control_loop)  # 100 Hz

    def joint_state_callback(self, msg):
        """Callback for joint state updates"""
        self.current_joint_states = msg

    def control_loop(self, event):
        """Main control loop"""
        # Simple walking pattern
        t = rospy.Time.now().to_sec()

        # Generate walking motion
        left_hip_pos = 0.2 * math.sin(t * 2)  # 2 rad/s oscillation
        right_hip_pos = 0.2 * math.sin(t * 2 + math.pi)  # Opposite phase

        # Publish commands
        self.joint_publishers['left_hip_joint'].publish(Float64(left_hip_pos))
        self.joint_publishers['right_hip_joint'].publish(Float64(right_hip_pos))

    def move_to_pose(self, joint_positions, duration=5.0):
        """Move robot to specified joint positions using trajectory controller"""
        trajectory_pub = rospy.Publisher('/simple_humanoid/joint_trajectory_controller/command',
                                       JointTrajectory, queue_size=10)

        traj = JointTrajectory()
        traj.joint_names = list(joint_positions.keys())

        point = JointTrajectoryPoint()
        point.positions = list(joint_positions.values())
        point.velocities = [0.0] * len(joint_positions)
        point.time_from_start = rospy.Duration(duration)

        traj.points = [point]
        traj.header.stamp = rospy.Time.now()

        trajectory_pub.publish(traj)

if __name__ == '__main__':
    try:
        controller = GazeboController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

## PyBullet: Fast Physics Simulation

PyBullet is a Python-based physics engine that provides fast simulation with good integration for machine learning applications.

### Installation

```bash
pip install pybullet
```

### Basic PyBullet Usage

```python
import pybullet as p
import pybullet_data
import numpy as np
import time

class PyBulletRobot:
    def __init__(self, urdf_path="robot.urdf", gui=True):
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

        # Load robot
        self.robot_id = p.loadURDF(urdf_path, [0, 0, 1])

        # Get joint information
        self.joint_info = {}
        self.motor_joints = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]

            if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                self.motor_joints.append(i)
                self.joint_info[i] = {
                    'name': joint_name,
                    'type': joint_type,
                    'lower_limit': joint_info[8],
                    'upper_limit': joint_info[9],
                    'max_force': joint_info[10],
                    'max_velocity': joint_info[11]
                }

        print(f"Loaded robot with {len(self.motor_joints)} motor joints")

    def set_joint_positions(self, joint_positions):
        """Set target positions for all joints"""
        for i, joint_id in enumerate(self.motor_joints):
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=self.joint_info[joint_id]['max_force']
            )

    def set_joint_velocities(self, joint_velocities):
        """Set target velocities for all joints"""
        for i, joint_id in enumerate(self.motor_joints):
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_id,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=joint_velocities[i],
                force=self.joint_info[joint_id]['max_force']
            )

    def set_joint_torques(self, joint_torques):
        """Apply torques to all joints"""
        p.setJointMotorControlArray(
            bodyIndex=self.robot_id,
            jointIndices=self.motor_joints,
            controlMode=p.TORQUE_CONTROL,
            forces=joint_torques
        )

    def get_joint_states(self):
        """Get current joint positions and velocities"""
        joint_states = p.getJointStates(self.robot_id, self.motor_joints)
        positions = [state[0] for state in joint_states]
        velocities = [state[1] for state in joint_states]
        return positions, velocities

    def get_link_state(self, link_index):
        """Get position and orientation of a specific link"""
        return p.getLinkState(self.robot_id, link_index)

    def step_simulation(self):
        """Step the simulation forward"""
        p.stepSimulation()
        time.sleep(1./240.)  # Real-time simulation at 240 Hz

    def reset_robot(self, position=[0, 0, 1]):
        """Reset robot to initial position"""
        p.resetBasePositionAndOrientation(self.robot_id, position, [0, 0, 0, 1])
        # Reset joint positions to zero
        for joint_id in self.motor_joints:
            p.resetJointState(self.robot_id, joint_id, 0)

    def add_obstacle(self, position, size=[1, 1, 1]):
        """Add a box obstacle to the environment"""
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
        visualShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[1, 0, 0, 1])
        obstacle_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId,
                                       baseVisualShapeIndex=visualShapeId, basePosition=position)
        return obstacle_id

    def disconnect(self):
        """Disconnect from physics server"""
        p.disconnect(self.physics_client)

# Example usage
def main():
    # Create simulation
    robot_sim = PyBulletRobot(gui=True)

    # Add some obstacles
    robot_sim.add_obstacle([2, 0, 0.5], [0.2, 0.2, 0.5])
    robot_sim.add_obstacle([-1, 1, 0.5], [0.3, 0.3, 0.5])

    # Simple control loop
    for i in range(1000):
        # Simple walking pattern
        t = i / 240.0  # Time in seconds (240 Hz)

        # Generate joint positions for walking
        joint_positions = [0] * len(robot_sim.motor_joints)
        if len(joint_positions) >= 2:
            joint_positions[0] = 0.2 * np.sin(t * 2)  # Left hip
            joint_positions[1] = -0.2 * np.sin(t * 2)  # Right hip (opposite phase)

        robot_sim.set_joint_positions(joint_positions)
        robot_sim.step_simulation()

    robot_sim.disconnect()

if __name__ == "__main__":
    main()
```

### PyBullet for Reinforcement Learning

```python
import pybullet as p
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PyBulletEnvironment:
    def __init__(self, robot_urdf, gui=False):
        self.robot_urdf = robot_urdf
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load plane and robot
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(robot_urdf, [0, 0, 1])

        # Get motor joints
        self.motor_joints = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.motor_joints.append(i)

        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 1], [0, 0, 0, 1])

        # Reset all joints to zero
        for joint_id in self.motor_joints:
            p.resetJointState(self.robot_id, joint_id, 0)

        return self.get_observation()

    def get_observation(self):
        """Get current observation from environment"""
        # Get joint states
        joint_states = p.getJointStates(self.robot_id, self.motor_joints)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        # Get base position and orientation
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)

        # Combine into observation vector
        observation = np.concatenate([
            joint_positions,
            joint_velocities,
            pos,  # Base position
            orn,  # Base orientation
            lin_vel,  # Base linear velocity
            ang_vel   # Base angular velocity
        ])

        return observation

    def step(self, action):
        """Execute action and return (observation, reward, done, info)"""
        # Apply action (torques or positions)
        # Here we'll use position control with the action as target positions
        for i, joint_id in enumerate(self.motor_joints):
            if i < len(action):
                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=joint_id,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=action[i],
                    force=100  # Max force
                )

        # Step simulation
        p.stepSimulation()

        # Get new observation
        observation = self.get_observation()

        # Calculate reward (example: move forward)
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        reward = pos[0] * 0.1  # Positive X is forward, reward for forward movement

        # Check if episode is done (example: robot fell)
        done = pos[2] < 0.3  # If robot height is too low, it probably fell

        info = {}  # Additional information

        return observation, reward, done, info

    def close(self):
        """Close the environment"""
        p.disconnect(self.physics_client)

# Example: Simple policy network for PyBullet environment
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions are bounded to [-1, 1]
        )

    def forward(self, state):
        return self.network(state)

# Example: Simple policy gradient training
def train_simple_policy():
    env = PyBulletEnvironment("simple_robot.urdf", gui=False)

    state_dim = len(env.get_observation())
    action_dim = len(env.motor_joints)

    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    # Training loop (simplified)
    for episode in range(100):
        state = env.reset()
        total_reward = 0

        for step in range(200):  # Max steps per episode
            state_tensor = torch.FloatTensor(state)
            action = policy(state_tensor).detach().numpy()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break

            state = next_state

        print(f"Episode {episode}, Total Reward: {total_reward:.2f}")

    env.close()
```

## Mujoco: High-Fidelity Physics

Mujoco (Multi-Joint dynamics with Contact) is a commercial physics engine known for high-fidelity simulation.

### Mujoco Installation

```bash
pip install mujoco
```

### Mujoco XML Model Example

```xml
<!-- simple_humanoid.xml -->
<mujoco model="simple_humanoid">
  <compiler angle="degree" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom friction="0.8 0.1 0.1" margin="0.001" rgba="0.8 0.6 0.4 1"/>
  </default>

  <option integrator="RK4" timestep="0.002"/>

  <worldbody>
    <light directional="false" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" pos="0 0 0" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>

    <body name="torso" pos="0 0 1.2">
      <geom name="torso_geom" type="capsule" fromto="0 0 -0.1 0 0 0.1" size="0.15"/>
      <joint name="root" type="free" pos="0 0 0"/>

      <body name="head" pos="0 0 0.2">
        <geom name="head_geom" type="sphere" size="0.1"/>
      </body>

      <body name="right_thigh" pos="0 -0.1 -0.2">
        <geom name="right_thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.06"/>
        <joint name="right_hip" type="hinge" pos="0 0 0" axis="0 1 0" range="-20 100"/>

        <body name="right_shin" pos="0 0 -0.6">
          <geom name="right_shin_geom" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.05"/>
          <joint name="right_knee" type="hinge" pos="0 0 0.02" axis="0 1 0" range="-150 0"/>

          <body name="right_foot" pos="0 0 -0.35">
            <geom name="right_foot_geom" type="box" size="0.1 0.025 0.05" pos="0 0 0.025"/>
            <joint name="right_ankle" type="hinge" pos="0 0 0.05" axis="0 1 0" range="-45 45"/>
          </body>
        </body>
      </body>

      <body name="left_thigh" pos="0 0.1 -0.2">
        <geom name="left_thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.06"/>
        <joint name="left_hip" type="hinge" pos="0 0 0" axis="0 1 0" range="-20 100"/>

        <body name="left_shin" pos="0 0 -0.6">
          <geom name="left_shin_geom" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.05"/>
          <joint name="left_knee" type="hinge" pos="0 0 0.02" axis="0 1 0" range="-150 0"/>

          <body name="left_foot" pos="0 0 -0.35">
            <geom name="left_foot_geom" type="box" size="0.1 0.025 0.05" pos="0 0 0.025"/>
            <joint name="left_ankle" type="hinge" pos="0 0 0.05" axis="0 1 0" range="-45 45"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="right_hip" joint="right_hip" gear="100"/>
    <motor name="right_knee" joint="right_knee" gear="100"/>
    <motor name="right_ankle" joint="right_ankle" gear="50"/>
    <motor name="left_hip" joint="left_hip" gear="100"/>
    <motor name="left_knee" joint="left_knee" gear="100"/>
    <motor name="left_ankle" joint="left_ankle" gear="50"/>
  </actuator>
</mujoco>
```

### Mujoco Python Control

```python
import mujoco
import mujoco.viewer
import numpy as np

class MujocoHumanoid:
    def __init__(self, model_path="simple_humanoid.xml"):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Initialize the model
        mujoco.mj_forward(self.model, self.data)

    def reset(self):
        """Reset the simulation to initial state"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        """Step the simulation with given actions"""
        # Set control signals
        self.data.ctrl[:] = action

        # Step simulation
        mujoco.mj_step(self.model, self.data)

    def get_observation(self):
        """Get current observation"""
        # Example observation: joint positions, velocities, body positions
        obs = np.concatenate([
            self.data.qpos,  # Joint positions
            self.data.qvel,  # Joint velocities
            self.data.body("torso").xpos,  # Torso position
            self.data.body("torso").xquat, # Torso orientation
        ])
        return obs

    def get_reward(self):
        """Calculate reward (example: forward velocity)"""
        # Reward for moving forward
        torso_vel = self.data.body("torso").cvel[1]  # Linear velocity in x direction
        reward = torso_vel * 0.1

        # Small penalty for large torques
        torque_penalty = -0.001 * np.sum(np.square(self.data.ctrl))
        reward += torque_penalty

        return reward

    def render(self):
        """Render the simulation"""
        # This would typically be used with mujoco.viewer
        pass

# Example usage
def run_mujoco_simulation():
    humanoid = MujocoHumanoid()

    # Run simulation with simple controller
    for i in range(1000):
        # Simple controller: move legs in alternating pattern
        t = i * 0.002  # Assuming 0.002s timestep

        action = np.array([
            0.2 * np.sin(t * 5),    # Right hip
            -0.3 * np.sin(t * 5),   # Right knee
            0.1 * np.sin(t * 5),    # Right ankle
            0.2 * np.sin(t * 5 + np.pi),  # Left hip (opposite phase)
            -0.3 * np.sin(t * 5 + np.pi), # Left knee
            0.1 * np.sin(t * 5 + np.pi),  # Left ankle
        ])

        humanoid.step(action)

        # Print reward every 100 steps
        if i % 100 == 0:
            reward = humanoid.get_reward()
            print(f"Step {i}, Reward: {reward:.3f}")

    print("Simulation completed")

# Run with viewer
def run_with_viewer():
    humanoid = MujocoHumanoid()

    with mujoco.viewer.launch_passive(humanoid.model, humanoid.data) as viewer:
        while viewer.is_running():
            # Simple walking controller
            t = humanoid.data.time
            action = np.array([
                0.2 * np.sin(t * 5),
                -0.3 * np.sin(t * 5),
                0.1 * np.sin(t * 5),
                0.2 * np.sin(t * 5 + np.pi),
                -0.3 * np.sin(t * 5 + np.pi),
                0.1 * np.sin(t * 5 + np.pi),
            ])

            humanoid.step(action)
            viewer.sync()
```

## Isaac Gym: GPU-Accelerated Simulation

Isaac Gym provides GPU-accelerated simulation for large-scale RL training.

### Isaac Gym Installation

```bash
# Isaac Gym is part of NVIDIA Omniverse
# Download from NVIDIA Developer website
# pip install -e . # in the Isaac Gym installation directory
```

### Isaac Gym Example

```python
import isaacgym
import torch
import numpy as np

# Import required Isaac Gym features
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

class IsaacGymHumanoid:
    def __init__(self, headless=True):
        # Initialize gym
        self.gym = gymapi.acquire_gym()

        # Configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0, 0, -9.81)

        # Set physics engine parameters
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
        sim_params.physx.num_threads = 4
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.friction_offset_threshold = 0.01

        # Create sim
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        # Create viewer
        if not headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.Vec3(5, 5, 1))
            if self.viewer is None:
                print("*** Failed to create viewer")
                quit()
        else:
            self.viewer = None

        # Initialize environments
        self._create_envs()

    def _create_envs(self):
        # Set default environment spacing
        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # Load robot asset
        asset_root = "path/to/assets"
        asset_file = "simple_humanoid.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0

        print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
        self.humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Create environment
        num_envs = 1
        self.envs = []
        for i in range(num_envs):
            # Create environment
            env = self.gym.create_env(self.sim, env_lower, env_upper, 1)
            self.envs.append(env)

            # Add robot to environment
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            # Add robot to environment
            humanoid_handle = self.gym.create_actor(env, self.humanoid_asset, pose, "humanoid", i, 1, 0)

            # Store the number of DOFs
            self.num_dofs = self.gym.get_actor_dof_count(env, humanoid_handle)

            # Set DOF properties
            dof_props = self.gym.get_actor_dof_properties(env, humanoid_handle)
            dof_props["driveMode"] = gymapi.DOF_MODE_EFFORT
            dof_props["stiffness"] = 0.0
            dof_props["damping"] = 0.1
            self.gym.set_actor_dof_properties(env, humanoid_handle, dof_props)

            # Initialize DOF control
            self.pos_action = torch.zeros((len(self.envs), self.num_dofs), dtype=torch.float32, device="cpu")
            self.effort_action = torch.zeros((len(self.envs), self.num_dofs), dtype=torch.float32, device="cpu")

    def step(self, actions):
        """Step the simulation with actions"""
        # Set actions
        self.effort_action[:] = torch.tensor(actions, dtype=torch.float32)

        # Apply actions
        for i, env in enumerate(self.envs):
            actor_handle = self.gym.get_actor_handle(env, 0)
            self.gym.apply_actor_dof_efforts(env, actor_handle, self.effort_action[i].numpy())

        # Step simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Update viewer
        if self.viewer:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)

    def reset(self):
        """Reset all environments"""
        pass

    def get_observation(self):
        """Get current observations"""
        obs = []
        for env in self.envs:
            # Get actor state
            actor_handle = self.gym.get_actor_handle(env, 0)
            actor_rigid_body_states = self.gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_POS)
            obs.append(actor_rigid_body_states)

        return torch.tensor(obs, dtype=torch.float32)

    def close(self):
        """Close the simulation"""
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
```

## Best Practices for Simulation

### Model Accuracy

```python
# Tips for creating accurate simulation models

# 1. Proper mass and inertia properties
def calculate_inertia_solid_cylinder(mass, radius, length):
    """Calculate inertia tensor for a solid cylinder"""
    # For a cylinder aligned along z-axis
    ixx = (1/12) * mass * (3 * radius**2 + length**2)
    iyy = ixx  # Ixx = Iyy for cylinder
    izz = (1/2) * mass * radius**2

    return np.array([ixx, iyy, izz, 0, 0, 0])  # Ixx, Iyy, Izz, Ixy, Ixz, Iyz

# 2. Realistic joint limits and dynamics
def create_realistic_joint_params(joint_type, joint_name):
    """Create realistic joint parameters based on joint type"""
    params = {}

    if joint_type == "hip":
        params['range'] = (-45, 90)  # degrees
        params['max_effort'] = 200   # N*m
        params['max_velocity'] = 3   # rad/s
        params['damping'] = 2.0
        params['friction'] = 0.5
    elif joint_type == "knee":
        params['range'] = (-150, 0)  # degrees
        params['max_effort'] = 150
        params['max_velocity'] = 4
        params['damping'] = 1.5
        params['friction'] = 0.3
    elif joint_type == "ankle":
        params['range'] = (-30, 30)  # degrees
        params['max_effort'] = 50
        params['max_velocity'] = 5
        params['damping'] = 0.8
        params['friction'] = 0.2
    else:
        # Default values
        params['range'] = (-90, 90)
        params['max_effort'] = 100
        params['max_velocity'] = 2
        params['damping'] = 1.0
        params['friction'] = 0.1

    return params

# 3. Sensor noise modeling
class NoisySensor:
    """Add realistic noise to sensor readings"""
    def __init__(self, mean=0.0, std_dev=0.01, bias=0.0):
        self.mean = mean
        self.std_dev = std_dev
        self.bias = bias

    def add_noise(self, true_value):
        """Add noise to a true sensor value"""
        noise = np.random.normal(self.mean, self.std_dev)
        return true_value + self.bias + noise

# Example usage
position_sensor = NoisySensor(mean=0.0, std_dev=0.001, bias=0.0005)  # 1mm std dev, 0.5mm bias
noisy_reading = position_sensor.add_noise(1.234)  # True value was 1.234m
```

### Simulation Fidelity Considerations

```python
# Guidelines for simulation fidelity

class SimulationFidelityManager:
    """Manage different levels of simulation fidelity"""

    def __init__(self):
        self.fidelity_levels = {
            'fast_research': {
                'timestep': 0.01,
                'solver_iterations': 10,
                'contact_approximation': 'mesh',
                'sensor_noise': 0.05,
                'actuator_delay': 0.02
            },
            'accurate_validation': {
                'timestep': 0.001,
                'solver_iterations': 100,
                'contact_approximation': 'convex',
                'sensor_noise': 0.01,
                'actuator_delay': 0.005
            },
            'hardware_test': {
                'timestep': 0.002,
                'solver_iterations': 50,
                'contact_approximation': 'convex',
                'sensor_noise': 0.02,
                'actuator_delay': 0.01
            }
        }

    def set_fidelity(self, level='accurate_validation'):
        """Set simulation parameters based on fidelity level"""
        if level not in self.fidelity_levels:
            raise ValueError(f"Unknown fidelity level: {level}")

        params = self.fidelity_levels[level]
        print(f"Setting fidelity to {level}: {params}")
        return params

# Usage
fidelity_manager = SimulationFidelityManager()
params = fidelity_manager.set_fidelity('fast_research')
```

## Learning Objectives

After completing this tutorial, you should be able to:

1. Set up and configure different simulation environments (Gazebo, PyBullet, Mujoco)
2. Create robot models in URDF/SDF/XML formats
3. Implement basic control algorithms in simulation
4. Understand the trade-offs between different simulation environments
5. Apply best practices for realistic simulation modeling
6. Integrate simulation with reinforcement learning frameworks

## Hands-On Exercise

Create a simple walking controller for a humanoid robot in your chosen simulation environment:

1. Choose one simulation environment (Gazebo, PyBullet, or Mujoco)
2. Create or use an existing humanoid robot model
3. Implement a simple walking gait pattern
4. Add sensors (IMU, joint encoders) to the robot
5. Implement a basic balance controller
6. Test the controller in simulation and analyze the results

This exercise will help you understand the practical aspects of simulation-based robot development and control.