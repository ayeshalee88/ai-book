---
sidebar_position: 7
title: Tutorials and Hands-on Guides
---

# Tutorials and Hands-on Guides

## Introduction to Tutorials

This section provides practical, hands-on tutorials for building embodied AI systems. Each tutorial includes step-by-step instructions, code examples, and exercises to reinforce learning. The tutorials progress from basic concepts to advanced implementations.

## Setting Up Your Development Environment

### Required Software and Tools

#### Robot Operating System (ROS)
ROS provides the framework for robot software development:

```bash
# Install ROS Noetic (Ubuntu 20.04)
sudo apt update
sudo apt install ros-noetic-desktop-full
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Install ROS dependencies
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
sudo rosdep init
rosdep update
```

#### Simulation Environment
Gazebo provides a physics-based simulation environment:

```bash
sudo apt install ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control
```

#### Python Development
```bash
# Install Python packages for robotics
pip install numpy scipy matplotlib
pip install torch torchvision  # For deep learning
pip install opencv-python     # For computer vision
pip install transforms3d      # For 3D transformations
```

### Hardware Setup (Optional)
For physical robot tutorials, you may need:
- Arduino/Raspberry Pi for basic robotics
- Robot kit (e.g., TurtleBot3, Poppy Ergo Jr)
- Sensors (cameras, IMU, distance sensors)

## Tutorial 1: Basic Robot Simulation

### Creating Your First Robot Model

Let's create a simple differential drive robot in Gazebo:

#### URDF Robot Model
Create a URDF (Unified Robot Description Format) file to define your robot:

```xml
<!-- simple_robot.urdf -->
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Left Wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right Wheel -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="-0.2 0.15 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="-0.2 -0.15 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <!-- Differential Drive Plugin -->
  <gazebo>
    <plugin name="differential_drive_controller" filename="libgazebo_ros_diff_drive.so">
      <leftJoint>left_wheel_joint</leftJoint>
      <rightJoint>right_wheel_joint</rightJoint>
      <wheelSeparation>0.3</wheelSeparation>
      <wheelDiameter>0.2</wheelDiameter>
      <commandTopic>cmd_vel</commandTopic>
      <odometryTopic>odom</odometryTopic>
      <odometryFrame>odom</odometryFrame>
      <robotBaseFrame>base_link</robotBaseFrame>
    </plugin>
  </gazebo>
</robot>
```

#### Launch File
Create a launch file to spawn your robot in Gazebo:

```xml
<!-- launch/simple_robot.launch -->
<launch>
  <!-- Start Gazebo with empty world -->
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="paused" value="false"/>
    <arg name="use_sim_time" value="true"/>
    <arg name="gui" value="true"/>
    <arg name="headless" value="false"/>
    <arg name="debug" value="false"/>
  </include>

  <!-- Load robot description -->
  <param name="robot_description" command="$(find xacro)/xacro --inorder '$(find simple_robot)/urdf/simple_robot.urdf'" />

  <!-- Spawn robot in Gazebo -->
  <node name="spawn_urdf" pkg="gazebo_ros" type="spawn_model"
        args="-param robot_description -model simple_robot -x 0 -y 0 -z 0.1" />

  <!-- Robot state publisher -->
  <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher" />
</launch>
```

### Running the Simulation
```bash
# Create a catkin workspace
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin_make
source devel/setup.bash

# Create your robot package
cd src
catkin_create_pkg simple_robot std_msgs rospy roscpp
cd ..

# Copy the URDF file to simple_robot/urdf/
# Copy the launch file to simple_robot/launch/

# Build and run
catkin_make
source devel/setup.bash
roslaunch simple_robot simple_robot.launch
```

### Controlling the Robot
Create a simple Python script to control your robot:

```python
#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist

def move_robot():
    rospy.init_node('robot_mover', anonymous=True)
    velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(10)  # 10hz

    vel_msg = Twist()

    # Linear velocity in x
    vel_msg.linear.x = 0.5
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0

    # Angular velocity in z
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0.2

    while not rospy.is_shutdown():
        velocity_publisher.publish(vel_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        move_robot()
    except rospy.ROSInterruptException:
        pass
```

## Tutorial 2: Sensor Integration and Perception

### Adding Sensors to Your Robot

Let's enhance our robot with sensors:

```xml
<!-- Enhanced robot with sensors -->
<robot name="sensor_robot">
  <!-- Base link with camera -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Camera sensor -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.2 0 0.05" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo plugins for sensors -->
  <gazebo reference="camera_link">
    <sensor type="camera" name="camera1">
      <update_rate>30.0</update_rate>
      <camera name="head">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>800</width>
          <height>600</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.02</near>
          <far>300</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <alwaysOn>true</alwaysOn>
        <updateRate>0.0</updateRate>
        <cameraName>simple_camera</cameraName>
        <imageTopicName>image_raw</imageTopicName>
        <cameraInfoTopicName>camera_info</cameraInfoTopicName>
        <frameName>camera_link</frameName>
        <hackBaseline>0.07</hackBaseline>
        <distortion_k1>0.0</distortion_k1>
        <distortion_k2>0.0</distortion_k2>
        <distortion_k3>0.0</distortion_k3>
        <distortion_t1>0.0</distortion_t1>
        <distortion_t2>0.0</distortion_t2>
      </plugin>
    </sensor>
  </gazebo>
</robot>
```

### Processing Sensor Data

```python
#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class ImageProcessor:
    def __init__(self):
        rospy.init_node('image_processor', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/simple_camera/image_raw", Image, self.callback)
        self.cv_image = None

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr(e)

    def detect_objects(self):
        if self.cv_image is not None:
            # Simple color-based object detection
            hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)

            # Define range for red color
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])

            mask = cv2.inRange(hsv, lower_red, upper_red)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours on image
            output = self.cv_image.copy()
            cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

            # Display result
            cv2.imshow("Processed Image", output)
            cv2.waitKey(1)

def main():
    processor = ImageProcessor()
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        processor.detect_objects()
        rate.sleep()

if __name__ == '__main__':
    main()
```

## Tutorial 3: AI Integration with Reinforcement Learning

### Setting up a Simple RL Environment

```python
#!/usr/bin/env python3
import rospy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import random

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.tanh(x)  # Actions between -1 and 1

class RobotEnvironment:
    def __init__(self):
        rospy.init_node('rl_robot', anonymous=True)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        self.scan_data = None
        self.state_size = 10  # Simplified state representation
        self.action_size = 2  # Linear and angular velocity

        # Initialize policy network
        self.policy = PolicyNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)

        self.rate = rospy.Rate(10)  # 10 Hz

    def scan_callback(self, data):
        # Process laser scan data
        self.scan_data = np.array(data.ranges)

    def get_state(self):
        if self.scan_data is None:
            return np.zeros(self.state_size)

        # Simplify the laser scan to fixed size state
        # Take readings at regular intervals
        step = len(self.scan_data) // self.state_size
        simplified = [self.scan_data[i] if not np.isnan(self.scan_data[i]) else 10.0
                     for i in range(0, len(self.scan_data), step)]

        # Pad if necessary
        if len(simplified) < self.state_size:
            simplified.extend([10.0] * (self.state_size - len(simplified)))

        return np.array(simplified[:self.state_size])

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.policy(state_tensor)
        return action.numpy()[0]

    def move_robot(self, action):
        cmd = Twist()
        cmd.linear.x = float(action[0]) * 0.5  # Scale linear velocity
        cmd.angular.z = float(action[1]) * 0.5  # Scale angular velocity
        self.cmd_pub.publish(cmd)

    def run_episode(self):
        for _ in range(100):  # 10 seconds at 10 Hz
            state = self.get_state()
            action = self.select_action(state)
            self.move_robot(action)
            self.rate.sleep()

def main():
    env = RobotEnvironment()

    # Run for multiple episodes
    for episode in range(1000):
        rospy.loginfo(f"Episode {episode}")
        env.run_episode()

if __name__ == '__main__':
    main()
```

## Tutorial 4: Humanoid Robot Simulation with PyBullet

### Setting up PyBullet Environment

```python
#!/usr/bin/env python3
import pybullet as p
import pybullet_data
import numpy as np
import time

class SimpleHumanoid:
    def __init__(self, urdf_path):
        # Connect to physics server
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set gravity
        p.setGravity(0, 0, -9.81)

        # Load plane
        p.loadURDF("plane.urdf")

        # Load humanoid robot
        self.robot_id = p.loadURDF(urdf_path, [0, 0, 1])

        # Get joint information
        self.joint_indices = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] != p.JOINT_FIXED:  # Not a fixed joint
                self.joint_indices.append(i)

        # Set control parameters
        self.max_force = 100

    def move_to_position(self, joint_positions):
        """Move all joints to specified positions"""
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=self.max_force
            )

    def get_robot_state(self):
        """Get current joint positions and velocities"""
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        positions = [state[0] for state in joint_states]
        velocities = [state[1] for state in joint_states]
        return positions, velocities

    def run_simulation(self, steps=1000):
        """Run the simulation for specified steps"""
        for step in range(steps):
            # Simple walking pattern
            t = step / 10.0  # Time parameter

            # Generate walking motion (simplified)
            left_leg_pos = np.sin(t) * 0.2
            right_leg_pos = np.sin(t + np.pi) * 0.2

            # Set joint positions for walking
            if len(self.joint_indices) >= 4:  # At least 4 joints for legs
                # Left hip
                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=self.joint_indices[0],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=left_leg_pos,
                    force=self.max_force
                )

                # Right hip
                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=self.joint_indices[1],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=right_leg_pos,
                    force=self.max_force
                )

            p.stepSimulation()
            time.sleep(1./240.)  # Real-time simulation

def main():
    # Create a simple humanoid URDF or use an existing one
    humanoid = SimpleHumanoid("path/to/humanoid.urdf")
    humanoid.run_simulation()
    p.disconnect()

if __name__ == "__main__":
    main()
```

## Tutorial 5: Integrating LLMs with Physical Robots

### Using OpenAI API for Robot Commands

```python
#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import openai
import speech_recognition as sr
import pyttsx3

class LLMRobotController:
    def __init__(self):
        rospy.init_node('llm_robot_controller', anonymous=True)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Initialize speech components
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()

        # Set OpenAI API key (use environment variable in production)
        openai.api_key = "YOUR_API_KEY_HERE"

        # Define robot actions
        self.actions = {
            'move_forward': self.move_forward,
            'move_backward': self.move_backward,
            'turn_left': self.turn_left,
            'turn_right': self.turn_right,
            'stop': self.stop_robot
        }

    def recognize_speech(self):
        """Capture and recognize speech"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = self.recognizer.listen(source)

        try:
            text = self.recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Error with speech recognition service: {e}")
            return None

    def generate_response(self, command):
        """Use LLM to interpret command and generate response"""
        prompt = f"""
        You are a helpful robot assistant. A user has given the following command: "{command}"

        Available actions: move_forward, move_backward, turn_left, turn_right, stop

        Respond with the action to take and a brief explanation.
        Format: ACTION: [action_name], EXPLANATION: [explanation]
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error with LLM: {e}")
            return None

    def parse_llm_response(self, response):
        """Parse the LLM response to extract action"""
        if not response:
            return None

        # Simple parsing - in practice, use more robust NLP
        response_lower = response.lower()

        if 'move_forward' in response_lower or 'forward' in response_lower:
            return 'move_forward'
        elif 'move_backward' in response_lower or 'backward' in response_lower:
            return 'move_backward'
        elif 'turn_left' in response_lower or 'left' in response_lower:
            return 'turn_left'
        elif 'turn_right' in response_lower or 'right' in response_lower:
            return 'turn_right'
        elif 'stop' in response_lower:
            return 'stop'

        return None

    def move_forward(self):
        cmd = Twist()
        cmd.linear.x = 0.5
        self.cmd_pub.publish(cmd)
        self.speak("Moving forward")

    def move_backward(self):
        cmd = Twist()
        cmd.linear.x = -0.5
        self.cmd_pub.publish(cmd)
        self.speak("Moving backward")

    def turn_left(self):
        cmd = Twist()
        cmd.angular.z = 0.5
        self.cmd_pub.publish(cmd)
        self.speak("Turning left")

    def turn_right(self):
        cmd = Twist()
        cmd.angular.z = -0.5
        self.cmd_pub.publish(cmd)
        self.speak("Turning right")

    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0
        cmd.angular.z = 0
        self.cmd_pub.publish(cmd)
        self.speak("Stopping")

    def speak(self, text):
        """Text-to-speech output"""
        print(f"Robot says: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def run(self):
        """Main control loop"""
        self.speak("Robot is ready for commands")

        while not rospy.is_shutdown():
            # Listen for command
            command = self.recognize_speech()

            if command:
                # Get LLM interpretation
                llm_response = self.generate_response(command)
                print(f"LLM response: {llm_response}")

                # Parse and execute action
                action = self.parse_llm_response(llm_response)

                if action and action in self.actions:
                    self.actions[action]()
                else:
                    self.speak("I didn't understand that command")

            rospy.sleep(1)

def main():
    controller = LLMRobotController()
    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
```

## Best Practices and Troubleshooting

### Common Issues and Solutions

1. **Simulation Instability**
   - Reduce time step in physics engine
   - Check mass and inertia values
   - Ensure proper joint limits

2. **Sensor Noise**
   - Apply filtering to sensor data
   - Use multiple sensor fusion
   - Implement outlier detection

3. **Control Oscillation**
   - Tune PID parameters
   - Add damping to control system
   - Implement smooth trajectory generation

### Development Tips

1. **Start Simple**
   - Begin with basic movements before complex behaviors
   - Test in simulation before real hardware
   - Implement one feature at a time

2. **Modular Design**
   - Separate perception, planning, and control
   - Use ROS nodes for different functions
   - Make components reusable

3. **Safety First**
   - Implement emergency stop mechanisms
   - Use simulation extensively
   - Test thoroughly before deployment

### Exercise: Build Your Own Embodied AI System

Now that you've completed the tutorials, try building your own embodied AI system:

1. **Design Challenge**: Create a robot that can navigate to a target while avoiding obstacles
2. **Implementation**: Use the techniques learned to implement perception, planning, and control
3. **AI Integration**: Add machine learning for improved navigation
4. **Evaluation**: Test in simulation and measure performance

These tutorials provide a foundation for developing embodied AI systems. As you progress, consider more complex challenges like multi-robot coordination, learning from human demonstrations, or integrating advanced AI models for perception and decision-making.