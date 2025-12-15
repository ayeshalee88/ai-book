#!/usr/bin/env python3
"""
AI Integration Example for Physical AI & Humanoid Robotics
This script demonstrates the integration of computer vision and reinforcement learning
for robotic manipulation tasks.
"""

import numpy as np
import rospy
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time

class VisionProcessor:
    """Handles computer vision processing for the robot"""

    def __init__(self):
        self.bridge = CvBridge()
        self.object_positions = {}

    def process_image(self, image_msg):
        """
        Process image to detect objects and return their positions
        """
        try:
            # Convert ROS image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

            # Convert BGR to HSV for color-based object detection
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # Define range for red color (example object)
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])

            # Create mask for red objects
            mask = cv2.inRange(hsv, lower_red, upper_red)

            # Find contours of red objects
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            object_positions = []
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Minimum area threshold
                    # Calculate center of contour
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        object_positions.append((cX, cY))

            return object_positions

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
            return []

class Actor(nn.Module):
    """Actor network for reinforcement learning"""

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.max_action
        return a

class Critic(nn.Module):
    """Critic network for reinforcement learning"""

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

class TD3Agent:
    """TD3 Reinforcement Learning Agent"""

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
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.policy_freq = 2
        self.total_it = 0

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).cpu().data.numpy().flatten()

        if add_noise:
            noise = np.random.normal(0, 0.1 * self.policy_noise, size=action.shape)
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def train(self, batch_size=100):
        if len(self.replay_buffer) < batch_size:
            return

        self.total_it += 1

        batch = random.sample(self.replay_buffer, batch_size)
        state, action, next_state, reward, done = map(np.stack, zip(*batch))

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        done = torch.BoolTensor(done).unsqueeze(1)

        # Select next action
        noise = torch.FloatTensor(action).data.normal_(0, self.policy_noise)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)

        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

        # Compute target Q-value
        target_Q1 = self.critic_1_target(next_state, next_action)
        target_Q2 = self.critic_2_target(next_state, next_action)
        target_Q = reward + done.float() * self.gamma * torch.min(target_Q1, target_Q2)

        # Get current Q estimates
        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        # Optimize critics
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic_1(state, self.actor(state)).mean()

            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, state, action, next_state, reward, done):
        self.replay_buffer.append((state, action, next_state, reward, done))

class AIVisionRobot:
    """Main class that integrates AI and vision for robotic control"""

    def __init__(self):
        rospy.init_node('ai_vision_robot', anonymous=True)

        # Initialize vision processor
        self.vision = VisionProcessor()

        # Initialize RL agent
        self.state_dim = 6  # Example: [object_x, object_y, robot_x, robot_y, velocity_x, velocity_y]
        self.action_dim = 2  # Example: [velocity_x, velocity_y]
        self.max_action = 1.0
        self.rl_agent = TD3Agent(self.state_dim, self.action_dim, self.max_action)

        # ROS publishers and subscribers
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        self.joint_state_sub = rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Internal state
        self.current_object_positions = []
        self.robot_position = [0.0, 0.0]
        self.robot_velocity = [0.0, 0.0]
        self.latest_image = None
        self.latest_joint_state = None

        # Control loop rate
        self.rate = rospy.Rate(10)  # 10 Hz

        print("AI Vision Robot initialized")

    def image_callback(self, data):
        """Callback for image data"""
        self.current_object_positions = self.vision.process_image(data)
        self.latest_image = data

    def joint_state_callback(self, data):
        """Callback for joint state data"""
        self.latest_joint_state = data
        # Extract position and velocity information
        # This is a simplified example
        if len(data.position) >= 2:
            self.robot_position = [data.position[0], data.position[1]]
        if len(data.velocity) >= 2:
            self.robot_velocity = [data.velocity[0], data.velocity[1]]

    def get_state(self):
        """Get current state for RL agent"""
        # Create state vector
        if self.current_object_positions:
            # Use first detected object as target
            obj_x, obj_y = self.current_object_positions[0]
            # Normalize image coordinates to [0, 1] range
            obj_x_norm = obj_x / 640.0  # Assuming 640x480 image
            obj_y_norm = obj_y / 480.0
        else:
            # No object detected
            obj_x_norm, obj_y_norm = 0.5, 0.5  # Center of image

        # Robot position and velocity (normalized)
        robot_x_norm = np.clip(self.robot_position[0] / 5.0, -1.0, 1.0)  # Assuming workspace of 5m
        robot_y_norm = np.clip(self.robot_position[1] / 5.0, -1.0, 1.0)
        vel_x_norm = np.clip(self.robot_velocity[0] / 1.0, -1.0, 1.0)  # Assuming max velocity of 1m/s
        vel_y_norm = np.clip(self.robot_velocity[1] / 1.0, -1.0, 1.0)

        state = np.array([
            obj_x_norm, obj_y_norm,  # Object position (normalized)
            robot_x_norm, robot_y_norm,  # Robot position (normalized)
            vel_x_norm, vel_y_norm   # Robot velocity (normalized)
        ])

        return state

    def calculate_reward(self, action):
        """Calculate reward based on current state and action"""
        reward = 0.0

        if self.current_object_positions:
            # Get first detected object
            obj_x, obj_y = self.current_object_positions[0]

            # Reward for moving toward object
            # In a real system, this would involve more complex calculations
            # based on distance to object, alignment, etc.
            target_x_norm = obj_x / 640.0
            target_y_norm = obj_y / 480.0

            # Simple reward based on how well action aligns with moving toward target
            robot_to_target_x = target_x_norm - self.robot_position[0]/5.0
            robot_to_target_y = target_y_norm - self.robot_position[1]/5.0

            # Dot product of action direction and direction to target
            alignment = (action[0] * robot_to_target_x + action[1] * robot_to_target_y)
            reward += max(0, alignment) * 10.0  # Positive reward for moving toward target

        # Penalty for excessive action magnitude
        action_penalty = -np.sum(np.abs(action)) * 0.1
        reward += action_penalty

        # Small survival bonus
        reward += 0.1

        return reward

    def execute_action(self, action):
        """Execute the action by publishing to robot"""
        cmd_vel = Twist()

        # Scale action to appropriate velocity ranges
        cmd_vel.linear.x = action[0] * 0.5  # Max 0.5 m/s
        cmd_vel.linear.y = action[1] * 0.5  # Max 0.5 m/s
        # Add other components as needed

        self.cmd_vel_pub.publish(cmd_vel)

    def run_control_loop(self):
        """Main control loop"""
        print("Starting AI Vision Robot control loop...")

        prev_state = self.get_state()

        while not rospy.is_shutdown():
            # Get current state
            current_state = self.get_state()

            # Get action from RL agent
            action = self.rl_agent.select_action(current_state)

            # Execute action
            self.execute_action(action)

            # Calculate reward
            reward = self.calculate_reward(action)

            # Determine if episode is done (simplified)
            done = False  # In a real implementation, this would check for termination conditions

            # Store transition in replay buffer
            self.rl_agent.store_transition(prev_state, action, current_state, reward, done)

            # Train RL agent
            self.rl_agent.train()

            # Update previous state
            prev_state = current_state

            # Sleep to maintain control rate
            self.rate.sleep()

def main():
    """Main function to run the AI Vision Robot"""
    try:
        robot = AIVisionRobot()
        robot.run_control_loop()
    except rospy.ROSInterruptException:
        print("ROS Interrupt received. Shutting down...")
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down...")

if __name__ == '__main__':
    main()