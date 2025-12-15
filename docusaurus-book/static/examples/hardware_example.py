#!/usr/bin/env python3
"""
Complete hardware integration example for Physical AI & Humanoid Robotics
This example demonstrates how to integrate various hardware components
"""

import time
import threading
import queue
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import math

# Simulated hardware interfaces (in real implementation, these would interface with actual hardware)
class SimulatedIMU:
    """Simulated IMU sensor"""
    def __init__(self, noise_level=0.01):
        self.noise_level = noise_level
        self.time = time.time()
        self.orientation = [0, 0, 0, 1]  # [x, y, z, w] quaternion
        self.angular_velocity = [0, 0, 0]
        self.linear_acceleration = [0, 0, 9.81]  # Gravity

    def read(self):
        """Read sensor data"""
        current_time = time.time()
        dt = current_time - self.time
        self.time = current_time

        # Simulate small changes in orientation
        self.angular_velocity = [
            0.1 * np.random.normal(0, self.noise_level),
            0.05 * np.random.normal(0, self.noise_level),
            0.02 * np.random.normal(0, self.noise_level)
        ]

        # Update orientation (simplified integration)
        angle_change = [w * dt for w in self.angular_velocity]
        magnitude = np.linalg.norm(angle_change)
        if magnitude > 0:
            axis = [v/magnitude for v in angle_change]
            # Convert axis-angle to quaternion (simplified)
            s = np.sin(magnitude/2)
            self.orientation = [
                axis[0] * s,
                axis[1] * s,
                axis[2] * s,
                np.cos(magnitude/2)
            ]

        # Add noise to acceleration
        self.linear_acceleration = [
            0.1 * np.random.normal(0, self.noise_level),
            0.05 * np.random.normal(0, self.noise_level),
            9.81 + 0.1 * np.random.normal(0, self.noise_level)
        ]

        return {
            'orientation': self.orientation.copy(),
            'angular_velocity': self.angular_velocity.copy(),
            'linear_acceleration': self.linear_acceleration.copy(),
            'timestamp': current_time
        }

class SimulatedServo:
    """Simulated servo motor"""
    def __init__(self, joint_name, min_angle=-90, max_angle=90, noise_level=0.1):
        self.joint_name = joint_name
        self.min_angle = math.radians(min_angle)
        self.max_angle = math.radians(max_angle)
        self.noise_level = noise_level
        self.target_position = 0
        self.current_position = 0
        self.velocity = 0
        self.max_velocity = math.radians(180)  # 180 deg/s max
        self.is_enabled = True

    def set_position(self, position_radians):
        """Set target position in radians"""
        self.target_position = max(self.min_angle, min(self.max_angle, position_radians))

    def get_position(self):
        """Get current position with noise"""
        return self.current_position + np.random.normal(0, self.noise_level)

    def update(self, dt):
        """Update servo position based on target"""
        if not self.is_enabled:
            return

        # Simple first-order dynamics
        position_error = self.target_position - self.current_position
        max_move = self.max_velocity * dt
        actual_move = max(-max_move, min(max_move, position_error))
        self.current_position += actual_move
        self.velocity = actual_move / dt

class SimulatedEncoder:
    """Simulated rotary encoder"""
    def __init__(self, resolution=1000, gear_ratio=1.0):
        self.resolution = resolution
        self.gear_ratio = gear_ratio
        self.position = 0  # radians
        self.velocity = 0  # rad/s
        self.count = 0
        self.last_update = time.time()

    def update_position(self, position_radians):
        """Update encoder position"""
        self.position = position_radians
        self.count = int(position_radians * self.gear_ratio * self.resolution / (2 * math.pi))

    def read(self):
        """Read encoder data"""
        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time

        # Calculate velocity
        if dt > 0:
            self.velocity = (self.position - self.position) / dt  # This is simplified

        return {
            'position_radians': self.position,
            'position_counts': self.count,
            'velocity_rps': self.velocity,
            'timestamp': current_time
        }

class SafetySystem:
    """Safety system for hardware integration"""
    def __init__(self):
        self.emergency_stop = False
        self.safety_limits = {}
        self.faults = []

    def add_limit(self, component, parameter, min_val, max_val):
        """Add safety limit for a component parameter"""
        self.safety_limits[f"{component}_{parameter}"] = (min_val, max_val)

    def check_safety(self, sensor_data):
        """Check if sensor data is within safety limits"""
        if self.emergency_stop:
            return False, ["Emergency stop activated"]

        violations = []
        for key, value in sensor_data.items():
            if key in self.safety_limits:
                min_val, max_val = self.safety_limits[key]
                if not (min_val <= value <= max_val):
                    violations.append(f"{key} violation: {value} not in [{min_val}, {max_val}]")

        return len(violations) == 0, violations

    def activate_emergency_stop(self):
        """Activate emergency stop"""
        self.emergency_stop = True
        print("EMERGENCY STOP ACTIVATED!")

class RobotHardwareInterface:
    """Main hardware interface class"""
    def __init__(self):
        self.imu = SimulatedIMU()
        self.servos = {}
        self.encoders = {}
        self.safety = SafetySystem()
        self.running = False
        self.control_thread = None
        self.data_queue = queue.Queue()
        self.command_queue = queue.Queue()

        # Initialize hardware components
        self.initialize_hardware()

    def initialize_hardware(self):
        """Initialize all hardware components"""
        # Add servos for different joints
        self.servos['hip_pitch_left'] = SimulatedServo('left_hip_pitch', -45, 90)
        self.servos['hip_pitch_right'] = SimulatedServo('right_hip_pitch', -45, 90)
        self.servos['knee_left'] = SimulatedServo('left_knee', -150, 0)
        self.servos['knee_right'] = SimulatedServo('right_knee', -150, 0)
        self.servos['ankle_left'] = SimulatedServo('left_ankle', -30, 30)
        self.servos['ankle_right'] = SimulatedServo('right_ankle', -30, 30)

        # Add encoders
        self.encoders['hip_left'] = SimulatedEncoder(resolution=2000)
        self.encoders['hip_right'] = SimulatedEncoder(resolution=2000)
        self.encoders['knee_left'] = SimulatedEncoder(resolution=2000)
        self.encoders['knee_right'] = SimulatedEncoder(resolution=2000)

        # Set up safety limits
        self.safety.add_limit('imu', 'linear_acceleration_z', 5.0, 15.0)
        self.safety.add_limit('servo', 'hip_pitch_left', math.radians(-45), math.radians(90))
        self.safety.add_limit('servo', 'hip_pitch_right', math.radians(-45), math.radians(90))

    def start(self):
        """Start the hardware interface"""
        if self.running:
            return

        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

    def stop(self):
        """Stop the hardware interface"""
        self.running = False
        if self.control_thread:
            self.control_thread.join()

    def _control_loop(self):
        """Main control loop"""
        last_time = time.time()

        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # Read all sensors
            sensor_data = self.read_sensors()

            # Check safety
            is_safe, violations = self.safety.check_safety({
                'imu_linear_acceleration_z': sensor_data['imu']['linear_acceleration'][2]
            })

            if not is_safe:
                print(f"Safety violations: {violations}")
                self.safety.activate_emergency_stop()
                continue

            # Process any pending commands
            while not self.command_queue.empty():
                try:
                    cmd = self.command_queue.get_nowait()
                    self._execute_command(cmd)
                except queue.Empty:
                    break

            # Update servos
            for servo in self.servos.values():
                servo.update(dt)

            # Update encoders based on servo positions
            for joint_name, servo in self.servos.items():
                if joint_name in self.encoders:
                    self.encoders[joint_name.replace('pitch', '').replace('knee', '').replace('ankle', '')].update_position(
                        servo.get_position()
                    )

            # Put sensor data in queue for external access
            self.data_queue.put({
                'timestamp': current_time,
                'sensor_data': sensor_data,
                'dt': dt
            })

            # Control loop rate (100Hz)
            time.sleep(max(0, 0.01 - (time.time() - current_time)))

    def read_sensors(self):
        """Read all sensors"""
        return {
            'imu': self.imu.read(),
            'encoders': {name: enc.read() for name, enc in self.encoders.items()}
        }

    def _execute_command(self, command):
        """Execute a hardware command"""
        if command['type'] == 'set_servo_position':
            joint = command['joint']
            position = command['position']
            if joint in self.servos:
                self.servos[joint].set_position(position)
        elif command['type'] == 'set_servo_positions':
            positions = command['positions']
            for joint, pos in positions.items():
                if joint in self.servos:
                    self.servos[joint].set_position(pos)

    def set_servo_position(self, joint_name, position_radians):
        """Set servo position (thread-safe)"""
        cmd = {
            'type': 'set_servo_position',
            'joint': joint_name,
            'position': position_radians
        }
        self.command_queue.put(cmd)

    def set_servo_positions(self, positions_dict):
        """Set multiple servo positions (thread-safe)"""
        cmd = {
            'type': 'set_servo_positions',
            'positions': positions_dict
        }
        self.command_queue.put(cmd)

    def get_sensor_data(self, timeout=0.1):
        """Get sensor data (non-blocking)"""
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None

class WalkingController:
    """Simple walking controller using the hardware interface"""
    def __init__(self, hardware_interface):
        self.hw = hardware_interface
        self.time = 0
        self.step_phase = 0
        self.step_frequency = 1.0  # steps per second
        self.step_height = 0.05    # meters
        self.step_length = 0.1     # meters

    def update(self, dt):
        """Update walking controller"""
        self.time += dt
        self.step_phase += dt * self.step_frequency * 2 * math.pi

        # Simple walking pattern
        left_swing = math.sin(self.step_phase) > 0
        right_swing = not left_swing

        # Calculate joint angles for walking
        hip_amplitude = 0.2
        knee_amplitude = 0.3
        ankle_amplitude = 0.1

        left_hip = hip_amplitude * math.sin(self.step_phase) if left_swing else 0
        right_hip = hip_amplitude * math.sin(self.step_phase + math.pi) if right_swing else 0

        left_knee = knee_amplitude * math.sin(self.step_phase * 2) if left_swing else 0
        right_knee = knee_amplitude * math.sin(self.step_phase * 2 + math.pi) if right_swing else 0

        left_ankle = -left_hip if left_swing else 0
        right_ankle = -right_hip if right_swing else 0

        # Send commands to servos
        positions = {
            'hip_pitch_left': left_hip,
            'hip_pitch_right': right_hip,
            'knee_left': left_knee,
            'knee_right': right_knee,
            'ankle_left': left_ankle,
            'ankle_right': right_ankle
        }

        self.hw.set_servo_positions(positions)

def main():
    """Main function demonstrating hardware integration"""
    print("Initializing robot hardware interface...")

    # Create hardware interface
    hw_interface = RobotHardwareInterface()

    # Start the interface
    hw_interface.start()

    # Create walking controller
    walker = WalkingController(hw_interface)

    print("Hardware interface started, beginning walking demonstration...")
    print("Reading sensors and controlling servos at 100Hz")
    print("Press Ctrl+C to stop")

    start_time = time.time()
    last_sensor_time = start_time

    try:
        while True:
            current_time = time.time()
            dt = current_time - (start_time if 'last_time' not in locals() else last_time)
            last_time = current_time

            # Update walking controller
            walker.update(dt)

            # Read sensor data periodically
            if current_time - last_sensor_time > 0.5:  # Every 0.5 seconds
                sensor_data = hw_interface.get_sensor_data()
                if sensor_data:
                    imu_data = sensor_data['sensor_data']['imu']
                    print(f"Time: {current_time - start_time:.1f}s")
                    print(f"  IMU Accel Z: {imu_data['linear_acceleration'][2]:.2f}")
                    print(f"  IMU Orientation: [{imu_data['orientation'][0]:.2f}, {imu_data['orientation'][1]:.2f}, {imu_data['orientation'][2]:.2f}, {imu_data['orientation'][3]:.2f}]")
                    print(f"  Left Hip Position: {hw_interface.servos['hip_pitch_left'].get_position():.2f} rad")
                    print(f"  Right Hip Position: {hw_interface.servos['hip_pitch_right'].get_position():.2f} rad")

                last_sensor_time = current_time

            # Run for 30 seconds then stop
            if current_time - start_time > 30:
                print("Demo time limit reached")
                break

            time.sleep(0.01)  # Small delay to prevent excessive CPU usage

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        print("Stopping hardware interface...")
        hw_interface.stop()
        print("Hardware interface stopped")

if __name__ == "__main__":
    main()