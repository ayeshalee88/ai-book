---
id: hardware-integration
title: Hardware Integration for Physical AI Systems
sidebar_label: Hardware Integration
---

# Hardware Integration for Physical AI Systems

## Introduction to Hardware Integration

Hardware integration is a critical aspect of physical AI systems, requiring careful consideration of mechanical design, electronics, sensors, actuators, and software interfaces. This tutorial covers the practical aspects of integrating various hardware components into functional robotic systems.

### Key Challenges in Hardware Integration

Hardware integration for physical AI systems presents several unique challenges:

1. **Mechanical Integration**: Ensuring components fit together and function as intended
2. **Electrical Integration**: Managing power distribution, signal integrity, and communication
3. **Software Integration**: Creating interfaces between hardware and control algorithms
4. **Real-time Performance**: Meeting timing constraints for responsive behavior
5. **Safety Considerations**: Ensuring safe operation around humans and environments
6. **Reliability**: Maintaining consistent performance over extended operation

## Mechanical Integration

### Design for Assembly and Maintenance

When designing mechanical systems, consider assembly and maintenance requirements:

```python
# Example: Modular robot design with standardized interfaces
class ModularRobotDesign:
    """
    A framework for designing modular robots with standardized interfaces
    """
    def __init__(self):
        self.modules = []
        self.interfaces = []
        self.connection_points = []

    def add_module(self, module_type, position, orientation):
        """
        Add a module (e.g., actuator, sensor, link) at specified position
        """
        module = {
            'type': module_type,
            'position': position,
            'orientation': orientation,
            'connections': [],
            'mass': self.get_module_mass(module_type),
            'inertia': self.get_module_inertia(module_type)
        }
        self.modules.append(module)
        return module

    def add_standardized_interface(self, module1, module2, interface_type='bolt_pattern'):
        """
        Add a standardized interface between two modules
        """
        interface = {
            'module1': module1,
            'module2': module2,
            'type': interface_type,
            'specification': self.get_interface_spec(interface_type),
            'strength': self.calculate_connection_strength(interface_type)
        }
        self.interfaces.append(interface)
        return interface

    def get_module_mass(self, module_type):
        """Get typical mass for module type"""
        mass_table = {
            'servo_motor': 0.1,  # kg
            'dc_motor': 0.5,
            'imu_sensor': 0.05,
            'camera': 0.2,
            'link_aluminum': 0.3,
            'link_carbon_fiber': 0.1
        }
        return mass_table.get(module_type, 0.1)

    def get_module_inertia(self, module_type):
        """Get typical inertia for module type"""
        # Simplified - in reality this would be 3x3 tensor
        inertia_table = {
            'servo_motor': [0.001, 0.001, 0.002],
            'dc_motor': [0.01, 0.01, 0.02],
            'imu_sensor': [0.0001, 0.0001, 0.0001],
            'camera': [0.001, 0.002, 0.001],
            'link_aluminum': [0.01, 0.01, 0.001],
            'link_carbon_fiber': [0.005, 0.005, 0.0005]
        }
        return inertia_table.get(module_type, [0.001, 0.001, 0.001])

    def calculate_connection_strength(self, interface_type):
        """Calculate load capacity of connection interface"""
        strength_table = {
            'bolt_pattern': 500,  # N
            'quick_disconnect': 200,
            'snap_fit': 50,
            'screw_mount': 100
        }
        return strength_table.get(interface_type, 100)

    def check_mechanical_feasibility(self):
        """
        Check if the mechanical design is feasible
        """
        total_mass = sum(module['mass'] for module in self.modules)
        connection_loads = self.calculate_connection_loads()

        # Check if connections can handle loads
        for interface in self.interfaces:
            if connection_loads[interface['type']] > interface['strength']:
                return False, f"Connection {interface['type']} exceeds strength limits"

        return True, f"Design is mechanically feasible (total mass: {total_mass:.2f} kg)"

    def calculate_connection_loads(self):
        """Calculate expected loads on connections"""
        # Simplified load calculation
        return {'bolt_pattern': 300, 'quick_disconnect': 150}

# Example usage
robot_design = ModularRobotDesign()
torso = robot_design.add_module('link_carbon_fiber', [0, 0, 0], [0, 0, 0])
head = robot_design.add_module('camera', [0, 0, 0.1], [0, 0, 0])
robot_design.add_standardized_interface(torso, head, 'bolt_pattern')

is_feasible, message = robot_design.check_mechanical_feasibility()
print(message)
```

### Material Selection and Properties

Material selection significantly impacts robot performance:

```python
class MaterialSelector:
    """
    Select appropriate materials based on requirements
    """
    def __init__(self):
        self.material_properties = {
            'aluminum_6061': {
                'density': 2700,  # kg/m³
                'young_modulus': 69e9,  # Pa
                'yield_strength': 276e6,  # Pa
                'cost': 2.5,  # $/kg
                'machinability': 'good',
                'corrosion_resistance': 'good'
            },
            'carbon_fiber': {
                'density': 1700,
                'young_modulus': 230e9,
                'yield_strength': 1000e6,
                'cost': 50.0,
                'machinability': 'poor',
                'corrosion_resistance': 'excellent'
            },
            'steel_4140': {
                'density': 7850,
                'young_modulus': 200e9,
                'yield_strength': 415e6,
                'cost': 5.0,
                'machinability': 'good',
                'corrosion_resistance': 'poor'
            },
            'plastic_abs': {
                'density': 1040,
                'young_modulus': 2.5e9,
                'yield_strength': 40e6,
                'cost': 3.0,
                'machinability': 'excellent',
                'corrosion_resistance': 'good'
            }
        }

    def select_material(self, requirements):
        """
        Select material based on requirements

        requirements: dict with keys like 'max_weight', 'min_strength', 'max_cost'
        """
        candidates = []

        for material, props in self.material_properties.items():
            if (requirements.get('max_density', float('inf')) >= props['density'] and
                requirements.get('min_strength', 0) <= props['yield_strength'] and
                requirements.get('max_cost', float('inf')) >= props['cost']):

                # Calculate score based on requirements
                score = 0
                if 'weight_priority' in requirements:
                    score += (1 / props['density']) * requirements.get('weight_priority', 1)
                if 'strength_priority' in requirements:
                    score += (props['yield_strength'] / 1e6) * requirements.get('strength_priority', 1)

                candidates.append((material, score))

        # Return highest scoring material
        if candidates:
            return max(candidates, key=lambda x: x[1])[0]
        else:
            return None

# Example usage
selector = MaterialSelector()
requirements = {
    'max_density': 3000,
    'min_strength': 200e6,
    'max_cost': 20.0,
    'weight_priority': 2,
    'strength_priority': 1
}
material = selector.select_material(requirements)
print(f"Recommended material: {material}")
```

## Electrical Integration

### Power Distribution and Management

Proper power distribution is critical for reliable operation:

```python
class PowerDistributionSystem:
    """
    Design and analyze power distribution for robot
    """
    def __init__(self, battery_voltage=24.0, battery_capacity=50.0):  # V, Ah
        self.battery_voltage = battery_voltage
        self.battery_capacity = battery_capacity
        self.components = []
        self.wiring = []

    def add_component(self, name, voltage, current_draw, duty_cycle=1.0):
        """
        Add electrical component to the system
        """
        component = {
            'name': name,
            'voltage': voltage,  # V
            'current_draw': current_draw,  # A
            'duty_cycle': duty_cycle,  # Fraction of time active
            'power_draw': voltage * current_draw * duty_cycle,  # W
            'power_supply': self.select_power_supply(voltage)
        }
        self.components.append(component)
        return component

    def select_power_supply(self, required_voltage):
        """Select appropriate power supply method"""
        if required_voltage == self.battery_voltage:
            return "direct_battery"
        elif required_voltage < self.battery_voltage:
            return "buck_converter"
        else:
            return "boost_converter"

    def calculate_total_power_consumption(self):
        """Calculate total power consumption"""
        total_power = sum(comp['power_draw'] for comp in self.components)
        return total_power

    def calculate_battery_life(self):
        """Calculate estimated battery life"""
        total_current = 0
        for comp in self.components:
            total_current += comp['current_draw'] * comp['duty_cycle']

        battery_life_hours = (self.battery_capacity * self.battery_voltage) / \
                            (self.calculate_total_power_consumption())
        return battery_life_hours

    def design_wiring_harness(self):
        """Design wiring harness based on power requirements"""
        wiring_harness = []

        # Group components by voltage requirements
        voltage_groups = {}
        for comp in self.components:
            voltage = comp['voltage']
            if voltage not in voltage_groups:
                voltage_groups[voltage] = []
            voltage_groups[voltage].append(comp)

        # Design wire gauge for each group
        for voltage, components in voltage_groups.items():
            total_current = sum(comp['current_draw'] * comp['duty_cycle'] for comp in components)

            # Calculate required wire gauge (simplified)
            # Using rule of thumb: 1 mm² per 10A for safety
            required_area = max(0.5, total_current / 10.0)  # mm²
            wire_gauge = self.area_to_awg(required_area)

            harness = {
                'voltage': voltage,
                'components': [comp['name'] for comp in components],
                'wire_gauge': wire_gauge,
                'current_rating': total_current
            }
            wiring_harness.append(harness)

        return wiring_harness

    def area_to_awg(self, area_mm2):
        """Convert wire cross-sectional area to AWG gauge"""
        # Simplified conversion
        awg_lookup = {
            0.05: 24, 0.08: 22, 0.13: 20, 0.20: 18,
            0.32: 16, 0.51: 14, 0.81: 12, 1.31: 10,
            2.08: 8, 3.31: 6, 5.26: 4, 8.37: 2
        }

        for area, awg in sorted(awg_lookup.items(), reverse=True):
            if area_mm2 >= area:
                return awg
        return 24  # Default to smallest wire

    def check_power_budget(self):
        """Verify power consumption is within battery limits"""
        total_power = self.calculate_total_power_consumption()
        max_available_power = self.battery_voltage * (self.battery_capacity / 1)  # 1 hour discharge

        if total_power > max_available_power:
            return False, f"Power consumption ({total_power:.1f}W) exceeds available power ({max_available_power:.1f}W)"
        else:
            return True, f"Power budget OK (available: {max_available_power:.1f}W, consuming: {total_power:.1f}W)"

# Example: Design power system for a small robot
power_system = PowerDistributionSystem(battery_voltage=24.0, battery_capacity=20.0)

# Add components
power_system.add_component("main_controller", 5.0, 2.0, 1.0)
power_system.add_component("motors", 24.0, 10.0, 0.3)  # 30% duty cycle
power_system.add_component("sensors", 12.0, 1.0, 1.0)
power_system.add_component("communication", 5.0, 0.5, 1.0)

# Check power budget
is_ok, message = power_system.check_power_budget()
print(message)

# Design wiring harness
wiring = power_system.design_wiring_harness()
for harness in wiring:
    print(f"Voltage {harness['voltage']}V: {harness['wire_gauge']} AWG wire for {harness['components']}")

# Calculate battery life
battery_life = power_system.calculate_battery_life()
print(f"Estimated battery life: {battery_life:.2f} hours")
```

### Communication Protocols and Interfaces

Robots typically use multiple communication protocols:

```python
class CommunicationSystem:
    """
    Design communication system for robot
    """
    def __init__(self):
        self.interfaces = []
        self.protocols = {
            'i2c': {'max_devices': 128, 'speed': 400000, 'distance': 1},  # bps, meters
            'spi': {'max_devices': 1, 'speed': 10000000, 'distance': 0.5},
            'uart': {'max_devices': 2, 'speed': 115200, 'distance': 15},
            'can': {'max_devices': 110, 'speed': 1000000, 'distance': 40},
            'ethernet': {'max_devices': 1000, 'speed': 100000000, 'distance': 100}
        }

    def add_interface(self, protocol, device_count, data_rate, distance):
        """
        Add communication interface
        """
        if protocol not in self.protocols:
            raise ValueError(f"Unknown protocol: {protocol}")

        spec = self.protocols[protocol]

        if device_count > spec['max_devices']:
            raise ValueError(f"Too many devices for {protocol} (max: {spec['max_devices']})")

        if data_rate > spec['speed']:
            raise ValueError(f"Data rate too high for {protocol} (max: {spec['speed']} bps)")

        if distance > spec['distance']:
            raise ValueError(f"Distance too long for {protocol} (max: {spec['distance']} m)")

        interface = {
            'protocol': protocol,
            'device_count': device_count,
            'data_rate': data_rate,
            'distance': distance,
            'spec': spec
        }
        self.interfaces.append(interface)
        return interface

    def optimize_communication_topology(self):
        """
        Optimize communication based on requirements
        """
        # Group devices by communication needs
        high_speed_devices = []
        medium_speed_devices = []
        low_speed_devices = []

        for interface in self.interfaces:
            if interface['data_rate'] > 1000000:  # > 1 Mbps
                high_speed_devices.append(interface)
            elif interface['data_rate'] > 100000:  # > 100 kbps
                medium_speed_devices.append(interface)
            else:
                low_speed_devices.append(interface)

        # Assign protocols based on needs
        topology = {
            'high_speed': 'ethernet' if high_speed_devices else None,
            'medium_speed': 'spi' if medium_speed_devices else 'uart',
            'low_speed': 'i2c' if low_speed_devices else 'can'
        }

        return topology

    def generate_device_addressing_scheme(self):
        """
        Generate addressing scheme for devices
        """
        addressing_scheme = {}

        # For I2C devices
        i2c_devices = [iface for iface in self.interfaces if iface['protocol'] == 'i2c']
        if i2c_devices:
            # Typical I2C addresses (simplified)
            base_address = 0x40
            for i in range(i2c_devices[0]['device_count']):
                addressing_scheme[f"i2c_device_{i}"] = hex(base_address + i)

        # For CAN devices
        can_devices = [iface for iface in self.interfaces if iface['protocol'] == 'can']
        if can_devices:
            for i in range(can_devices[0]['device_count']):
                addressing_scheme[f"can_node_{i}"] = i + 1  # CAN uses node IDs 1-110

        return addressing_scheme

# Example: Design communication for a sensor-laden robot
comm_system = CommunicationSystem()

# Add various interfaces
comm_system.add_interface('i2c', 5, 100000, 0.5)  # IMU, pressure sensors
comm_system.add_interface('spi', 1, 5000000, 0.2)  # High-speed ADC
comm_system.add_interface('uart', 2, 115200, 2.0)  # GPS, XBee
comm_system.add_interface('can', 8, 500000, 10.0)  # Motor controllers

# Optimize topology
topology = comm_system.optimize_communication_topology()
print("Communication topology:", topology)

# Generate addressing
addressing = comm_system.generate_device_addressing_scheme()
print("Device addressing:", addressing)
```

## Sensor Integration

### IMU Integration and Calibration

Inertial Measurement Units (IMUs) are crucial for balance and orientation:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class IMUIntegrator:
    """
    Integrate IMU data for orientation estimation
    """
    def __init__(self, sample_rate=100):
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.orientation = R.from_quat([0, 0, 0, 1])  # Initial orientation (identity)
        self.gyro_bias = np.zeros(3)  # Gyro bias
        self.accel_bias = np.zeros(3)  # Accelerometer bias
        self.calibration_samples = []

    def calibrate_sensor(self, accel_data, gyro_data, num_samples=1000):
        """
        Calibrate IMU by collecting samples in static position
        """
        print("Starting IMU calibration...")

        # Collect calibration samples
        for i in range(num_samples):
            # In real implementation, read from sensor
            # For simulation, we'll use the provided data
            if i < len(accel_data) and i < len(gyro_data):
                self.calibration_samples.append({
                    'accel': accel_data[i],
                    'gyro': gyro_data[i]
                })

        # Calculate biases (assuming robot is stationary)
        avg_accel = np.mean([s['accel'] for s in self.calibration_samples], axis=0)
        avg_gyro = np.mean([s['gyro'] for s in self.calibration_samples], axis=0)

        # Accelerometer should read [0, 0, 9.81] when level
        self.accel_bias = avg_accel - np.array([0, 0, 9.81])
        self.gyro_bias = avg_gyro  # Gyro should read [0, 0, 0] when stationary

        print(f"Calibration complete. Accel bias: {self.accel_bias}, Gyro bias: {self.gyro_bias}")

    def update_orientation(self, raw_accel, raw_gyro):
        """
        Update orientation estimate using accelerometer and gyroscope
        """
        # Apply calibration
        accel = np.array(raw_accel) - self.accel_bias
        gyro = np.array(raw_gyro) - self.gyro_bias

        # Normalize accelerometer (for gravity reference)
        accel_norm = accel / np.linalg.norm(accel)

        # Integrate gyroscope for orientation change
        angle_change = gyro * self.dt
        rotation_vector = R.from_rotvec(angle_change)

        # Update orientation
        self.orientation = self.orientation * rotation_vector

        # Use accelerometer to correct drift (simplified gravity alignment)
        current_gravity = self.orientation.apply([0, 0, 1])  # Robot's perceived gravity
        gravity_error = accel_norm - current_gravity

        # Apply correction (simple proportional control)
        correction_gain = 0.01
        correction_vector = np.cross(current_gravity, accel_norm)
        correction_rotation = R.from_rotvec(correction_gain * correction_vector)
        self.orientation = correction_rotation * self.orientation

        return self.orientation.as_quat()

    def get_euler_angles(self):
        """Get orientation as Euler angles (roll, pitch, yaw)"""
        return self.orientation.as_euler('xyz', degrees=True)

class MultiIMUManager:
    """
    Manage multiple IMUs for redundancy and accuracy
    """
    def __init__(self, num_imus=3):
        self.imus = [IMUIntegrator() for _ in range(num_imus)]
        self.weights = np.ones(num_imus) / num_imus  # Equal weights initially
        self.fault_detection_threshold = 0.1

    def calibrate_all_imus(self, accel_data, gyro_data):
        """Calibrate all IMUs"""
        for imu in self.imus:
            imu.calibrate_sensor(accel_data, gyro_data)

    def update_all_imus(self, all_accel_data, all_gyro_data):
        """
        Update all IMUs and combine readings with fault detection
        """
        orientations = []

        for i, (accel, gyro) in enumerate(zip(all_accel_data, all_gyro_data)):
            quat = self.imus[i].update_orientation(accel, gyro)
            orientations.append(np.array(quat))

        # Detect potential sensor faults by comparing readings
        self.detect_and_handle_faults(orientations)

        # Weighted average of orientations
        weighted_quat = np.average(orientations, axis=0, weights=self.weights)
        # Normalize the quaternion
        weighted_quat = weighted_quat / np.linalg.norm(weighted_quat)

        return weighted_quat

    def detect_and_handle_faults(self, orientations):
        """
        Detect sensor faults and adjust weights accordingly
        """
        if len(orientations) < 2:
            return

        # Calculate differences between all IMU readings
        for i in range(len(orientations)):
            total_diff = 0
            for j in range(len(orientations)):
                if i != j:
                    # Calculate angular difference between quaternions
                    q1, q2 = orientations[i], orientations[j]
                    # Convert to rotation vectors for comparison
                    r1 = R.from_quat(q1).as_rotvec()
                    r2 = R.from_quat(q2).as_rotvec()
                    diff = np.linalg.norm(r1 - r2)
                    total_diff += diff

            # If difference is too large, reduce weight
            avg_diff = total_diff / (len(orientations) - 1)
            if avg_diff > self.fault_detection_threshold:
                self.weights[i] *= 0.5  # Reduce weight of potentially faulty sensor
                print(f"Potential fault detected on IMU {i}, reducing weight")
            else:
                # Gradually restore weight if consistent
                self.weights[i] = min(1.0, self.weights[i] * 1.01)

# Example usage
imu_manager = MultiIMUManager(num_imus=3)

# Simulated calibration data (robot stationary)
calib_accel = [[0.1, -0.05, 9.85] for _ in range(100)]
calib_gyro = [[0.01, -0.02, 0.005] for _ in range(100)]

imu_manager.calibrate_all_imus(calib_accel, calib_gyro)

# Simulated real-time data
real_accel = [[0.05, 0.1, 9.7], [0.08, -0.02, 9.82], [0.04, 0.05, 9.78]]
real_gyro = [[0.1, 0.05, -0.02], [0.12, 0.04, -0.01], [0.09, 0.06, -0.03]]

combined_orientation = imu_manager.update_all_imus(real_accel, real_gyro)
print(f"Combined orientation: {combined_orientation}")
```

### Camera Integration

Camera systems require careful integration for computer vision applications:

```python
import cv2
import numpy as np

class CameraSystem:
    """
    Integrate and manage camera systems
    """
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.resolution = (640, 480)
        self.fps = 30

    def initialize_camera(self):
        """Initialize camera with settings"""
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")

        # Set resolution and FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        print(f"Camera initialized: {self.resolution[0]}x{self.resolution[1]} @ {self.fps} FPS")

    def calibrate_camera(self, calibration_images):
        """
        Calibrate camera using checkerboard images
        """
        # Prepare object points (3D points in real world)
        objp = np.zeros((6*9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        objpoints = []  # 3D points in real world
        imgpoints = []  # 2D points in image

        for img in calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        if len(objpoints) > 0:
            ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = \
                cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

            print(f"Camera calibration successful: {ret}")
            print(f"Camera matrix:\n{self.camera_matrix}")
            print(f"Distortion coefficients: {self.dist_coeffs.flatten()}")
        else:
            print("Camera calibration failed - no valid images found")

    def get_frame(self):
        """Get a frame from the camera"""
        if self.cap is None:
            self.initialize_camera()

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")

        return frame

    def undistort_image(self, img):
        """Remove lens distortion from image"""
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            h, w = img.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
            )

            undistorted = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)

            # Crop image based on ROI
            x, y, w, h = roi
            undistorted = undistorted[y:y+h, x:x+w]
            return undistorted
        else:
            return img  # Return original if not calibrated

    def get_camera_parameters(self):
        """Get camera intrinsic parameters"""
        return {
            'camera_matrix': self.camera_matrix,
            'distortion_coefficients': self.dist_coeffs,
            'resolution': self.resolution,
            'focal_length': self.camera_matrix[0, 0] if self.camera_matrix is not None else None
        }

class StereoCameraSystem:
    """
    Integrate stereo camera system for depth perception
    """
    def __init__(self, left_camera_id=0, right_camera_id=1):
        self.left_camera = CameraSystem(left_camera_id)
        self.right_camera = CameraSystem(right_camera_id)
        self.stereo_calib_data = None
        self.rectification_maps = None

    def calibrate_stereo_system(self, left_images, right_images):
        """
        Calibrate stereo camera system
        """
        # This would implement stereo calibration using corresponding points
        # For simplicity, we'll simulate the process
        print("Performing stereo calibration...")

        # Calibrate individual cameras first
        self.left_camera.calibrate_camera(left_images)
        self.right_camera.calibrate_camera(right_images)

        # In real implementation, you would:
        # 1. Find corresponding points in both images
        # 2. Use cv2.stereoCalibrate() to get rotation and translation
        # 3. Use cv2.stereoRectify() to get rectification parameters

        # Simulated stereo parameters
        self.stereo_calib_data = {
            'rotation': np.eye(3),  # Identity (cameras parallel)
            'translation': np.array([-0.1, 0, 0]),  # 10cm baseline
            'essential_matrix': np.array([[0, 0, 0], [0, 0, -0.1], [0, 0.1, 0]]),
            'fundamental_matrix': np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
        }

        print("Stereo calibration complete")

    def compute_depth_map(self, left_img, right_img):
        """
        Compute depth map from stereo pair
        """
        # Create stereo matcher
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16*10,  # Must be divisible by 16
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Compute disparity
        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

        # Convert disparity to depth (simplified)
        # Depth = (baseline * focal_length) / disparity
        if self.left_camera.camera_matrix is not None:
            baseline = 0.1  # 10cm baseline
            focal_length = self.left_camera.camera_matrix[0, 0]
            depth_map = (baseline * focal_length) / (disparity + 1e-6)  # Add small value to avoid division by zero
        else:
            depth_map = disparity  # Use disparity as proxy if no calibration

        return depth_map

    def get_stereo_frame(self):
        """Get synchronized stereo frame pair"""
        left_frame = self.left_camera.get_frame()
        right_frame = self.right_camera.get_frame()
        return left_frame, right_frame

# Example usage
def camera_example():
    # Initialize single camera
    camera = CameraSystem(0)
    camera.initialize_camera()

    # Get frame and process
    frame = camera.get_frame()
    undistorted = camera.undistort_image(frame)

    # For stereo, you would do:
    # stereo_system = StereoCameraSystem(0, 1)  # Left and right camera IDs
    # stereo_system.calibrate_stereo_system(left_calib_images, right_calib_images)
    # left_img, right_img = stereo_system.get_stereo_frame()
    # depth_map = stereo_system.compute_depth_map(left_img, right_img)

    print("Camera system initialized and ready")
    camera.cap.release()

# Run example (commented out to avoid camera access issues)
# camera_example()
```

## Actuator Integration

### Servo Motor Control

Servo motors are common in robotics for precise position control:

```python
import time
import threading

class ServoController:
    """
    Control servo motors with position, speed, and torque control
    """
    def __init__(self, pwm_frequency=50):  # 50Hz for standard servos
        self.pwm_frequency = pwm_frequency
        self.servos = {}
        self.running = False
        self.control_thread = None

    def add_servo(self, servo_id, pin, min_pulse=500, max_pulse=2500,
                  min_angle=-90, max_angle=90):
        """
        Add a servo motor to the controller
        """
        self.servos[servo_id] = {
            'pin': pin,
            'min_pulse': min_pulse,  # microseconds
            'max_pulse': max_pulse,
            'min_angle': min_angle,
            'max_angle': max_angle,
            'current_angle': 0,
            'target_angle': 0,
            'speed_limit': None,  # degrees per second
            'torque_limit': None,  # percentage (0-100)
            'enabled': True
        }

        # Initialize servo to center position
        self.set_angle(servo_id, 0)
        print(f"Added servo {servo_id} on pin {pin}")

    def angle_to_pulse(self, servo_id, angle):
        """Convert angle to PWM pulse width"""
        servo = self.servos[servo_id]

        # Clamp angle to valid range
        angle = max(servo['min_angle'], min(servo['max_angle'], angle))

        # Convert angle to pulse width
        pulse_range = servo['max_pulse'] - servo['min_pulse']
        angle_range = servo['max_angle'] - servo['min_angle']

        pulse = servo['min_pulse'] + (angle - servo['min_angle']) * \
                (pulse_range / angle_range)

        return int(pulse)

    def set_angle(self, servo_id, angle, blocking=False):
        """Set servo to specific angle"""
        if servo_id not in self.servos:
            raise ValueError(f"Servo {servo_id} not found")

        servo = self.servos[servo_id]
        servo['target_angle'] = max(servo['min_angle'], min(servo['max_angle'], angle))

        if blocking:
            # Wait until servo reaches target (simplified)
            while abs(servo['current_angle'] - servo['target_angle']) > 0.5:
                time.sleep(0.01)

    def set_speed_limit(self, servo_id, speed_dps):
        """Set maximum speed for servo movement (degrees per second)"""
        if servo_id in self.servos:
            self.servos[servo_id]['speed_limit'] = speed_dps

    def set_torque_limit(self, servo_id, torque_percent):
        """Set torque limit as percentage"""
        if servo_id in self.servos:
            self.servos[servo_id]['torque_limit'] = max(0, min(100, torque_percent))

    def enable_servo(self, servo_id, enabled=True):
        """Enable or disable servo"""
        if servo_id in self.servos:
            self.servos[servo_id]['enabled'] = enabled

    def get_position(self, servo_id):
        """Get current servo position"""
        if servo_id in self.servos:
            return self.servos[servo_id]['current_angle']
        return None

    def start_control_loop(self):
        """Start the control loop in a separate thread"""
        if self.running:
            return

        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

    def stop_control_loop(self):
        """Stop the control loop"""
        self.running = False
        if self.control_thread:
            self.control_thread.join()

    def _control_loop(self):
        """Internal control loop"""
        while self.running:
            current_time = time.time()

            for servo_id, servo in self.servos.items():
                if not servo['enabled']:
                    continue

                # Calculate how much to move based on speed limit
                if servo['speed_limit']:
                    max_move = servo['speed_limit'] * 0.02  # 20ms time step
                    angle_diff = servo['target_angle'] - servo['current_angle']

                    if abs(angle_diff) > max_move:
                        # Move by maximum allowed amount
                        direction = 1 if angle_diff > 0 else -1
                        servo['current_angle'] += direction * max_move
                    else:
                        servo['current_angle'] = servo['target_angle']
                else:
                    # Move directly to target
                    servo['current_angle'] = servo['target_angle']

                # Convert to PWM and send (in real implementation)
                pulse_width = self.angle_to_pulse(servo_id, servo['current_angle'])
                # self._send_pwm_signal(servo['pin'], pulse_width)

            time.sleep(0.02)  # 50Hz update rate

    def _send_pwm_signal(self, pin, pulse_width):
        """Send PWM signal to servo (implementation depends on hardware)"""
        # This would interface with actual hardware
        # For example: GPIO on Raspberry Pi, PCA9685 servo driver, etc.
        pass

class RobotArmController:
    """
    Control a multi-joint robot arm using servo controllers
    """
    def __init__(self):
        self.servo_controller = ServoController()
        self.joints = {}
        self.dh_parameters = {}  # Denavit-Hartenberg parameters

    def add_joint(self, joint_name, servo_id, pin, joint_type='revolute',
                  dh_params=None):
        """
        Add a joint to the robot arm
        dh_params: [a, alpha, d, theta_offset] for DH parameters
        """
        self.servo_controller.add_servo(servo_id, pin)
        self.joints[joint_name] = {
            'servo_id': servo_id,
            'type': joint_type,
            'dh_params': dh_params or [0, 0, 0, 0]  # Default DH params
        }

    def move_to_joint_positions(self, joint_positions, duration=2.0, blocking=True):
        """
        Move all joints to specified positions over duration
        joint_positions: dict like {'joint1': angle1, 'joint2': angle2, ...}
        """
        # Calculate intermediate positions for smooth motion
        start_positions = {name: self.servo_controller.get_position(
            self.joints[name]['servo_id']) for name in joint_positions.keys()}

        num_steps = int(duration * 50)  # 50Hz control rate
        for step in range(num_steps + 1):
            t = step / num_steps  # Normalized time [0, 1]

            # Interpolate to target position
            for joint_name, target_pos in joint_positions.items():
                start_pos = start_positions[joint_name]
                current_pos = start_pos + t * (target_pos - start_pos)

                servo_id = self.joints[joint_name]['servo_id']
                self.servo_controller.set_angle(servo_id, current_pos)

            time.sleep(0.02)  # 50Hz rate

    def move_to_cartesian(self, target_position, target_orientation=None):
        """
        Move end effector to Cartesian position using inverse kinematics
        """
        # This would implement inverse kinematics
        # For a simple 2-DOF arm:
        x, y, z = target_position

        # Calculate joint angles (simplified 2D example)
        r = (x**2 + y**2)**0.5
        # ... more complex IK calculations

        # For now, just a placeholder
        print(f"Moving to Cartesian position: {target_position}")

    def get_joint_positions(self):
        """Get current joint positions"""
        positions = {}
        for joint_name, joint_info in self.joints.items():
            pos = self.servo_controller.get_position(joint_info['servo_id'])
            positions[joint_name] = pos
        return positions

# Example usage
def servo_example():
    arm = RobotArmController()

    # Add joints (simplified example)
    arm.add_joint('shoulder', servo_id=1, pin=11, dh_params=[0.1, 0, 0, 0])
    arm.add_joint('elbow', servo_id=2, pin=12, dh_params=[0.1, 0, 0, 0])
    arm.add_joint('wrist', servo_id=3, pin=13, dh_params=[0.05, 0, 0, 0])

    # Start servo control
    arm.servo_controller.start_control_loop()

    # Move to positions
    positions = {'shoulder': 30, 'elbow': -45, 'wrist': 15}
    arm.move_to_joint_positions(positions, duration=2.0)

    print("Current joint positions:", arm.get_joint_positions())

    # Stop control
    arm.servo_controller.stop_control_loop()

# Run example (commented out to avoid hardware dependencies)
# servo_example()
```

### Motor Control with Feedback

DC motors with encoders provide velocity and position feedback:

```python
class DCMotorController:
    """
    Control DC motors with encoder feedback
    """
    def __init__(self, motor_id, encoder_resolution=1000):
        self.motor_id = motor_id
        self.encoder_resolution = encoder_resolution  # counts per revolution
        self.current_position = 0
        self.current_velocity = 0
        self.target_position = 0
        self.target_velocity = 0
        self.max_pwm = 255
        self.pwm_output = 0

        # PID controller parameters
        self.pid = {
            'position': {'kp': 10.0, 'ki': 0.1, 'kd': 0.5},
            'velocity': {'kp': 2.0, 'ki': 0.05, 'kd': 0.1}
        }

        self.integral_errors = {'position': 0, 'velocity': 0}
        self.previous_errors = {'position': 0, 'velocity': 0}
        self.control_rate = 100  # Hz
        self.dt = 1.0 / self.control_rate

    def update_encoder(self, encoder_count):
        """Update position based on encoder count"""
        # Calculate position in radians
        self.current_position = (encoder_count / self.encoder_resolution) * 2 * 3.14159

        # Calculate velocity (simplified)
        # In real implementation, this would use time-based differentiation

    def pid_control(self, error, mode='position'):
        """Apply PID control"""
        pid_params = self.pid[mode]

        # Update integral
        self.integral_errors[mode] += error * self.dt
        # Apply integral windup protection
        self.integral_errors[mode] = max(-10, min(10, self.integral_errors[mode]))

        # Calculate derivative
        derivative = (error - self.previous_errors[mode]) / self.dt

        # Apply PID formula
        output = (pid_params['kp'] * error +
                 pid_params['ki'] * self.integral_errors[mode] +
                 pid_params['kd'] * derivative)

        # Store current error for next derivative calculation
        self.previous_errors[mode] = error

        return output

    def control_position(self, target_position):
        """Control motor position using PID"""
        position_error = target_position - self.current_position
        pwm_output = self.pid_control(position_error, 'position')

        # Limit PWM output
        self.pwm_output = max(-self.max_pwm, min(self.max_pwm, int(pwm_output)))

        # Send PWM command (in real implementation)
        # self._send_pwm_command(self.pwm_output)

        return self.pwm_output

    def control_velocity(self, target_velocity):
        """Control motor velocity using PID"""
        velocity_error = target_velocity - self.current_velocity
        pwm_output = self.pid_control(velocity_error, 'velocity')

        # Limit PWM output
        self.pwm_output = max(-self.max_pwm, min(self.max_pwm, int(pwm_output)))

        # Send PWM command (in real implementation)
        # self._send_pwm_command(self.pwm_output)

        return self.pwm_output

    def enable_motor(self, enabled=True):
        """Enable or disable motor"""
        # In real implementation, this would control H-bridge enable pin
        pass

class MotorControlSystem:
    """
    Manage multiple DC motors with coordinated control
    """
    def __init__(self):
        self.motors = {}
        self.control_thread = None
        self.running = False

    def add_motor(self, motor_name, motor_id, encoder_resolution=1000):
        """Add a DC motor to the system"""
        self.motors[motor_name] = DCMotorController(motor_id, encoder_resolution)
        print(f"Added motor {motor_name} with ID {motor_id}")

    def start_control(self):
        """Start the control loop"""
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

    def stop_control(self):
        """Stop the control loop"""
        self.running = False
        if self.control_thread:
            self.control_thread.join()

    def _control_loop(self):
        """Main control loop"""
        import time

        while self.running:
            start_time = time.time()

            # Update all motors
            for motor_name, motor in self.motors.items():
                # For position control, call motor.control_position(target_pos)
                # For velocity control, call motor.control_velocity(target_vel)
                pass

            # Maintain control rate
            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0 / 100) - elapsed)  # 100Hz rate
            if sleep_time > 0:
                time.sleep(sleep_time)

    def move_motors_sync(self, motor_positions, duration=2.0):
        """
        Move multiple motors synchronously to target positions
        """
        start_positions = {name: motor.current_position
                          for name, motor in self.motors.items()}

        num_steps = int(duration * self.motors[list(self.motors.keys())[0]].control_rate)

        for step in range(num_steps + 1):
            t = step / num_steps  # Normalized time [0, 1]

            for motor_name, target_pos in motor_positions.items():
                if motor_name in self.motors:
                    start_pos = start_positions[motor_name]
                    current_target = start_pos + t * (target_pos - start_pos)
                    self.motors[motor_name].control_position(current_target)

            time.sleep(1.0 / 100)  # 100Hz rate

# Example usage
def motor_example():
    motor_system = MotorControlSystem()

    # Add motors
    motor_system.add_motor('left_wheel', 1, encoder_resolution=2000)
    motor_system.add_motor('right_wheel', 2, encoder_resolution=2000)

    # Start control system
    motor_system.start_control()

    # Move motors to positions
    positions = {'left_wheel': 1.57, 'right_wheel': 1.57}  # 90 degrees each
    motor_system.move_motors_sync(positions, duration=2.0)

    # Stop control
    motor_system.stop_control()

# Run example (commented out to avoid hardware dependencies)
# motor_example()
```

## Software Integration

### Real-time Control Considerations

Real-time performance is crucial for physical AI systems:

```python
import time
import threading
import queue
from collections import deque

class RealTimeController:
    """
    Real-time control system with timing guarantees
    """
    def __init__(self, control_frequency=1000):  # Hz
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        self.running = False
        self.control_thread = None
        self.sensors = {}
        self.actuators = {}
        self.control_law = None
        self.timing_stats = deque(maxlen=100)  # Keep last 100 timing samples

    def add_sensor(self, name, callback, frequency=None):
        """Add sensor with optional specific frequency"""
        self.sensors[name] = {
            'callback': callback,
            'frequency': frequency or self.control_frequency,
            'last_update': 0,
            'data': None
        }

    def add_actuator(self, name, callback):
        """Add actuator"""
        self.actuators[name] = {
            'callback': callback,
            'last_command': None
        }

    def set_control_law(self, control_function):
        """Set the control law function"""
        self.control_law = control_function

    def start(self):
        """Start real-time control loop"""
        if self.running:
            return

        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

    def stop(self):
        """Stop real-time control loop"""
        self.running = False
        if self.control_thread:
            self.control_thread.join()

    def _control_loop(self):
        """Main real-time control loop"""
        next_update = time.time()

        while self.running:
            loop_start = time.time()

            # Read sensors
            sensor_data = {}
            for name, sensor in self.sensors.items():
                if loop_start - sensor['last_update'] >= 1.0 / sensor['frequency']:
                    sensor['data'] = sensor['callback']()
                    sensor['last_update'] = loop_start
                sensor_data[name] = sensor['data']

            # Execute control law
            if self.control_law:
                commands = self.control_law(sensor_data)
            else:
                commands = {}

            # Send commands to actuators
            for name, command in commands.items():
                if name in self.actuators:
                    self.actuators[name]['callback'](command)
                    self.actuators[name]['last_command'] = command

            # Calculate timing
            loop_end = time.time()
            loop_time = loop_end - loop_start
            self.timing_stats.append({
                'loop_time': loop_time,
                'deadline_miss': loop_time > self.dt,
                'slack_time': self.dt - loop_time
            })

            # Sleep to maintain frequency
            next_update += self.dt
            sleep_time = next_update - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Deadline miss - log for analysis
                print(f"Deadline miss: {abs(sleep_time)*1000:.2f}ms late")

    def get_timing_analysis(self):
        """Get timing performance analysis"""
        if not self.timing_stats:
            return "No timing data available"

        loop_times = [s['loop_time'] for s in self.timing_stats]
        avg_time = sum(loop_times) / len(loop_times)
        max_time = max(loop_times)
        deadline_misses = sum(1 for s in self.timing_stats if s['deadline_miss'])

        return {
            'average_loop_time_ms': avg_time * 1000,
            'max_loop_time_ms': max_time * 1000,
            'deadline_misses': deadline_misses,
            'miss_rate_percent': (deadline_misses / len(self.timing_stats)) * 100
        }

class SensorEmulator:
    """Emulate sensor data for testing"""
    def __init__(self, base_value=0, noise_level=0.01):
        self.base_value = base_value
        self.noise_level = noise_level
        self.time = 0

    def read(self):
        """Simulate sensor reading with noise"""
        import random
        self.time += 0.001  # Simulate time passing
        return self.base_value + random.gauss(0, self.noise_level) + 0.1 * math.sin(2 * math.pi * 2 * self.time)

class ActuatorEmulator:
    """Emulate actuator for testing"""
    def __init__(self):
        self.current_value = 0

    def command(self, value):
        """Simulate sending command to actuator"""
        self.current_value = value
        # In real implementation, this would send signal to hardware

# Example usage
def real_time_example():
    import math

    controller = RealTimeController(control_frequency=100)  # 100 Hz

    # Add emulated sensors
    position_sensor = SensorEmulator(base_value=0, noise_level=0.01)
    velocity_sensor = SensorEmulator(base_value=0, noise_level=0.005)

    controller.add_sensor('position', position_sensor.read, frequency=100)
    controller.add_sensor('velocity', velocity_sensor.read, frequency=100)

    # Add emulated actuators
    motor = ActuatorEmulator()
    controller.add_actuator('motor', motor.command)

    # Define simple PD controller
    def simple_pd_control(sensor_data):
        position = sensor_data['position'] or 0
        velocity = sensor_data['velocity'] or 0

        # Simple PD controller to maintain position at 0
        kp = 10.0
        kd = 2.0
        target_position = 0

        error = target_position - position
        control_output = kp * error - kd * velocity

        return {'motor': max(-100, min(100, control_output))}  # Limit to ±100

    controller.set_control_law(simple_pd_control)

    # Start control
    controller.start()

    # Run for 5 seconds
    time.sleep(5)

    # Get timing analysis
    timing_info = controller.get_timing_analysis()
    print("Timing Analysis:", timing_info)

    # Stop control
    controller.stop()

# Run example
# real_time_example()
```

## Safety and Fault Handling

### Safety System Implementation

Safety is paramount in physical AI systems:

```python
class SafetySystem:
    """
    Comprehensive safety system for physical AI
    """
    def __init__(self):
        self.emergency_stop = False
        self.faults = {}
        self.safety_limits = {}
        self.protection_systems = []
        self.shutdown_sequence = []
        self.monitoring_enabled = True

    def add_current_monitor(self, motor_name, max_current):
        """Monitor motor current for overcurrent protection"""
        self.safety_limits[f'{motor_name}_current'] = max_current

    def add_temperature_monitor(self, component_name, max_temp):
        """Monitor component temperature"""
        self.safety_limits[f'{component_name}_temp'] = max_temp

    def add_position_limit(self, joint_name, min_pos, max_pos):
        """Set position limits for joints"""
        self.safety_limits[f'{joint_name}_min'] = min_pos
        self.safety_limits[f'{joint_name}_max'] = max_pos

    def add_velocity_limit(self, component_name, max_vel):
        """Set velocity limits"""
        self.safety_limits[f'{component_name}_vel'] = max_vel

    def check_safety(self, sensor_data):
        """
        Check all safety conditions
        Returns: (is_safe, list_of_violations)
        """
        violations = []

        # Check current limits
        for key, value in sensor_data.items():
            if key.endswith('_current'):
                motor_name = key.replace('_current', '')
                limit_key = f'{motor_name}_current'
                if limit_key in self.safety_limits:
                    if value > self.safety_limits[limit_key]:
                        violations.append(f"Overcurrent on {motor_name}: {value}A > {self.safety_limits[limit_key]}A")

        # Check temperature limits
        for key, value in sensor_data.items():
            if key.endswith('_temp'):
                component_name = key.replace('_temp', '')
                limit_key = f'{component_name}_temp'
                if limit_key in self.safety_limits:
                    if value > self.safety_limits[limit_key]:
                        violations.append(f"Overtemperature on {component_name}: {value}°C > {self.safety_limits[limit_key]}°C")

        # Check position limits
        for key, value in sensor_data.items():
            if key.endswith('_position'):
                joint_name = key.replace('_position', '')
                min_key = f'{joint_name}_min'
                max_key = f'{joint_name}_max'
                if min_key in self.safety_limits and max_key in self.safety_limits:
                    if value < self.safety_limits[min_key] or value > self.safety_limits[max_key]:
                        violations.append(f"Position limit violation on {joint_name}: {value} rad")

        # Check velocity limits
        for key, value in sensor_data.items():
            if key.endswith('_velocity'):
                component_name = key.replace('_velocity', '')
                limit_key = f'{component_name}_vel'
                if limit_key in self.safety_limits:
                    if abs(value) > self.safety_limits[limit_key]:
                        violations.append(f"Velocity limit violation on {component_name}: {abs(value)} rad/s > {self.safety_limits[limit_key]} rad/s")

        return len(violations) == 0, violations

    def emergency_stop(self):
        """Activate emergency stop"""
        print("EMERGENCY STOP ACTIVATED!")
        self.emergency_stop = True
        # In real implementation, this would:
        # - Cut power to all motors
        # - Apply brakes if available
        # - Execute shutdown sequence

    def fault_detected(self, fault_type, severity='medium'):
        """Handle fault detection"""
        self.faults[fault_type] = {
            'time': time.time(),
            'severity': severity,
            'handled': False
        }

        if severity == 'high':
            self.emergency_stop()
        elif severity == 'medium':
            print(f"Medium severity fault detected: {fault_type}")
        else:
            print(f"Low severity fault detected: {fault_type}")

    def get_safety_status(self):
        """Get current safety status"""
        return {
            'emergency_stop': self.emergency_stop,
            'faults': self.faults,
            'safety_limits': self.safety_limits
        }

class HardwareSafetyManager:
    """
    Manage safety for hardware components
    """
    def __init__(self):
        self.safety_system = SafetySystem()
        self.watchdog_timers = {}
        self.hardware_interfaces = []

    def add_watchdog(self, name, timeout=2.0):
        """Add watchdog timer for critical components"""
        self.watchdog_timers[name] = {
            'timeout': timeout,
            'last_update': time.time(),
            'enabled': True
        }

    def feed_watchdog(self, name):
        """Reset watchdog timer"""
        if name in self.watchdog_timers:
            self.watchdog_timers[name]['last_update'] = time.time()

    def check_watchdogs(self):
        """Check if any watchdogs have timed out"""
        current_time = time.time()
        timeouts = []

        for name, timer in self.watchdog_timers.items():
            if timer['enabled'] and (current_time - timer['last_update']) > timer['timeout']:
                timeouts.append(name)
                print(f"Watchdog timeout: {name}")
                self.safety_system.fault_detected(f'watchdog_timeout_{name}', 'high')

        return timeouts

    def monitor_system(self, sensor_data):
        """
        Monitor entire system for safety violations
        """
        # Check safety conditions
        is_safe, violations = self.safety_system.check_safety(sensor_data)

        if not is_safe:
            for violation in violations:
                print(f"Safety violation: {violation}")
                self.safety_system.fault_detected(violation, 'medium')

        # Check watchdogs
        self.check_watchdogs()

        return is_safe

# Example usage
def safety_example():
    safety_manager = HardwareSafetyManager()

    # Set up safety limits
    safety_manager.safety_system.add_current_monitor('left_motor', 10.0)  # 10A limit
    safety_manager.safety_system.add_current_monitor('right_motor', 10.0)
    safety_manager.safety_system.add_temperature_monitor('controller', 80)  # 80°C limit
    safety_manager.safety_system.add_position_limit('hip_joint', -1.57, 1.57)  # ±90°
    safety_manager.safety_system.add_velocity_limit('wheel', 5.0)  # 5 rad/s limit

    # Add watchdog for critical systems
    safety_manager.add_watchdog('motion_control', timeout=1.0)
    safety_manager.add_watchdog('safety_monitor', timeout=0.5)

    # Simulate sensor data
    sensor_data = {
        'left_motor_current': 8.5,
        'right_motor_current': 9.2,
        'controller_temp': 65,
        'hip_joint_position': 0.5,
        'wheel_velocity': 3.2
    }

    # Monitor system
    is_safe = safety_manager.monitor_system(sensor_data)
    print(f"System safety status: {'SAFE' if is_safe else 'UNSAFE'}")

    # Feed watchdogs
    safety_manager.feed_watchdog('motion_control')
    safety_manager.feed_watchdog('safety_monitor')

    print("Safety system operational")

# Run example
safety_example()
```

## Troubleshooting and Debugging

### Common Hardware Integration Issues

```python
class HardwareDiagnostics:
    """
    Diagnostic tools for hardware integration issues
    """
    def __init__(self):
        self.diagnostics = []
        self.known_issues = {
            'communication_timeout': {
                'symptoms': ['No response from device', 'Intermittent data'],
                'causes': ['Loose connections', 'Wrong baud rate', 'Noise interference'],
                'solutions': ['Check connections', 'Verify baud rate', 'Use shielded cables']
            },
            'power_instability': {
                'symptoms': ['Random resets', 'Inconsistent behavior', 'Overheating'],
                'causes': ['Insufficient power supply', 'Voltage drops', 'Poor grounding'],
                'solutions': ['Check power ratings', 'Add capacitors', 'Verify ground connections']
            },
            'calibration_drift': {
                'symptoms': ['Inaccurate readings', 'Drifting values', 'Poor performance'],
                'causes': ['Temperature changes', 'Mechanical stress', 'Component aging'],
                'solutions': ['Re-calibrate', 'Temperature compensation', 'Replace components']
            }
        }

    def run_power_diagnosis(self, voltage_readings, current_readings):
        """Diagnose power-related issues"""
        issues = []

        # Check for voltage drops
        for i, voltage in enumerate(voltage_readings):
            if voltage < 0.9 * self.get_expected_voltage():
                issues.append(f"Voltage drop detected at reading {i}: {voltage}V")

        # Check for current spikes
        avg_current = sum(current_readings) / len(current_readings)
        for i, current in enumerate(current_readings):
            if current > 1.5 * avg_current:
                issues.append(f"Current spike detected at reading {i}: {current}A")

        return issues

    def run_communication_diagnosis(self, device_responses, expected_responses):
        """Diagnose communication issues"""
        issues = []

        for device, response in device_responses.items():
            expected = expected_responses.get(device)
            if response != expected:
                issues.append(f"Communication error with {device}: got {response}, expected {expected}")

        return issues

    def get_expected_voltage(self):
        """Get expected system voltage"""
        # This would be configured based on system design
        return 12.0  # Example: 12V system

    def generate_diagnostic_report(self, test_results):
        """Generate comprehensive diagnostic report"""
        report = {
            'timestamp': time.time(),
            'system_health': 'good',
            'issues_found': [],
            'recommendations': [],
            'test_results': test_results
        }

        # Analyze results and add to report
        for test_name, results in test_results.items():
            if results.get('status') == 'fail':
                report['issues_found'].append({
                    'test': test_name,
                    'details': results.get('details', ''),
                    'severity': results.get('severity', 'medium')
                })

        # Determine overall health
        if report['issues_found']:
            report['system_health'] = 'degraded'
            if any(issue['severity'] == 'high' for issue in report['issues_found']):
                report['system_health'] = 'critical'

        return report

# Example diagnostic usage
def run_hardware_diagnostics():
    diagnostics = HardwareDiagnostics()

    # Simulate test results
    test_results = {
        'power_test': {
            'status': 'pass',
            'details': 'All voltages within acceptable range',
            'severity': 'low'
        },
        'communication_test': {
            'status': 'fail',
            'details': 'Device at address 0x42 not responding',
            'severity': 'high'
        },
        'sensor_test': {
            'status': 'pass',
            'details': 'All sensors responding correctly',
            'severity': 'low'
        }
    }

    report = diagnostics.generate_diagnostic_report(test_results)
    print("Diagnostic Report:")
    print(f"System Health: {report['system_health']}")
    print(f"Issues Found: {len(report['issues_found'])}")
    for issue in report['issues_found']:
        print(f"  - {issue['test']}: {issue['details']} (Severity: {issue['severity']})")

# Run diagnostic example
run_hardware_diagnostics()
```

## Learning Objectives

After completing this tutorial, you should be able to:

1. Design mechanical systems with proper integration considerations
2. Implement power distribution systems for robotic applications
3. Integrate various sensors (IMU, cameras, encoders) with proper calibration
4. Control different types of actuators (servos, DC motors) with feedback
5. Implement real-time control systems with timing guarantees
6. Design safety systems and fault handling mechanisms
7. Troubleshoot common hardware integration issues

## Hands-On Exercise

Design and implement a complete hardware integration for a simple robot:

1. Design the mechanical structure using appropriate materials
2. Create a power distribution system for your robot's components
3. Integrate at least 3 different types of sensors (IMU, camera, encoders)
4. Implement control for 2-3 actuators with feedback
5. Create a safety system with emergency stops and fault detection
6. Test your integration with a simple task (e.g., move to a position, avoid obstacles)
7. Document any issues encountered and how you resolved them

This exercise will give you practical experience with the challenges and solutions involved in physical AI hardware integration.