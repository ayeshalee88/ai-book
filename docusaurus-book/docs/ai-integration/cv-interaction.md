---
id: cv-interaction
title: Computer Vision for Physical AI Interaction
sidebar_label: CV for Interaction
---

# Computer Vision for Physical AI Interaction

## Introduction to Vision-Based Physical AI

Computer vision is a critical component of physical AI systems, enabling robots to perceive and understand their environment for effective interaction. Unlike traditional computer vision applications focused on image classification or object detection, vision for physical AI must provide real-time, actionable information that guides robotic behavior and enables safe, effective interaction with the physical world.

### The Vision-Action Loop

Physical AI systems close the loop between vision and action:

```
Image Capture → Visual Processing → Action Decision → Physical Action → Environment Change → New Image Capture
```

This continuous cycle requires:
- **Real-time processing**: Fast inference for responsive behavior
- **Robust perception**: Reliable operation under varying conditions
- **Actionable output**: Visual information that directly informs control
- **Sensor fusion**: Integration with other sensory modalities

### Key Requirements for Physical AI Vision

#### Real-Time Performance
- **High frame rates**: 30-60 FPS for responsive interaction
- **Low latency**: <50ms from capture to action for safety
- **Efficient algorithms**: Optimized for embedded platforms

#### Robustness
- **Illumination invariance**: Operation under varying lighting
- **Viewpoint invariance**: Recognition from different angles
- **Occlusion handling**: Robustness to partial object visibility
- **Motion blur tolerance**: Recognition during movement

#### Actionability
- **Metric measurements**: Real-world dimensions and positions
- **Uncertainty quantification**: Confidence in visual estimates
- **Temporal consistency**: Stable tracking across frames

## Object Detection and Recognition

### Deep Learning Approaches

#### YOLO (You Only Look Once) for Real-Time Detection

YOLO is particularly well-suited for robotic applications due to its speed:

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class YOLODetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        # Load pre-trained YOLO model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.eval()
        self.confidence_threshold = confidence_threshold

    def detect_objects(self, image):
        """
        Detect objects in image and return bounding boxes, classes, and confidence scores
        """
        # Convert PIL image to tensor
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Perform inference
        results = self.model(image)

        # Extract detections
        detections = results.pandas().xyxy[0]  # Convert to pandas DataFrame

        # Filter by confidence
        confident_detections = detections[detections['confidence'] > self.confidence_threshold]

        objects = []
        for _, detection in confident_detections.iterrows():
            obj = {
                'class': detection['name'],
                'confidence': detection['confidence'],
                'bbox': [detection['xmin'], detection['ymin'],
                         detection['xmax'], detection['ymax']],
                'center': [(detection['xmin'] + detection['xmax']) / 2,
                           (detection['ymin'] + detection['ymax']) / 2]
            }
            objects.append(obj)

        return objects

    def get_object_pose(self, object_name, image):
        """
        Get the 2D position of an object in the image
        """
        detections = self.detect_objects(image)

        for obj in detections:
            if obj['class'] == object_name:
                # Convert 2D image coordinates to robot frame coordinates
                pixel_x, pixel_y = obj['center']
                # Apply camera calibration to get metric coordinates
                metric_coords = self.pixel_to_metric(pixel_x, pixel_y)
                return metric_coords, obj['confidence']

        return None, 0.0

    def pixel_to_metric(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates to metric coordinates using camera calibration
        """
        # This would use actual camera calibration parameters
        # For example: using intrinsic matrix and known object size
        return [pixel_x * 0.001, pixel_y * 0.001]  # Simplified conversion
```

#### Feature-Based Detection

For specific objects or markers, feature-based methods can be more robust:

```python
import cv2
import numpy as np

class FeatureDetector:
    def __init__(self, reference_image_path):
        self.reference_image = cv2.imread(reference_image_path, 0)
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()

        # Compute reference keypoints and descriptors
        self.ref_kp, self.ref_desc = self.sift.detectAndCompute(self.reference_image, None)

    def detect_object(self, query_image):
        """
        Detect object using SIFT features
        """
        query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        query_kp, query_desc = self.sift.detectAndCompute(query_gray, None)

        if query_desc is None or self.ref_desc is None:
            return None

        # Match features
        matches = self.bf.knnMatch(self.ref_desc, query_desc, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m and n and m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Require minimum number of matches
        if len(good_matches) >= 10:
            # Estimate homography for pose
            src_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([query_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if homography is not None:
                # Calculate object center
                h, w = self.reference_image.shape
                object_center = np.float32([[w/2, h/2]]).reshape(-1, 1, 2)
                transformed_center = cv2.perspectiveTransform(object_center, homography)
                return transformed_center[0][0], len(good_matches)

        return None, 0
```

### 3D Object Detection

For physical interaction, 3D information is crucial:

```python
import open3d as o3d
import numpy as np

class Object3DTracker:
    def __init__(self, voxel_size=0.01):
        self.voxel_size = voxel_size
        self.previous_cloud = None
        self.tracked_objects = {}

    def process_point_cloud(self, point_cloud):
        """
        Process 3D point cloud to detect and track objects
        """
        # Downsample for efficiency
        cloud_down = point_cloud.voxel_down_sample(voxel_size=self.voxel_size)

        # Plane segmentation (for table detection)
        plane_model, inliers = cloud_down.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=1000
        )

        # Remove plane points to isolate objects
        object_cloud = cloud_down.select_by_index(inliers, invert=True)

        # Cluster objects
        with o3d.utility.VectorsInt() as cluster_idx:
            labels = np.array(object_cloud.cluster_dbscan(
                eps=self.voxel_size * 5,
                min_points=10,
                print_progress=False
            ))

        # Process each cluster
        objects = []
        for i in range(labels.max() + 1):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) > 50:  # Minimum cluster size
                cluster_cloud = object_cloud.select_by_index(cluster_indices)

                # Calculate bounding box and center
                bbox = cluster_cloud.get_axis_aligned_bounding_box()
                center = bbox.get_center()

                obj = {
                    'id': i,
                    'center': center,
                    'bbox': [bbox.min_bound, bbox.max_bound],
                    'point_count': len(cluster_indices)
                }
                objects.append(obj)

        return objects

    def estimate_pose(self, object_cloud, model_cloud):
        """
        Estimate 6D pose of object using ICP
        """
        # Estimate initial transformation
        initial_transformation = self.estimate_initial_transform(object_cloud, model_cloud)

        # Refine using ICP
        result = o3d.pipelines.registration.registration_icp(
            object_cloud, model_cloud, 0.02, initial_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
        )

        return result.transformation

    def estimate_initial_transform(self, source, target):
        """
        Estimate initial transformation using FPFH features
        """
        source_fpfh = self.compute_fpfh(source)
        target_fpfh = self.compute_fpfh(target)

        # Correspondence estimation
        corres = o3d.pipelines.registration.compute_correspondences(
            source_fpfh, target_fpfh, 0.05
        )

        # RANSAC for robust transformation estimation
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, True,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )

        return result.transformation
```

## Visual Servoing

Visual servoing uses visual feedback to control robot motion:

### Position-Based Visual Servoing

```python
class PositionBasedServo:
    def __init__(self, camera_matrix, target_position):
        self.K = camera_matrix  # Camera intrinsic matrix
        self.target_pos = target_position
        self.kp = 0.1  # Proportional gain

    def compute_control(self, current_object_pos, current_ee_pos):
        """
        Compute control command to move end-effector to target
        current_object_pos: [x, y, z] object position in camera frame
        current_ee_pos: [x, y, z] end-effector position in robot base frame
        """
        # Transform object position to robot base frame
        object_robot_frame = self.camera_to_robot_frame(current_object_pos)

        # Calculate position error
        pos_error = self.target_pos - object_robot_frame

        # Compute desired end-effector velocity
        ee_vel = self.kp * pos_error

        return ee_vel

    def camera_to_robot_frame(self, camera_pos):
        """
        Transform position from camera frame to robot base frame
        This would use the calibrated transformation between camera and robot
        """
        # Simplified transformation - in practice this uses calibrated extrinsics
        return camera_pos  # Placeholder
```

### Image-Based Visual Servoing

```python
class ImageBasedServo:
    def __init__(self, target_features, camera_params):
        self.target_features = target_features  # Target image features
        self.camera_params = camera_params
        self.kp = 0.5

    def compute_control(self, current_image_features):
        """
        Compute control based on image feature error
        """
        # Calculate feature error
        feature_error = self.target_features - current_image_features

        # Compute interaction matrix (simplified)
        interaction_matrix = self.compute_interaction_matrix()

        # Calculate image velocity
        image_vel = self.kp * feature_error

        # Convert to camera velocity
        camera_vel = np.linalg.inv(interaction_matrix) @ image_vel

        return camera_vel

    def compute_interaction_matrix(self):
        """
        Compute image Jacobian (interaction matrix)
        This relates image feature velocities to camera velocities
        """
        # Simplified 2D case for point features
        # In practice, this depends on the specific features being used
        L = np.zeros((2, 6))  # 2D features, 6D camera motion
        # Fill in the interaction matrix based on feature type
        return L
```

## Scene Understanding for Manipulation

### Semantic Segmentation

Semantic segmentation provides pixel-level understanding:

```python
import torch
import torchvision.transforms as transforms

class SemanticSegmenter:
    def __init__(self, model_path):
        # Load pre-trained segmentation model
        self.model = torch.hub.load('pytorch/vision:v0.10.0',
                                   'deeplabv3_resnet50',
                                   pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def segment_image(self, image):
        """
        Segment image into semantic classes
        """
        input_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
            predicted = torch.argmax(output, dim=0).cpu().numpy()

        return predicted

    def get_manipulable_objects(self, segmented_image, image):
        """
        Extract manipulable objects from segmentation
        """
        # Common classes for manipulation
        manipulable_classes = ['bottle', 'cup', 'book', 'bowl', 'remote', 'cell phone']

        objects = []
        for class_id, class_name in enumerate(self.get_class_names()):
            if class_name in manipulable_classes:
                mask = (segmented_image == class_id)
                if np.any(mask):
                    # Calculate object properties
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    for contour in contours:
                        if cv2.contourArea(contour) > 100:  # Minimum size
                            # Calculate bounding box
                            x, y, w, h = cv2.boundingRect(contour)
                            center_x, center_y = x + w//2, y + h//2

                            obj = {
                                'class': class_name,
                                'bbox': [x, y, w, h],
                                'center': [center_x, center_y],
                                'contour': contour
                            }
                            objects.append(obj)

        return objects

    def get_class_names(self):
        """
        Return mapping from class ID to name
        """
        return [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
```

### Instance Segmentation

For individual object tracking:

```python
class InstanceSegmenter:
    def __init__(self, confidence_threshold=0.5):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s-seg', pretrained=True)
        self.confidence_threshold = confidence_threshold

    def segment_instances(self, image):
        """
        Segment individual object instances
        """
        results = self.model(image)

        # Get masks and bounding boxes
        masks = results.masks.data.cpu().numpy() if results.masks is not None else []
        boxes = results.pandas().xyxy[0] if results.pandas().xyxy[0] is not None else []

        instances = []
        for i, (mask, box) in enumerate(zip(masks, boxes)):
            if box['confidence'] > self.confidence_threshold:
                instance = {
                    'class': box['name'],
                    'confidence': box['confidence'],
                    'bbox': [box['xmin'], box['ymin'], box['xmax'], box['ymax']],
                    'mask': mask,
                    'area': np.sum(mask > 0.5)
                }
                instances.append(instance)

        return instances
```

## Depth Perception and 3D Understanding

### Depth Estimation

For robots without depth sensors, monocular depth estimation:

```python
class MonocularDepthEstimator:
    def __init__(self):
        # Load pre-trained depth estimation model
        # Using MiDaS for monocular depth estimation
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model.eval()

        # Transform for MiDaS
        self.transform = transforms.Compose([
            transforms.Resize(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def estimate_depth(self, image):
        """
        Estimate depth from single RGB image
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        input_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            prediction = self.model(input_tensor)

        # Resize back to original size
        depth_map = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.size[::-1],  # [height, width]
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        return depth_map.cpu().numpy()
```

### Stereo Vision

For more accurate depth with stereo cameras:

```python
class StereoDepthEstimator:
    def __init__(self, left_camera_params, right_camera_params):
        # Stereo rectification parameters
        self.left_camera_matrix = left_camera_params['K']
        self.right_camera_matrix = right_camera_params['K']
        self.distortion_coeffs = left_camera_params['distortion']

        # Stereo block matcher
        self.stereo = cv2.StereoSGBM_create(
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

    def compute_disparity(self, left_image, right_image):
        """
        Compute disparity map from stereo images
        """
        gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        # Compute disparity
        disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

        return disparity

    def disparity_to_depth(self, disparity, baseline, focal_length):
        """
        Convert disparity to depth
        baseline: distance between cameras (meters)
        focal_length: focal length in pixels
        """
        # Avoid division by zero
        disparity[disparity == 0] = 0.01
        depth = (baseline * focal_length) / disparity
        return depth
```

## Visual Attention and Focus

### Saliency Detection

Identifying important regions in images:

```python
class SaliencyDetector:
    def __init__(self):
        # Use OpenCV's spectral residual saliency detector
        self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

    def detect_salient_regions(self, image):
        """
        Detect salient regions in image
        """
        success, saliency_map = self.saliency.computeSaliency(image)

        if success:
            # Threshold to get binary mask of salient regions
            _, binary_map = cv2.threshold(saliency_map, 0.5, 255, cv2.THRESH_BINARY)
            binary_map = binary_map.astype(np.uint8)

            # Find contours of salient regions
            contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            salient_regions = []
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Minimum region size
                    # Calculate bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x, center_y = x + w//2, y + h//2

                    region = {
                        'bbox': [x, y, w, h],
                        'center': [center_x, center_y],
                        'area': cv2.contourArea(contour)
                    }
                    salient_regions.append(region)

            return salient_regions, saliency_map

        return [], None
```

## Multi-Modal Fusion

Combining vision with other sensors:

```python
class MultiModalFusion:
    def __init__(self):
        self.vision_processor = YOLODetector("yolo_model.pt")
        self.depth_estimator = MonocularDepthEstimator()
        self.tactile_threshold = 0.5

    def fuse_modalities(self, rgb_image, tactile_data=None, force_data=None):
        """
        Fuse visual and tactile/force information
        """
        # Process visual information
        objects = self.vision_processor.detect_objects(rgb_image)
        depth_map = self.depth_estimator.estimate_depth(rgb_image)

        fused_percepts = []

        for obj in objects:
            # Get depth at object center
            center_x, center_y = int(obj['center'][0]), int(obj['center'][1])
            depth = depth_map[center_y, center_x] if (0 <= center_y < depth_map.shape[0] and
                                                    0 <= center_x < depth_map.shape[1]) else None

            # Incorporate tactile information if available
            tactile_feedback = None
            if tactile_data is not None:
                # Check if object location corresponds to tactile contact
                tactile_feedback = self.process_tactile_data(tactile_data, obj['bbox'])

            fused_percept = {
                'class': obj['class'],
                'confidence': obj['confidence'],
                'bbox': obj['bbox'],
                'center': obj['center'],
                'depth': depth,
                'tactile': tactile_feedback,
                'fusion_confidence': self.compute_fusion_confidence(obj, depth, tactile_feedback)
            }

            fused_percepts.append(fused_percept)

        return fused_percepts

    def process_tactile_data(self, tactile_data, bbox):
        """
        Process tactile sensor data in context of visual detection
        """
        # Implementation depends on tactile sensor type and layout
        # This is a simplified example
        if tactile_data.get('contact', False):
            return {
                'contact': True,
                'force': tactile_data.get('force', 0),
                'location': tactile_data.get('location', 'unknown')
            }
        return None

    def compute_fusion_confidence(self, vision_percept, depth, tactile):
        """
        Compute confidence in fused percept
        """
        confidence = vision_percept['confidence']

        # Boost confidence if depth is reasonable
        if depth and 0.1 < depth < 3.0:  # Reasonable depth range
            confidence *= 1.2

        # Boost confidence if tactile confirms contact
        if tactile and tactile['contact']:
            confidence *= 1.3

        return min(confidence, 1.0)  # Cap at 1.0
```

## Real-Time Performance Optimization

### Model Optimization for Edge Deployment

```python
class OptimizedVisionPipeline:
    def __init__(self, model_path):
        # Load optimized model (e.g., TensorRT, ONNX Runtime, OpenVINO)
        import onnxruntime as ort

        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, image):
        """
        Run optimized inference
        """
        # Preprocess image
        input_tensor = self.preprocess(image)

        # Run inference
        results = self.session.run(None, {self.input_name: input_tensor})

        return self.postprocess(results)

    def preprocess(self, image):
        """
        Preprocess image for optimized model
        """
        # Resize and normalize
        image = cv2.resize(image, (640, 640))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = np.expand_dims(image, axis=0)   # Add batch dimension

        return image

    def postprocess(self, results):
        """
        Postprocess model outputs
        """
        # Implementation depends on model architecture
        return results[0]
```

## Safety Considerations

### Safe Visual Processing

```python
class SafeVisionProcessor:
    def __init__(self):
        self.max_processing_time = 0.05  # 50ms maximum processing time
        self.confidence_threshold = 0.7
        self.min_object_size = 50  # Minimum pixels for valid detection

    def process_with_safety(self, image):
        """
        Process image with safety checks
        """
        import time

        start_time = time.time()

        # Run object detection
        detections = self.detect_objects(image)

        # Filter by safety criteria
        safe_detections = []
        for detection in detections:
            # Check confidence
            if detection['confidence'] < self.confidence_threshold:
                continue

            # Check object size
            bbox = detection['bbox']
            obj_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if obj_size < self.min_object_size:
                continue

            # Check processing time
            if time.time() - start_time > self.max_processing_time:
                print("Warning: Processing time exceeded safety limit")
                break

            safe_detections.append(detection)

        return safe_detections
```

## Learning Objectives

After studying this chapter, you should be able to:

1. Implement various computer vision techniques for physical AI interaction
2. Design vision pipelines that provide actionable information for control
3. Apply visual servoing for precise robot control
4. Fuse visual information with other sensory modalities
5. Optimize vision systems for real-time performance
6. Address safety considerations in vision-based control

## Interactive Vision Demonstration

Try out computer vision concepts with our interactive demonstration:

import InteractiveDemo from '@site/src/components/InteractiveDemo';

<InteractiveDemo
  simulationType="computer_vision"
  parameters={{environment: "object_detection", duration: 5, speed: 1.0}}
/>

## Hands-On Exercise

Implement a vision-based grasping system:

1. Set up a camera for object detection
2. Implement object detection using YOLO or similar
3. Calculate 3D position of detected objects
4. Plan robot motion to approach and grasp objects
5. Implement visual feedback for grasp verification
6. Test the system with various objects and evaluate performance

This exercise will help you understand the practical challenges of integrating computer vision with robotic manipulation.