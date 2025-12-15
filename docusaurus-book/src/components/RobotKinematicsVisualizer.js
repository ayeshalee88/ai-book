import React, { useState, useEffect, useRef } from 'react';
import styles from './RobotKinematicsVisualizer.module.css';

const RobotKinematicsVisualizer = ({
  robotModel = "simple_arm",
  jointAngles = [0, 0],
  targetPosition = null,
  showWorkspace = true,
  interactive = true
}) => {
  const canvasRef = useRef(null);
  const [currentAngles, setCurrentAngles] = useState(jointAngles);
  const [dragging, setDragging] = useState(false);
  const [dragPosition, setDragPosition] = useState(null);

  // Simple 2D robot arm visualization
  const canvasWidth = 400;
  const canvasHeight = 400;
  const baseX = canvasWidth / 2;
  const baseY = canvasHeight - 50;
  const link1Length = 100;
  const link2Length = 80;

  // Forward kinematics calculation
  const calculateEndEffector = (angles) => {
    const [theta1, theta2] = angles;

    const x1 = baseX + link1Length * Math.cos(theta1);
    const y1 = baseY - link1Length * Math.sin(theta1);

    const x2 = x1 + link2Length * Math.cos(theta1 + theta2);
    const y2 = y1 - link2Length * Math.sin(theta1 + theta2);

    return { x1, y1, x2, y2 };
  };

  // Inverse kinematics calculation (simple 2-DOF arm)
  const inverseKinematics = (targetX, targetY) => {
    // Transform to robot coordinate system
    const dx = targetX - baseX;
    const dy = baseY - targetY; // Y is inverted in canvas coordinates
    const r = Math.sqrt(dx * dx + dy * dy);

    // Check if target is reachable
    if (r > link1Length + link2Length) {
      // Target is outside workspace, extend fully toward target
      const angle = Math.atan2(dy, dx);
      return [angle, 0];
    } else if (r < Math.abs(link1Length - link2Length)) {
      // Target is inside workspace but unreachable
      return [0, 0]; // Return default position
    } else {
      // Calculate joint angles using law of cosines
      const cosTheta2 = (r * r - link1Length * link1Length - link2Length * link2Length) /
                        (2 * link1Length * link2Length);
      const theta2 = Math.acos(Math.max(-1, Math.min(1, cosTheta2)));

      const k1 = link1Length + link2Length * Math.cos(theta2);
      const k2 = link2Length * Math.sin(theta2);

      const theta1 = Math.atan2(dy, dx) - Math.atan2(k2, k1);

      return [theta1, theta2];
    }
  };

  // Draw the robot arm
  const drawRobot = (ctx) => {
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    // Draw workspace (if enabled)
    if (showWorkspace) {
      ctx.beginPath();
      ctx.arc(baseX, baseY, link1Length + link2Length, 0, 2 * Math.PI);
      ctx.strokeStyle = 'rgba(200, 200, 200, 0.3)';
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(baseX, baseY, Math.abs(link1Length - link2Length), 0, 2 * Math.PI);
      ctx.strokeStyle = 'rgba(200, 200, 200, 0.3)';
      ctx.stroke();
    }

    // Calculate end effector position
    const { x1, y1, x2, y2 } = calculateEndEffector(currentAngles);

    // Draw base
    ctx.beginPath();
    ctx.arc(baseX, baseY, 10, 0, 2 * Math.PI);
    ctx.fillStyle = '#333';
    ctx.fill();

    // Draw first link
    ctx.beginPath();
    ctx.moveTo(baseX, baseY);
    ctx.lineTo(x1, y1);
    ctx.lineWidth = 8;
    ctx.strokeStyle = '#2e8555';
    ctx.stroke();

    // Draw joint 1
    ctx.beginPath();
    ctx.arc(x1, y1, 6, 0, 2 * Math.PI);
    ctx.fillStyle = '#333';
    ctx.fill();

    // Draw second link
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.lineWidth = 6;
    ctx.strokeStyle = '#2e8555';
    ctx.stroke();

    // Draw end effector
    ctx.beginPath();
    ctx.arc(x2, y2, 8, 0, 2 * Math.PI);
    ctx.fillStyle = '#d95757';
    ctx.fill();

    // Draw target position (if specified)
    if (targetPosition) {
      ctx.beginPath();
      ctx.arc(targetPosition[0], targetPosition[1], 10, 0, 2 * Math.PI);
      ctx.strokeStyle = '#d95757';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(targetPosition[0], targetPosition[1], 3, 0, 2 * Math.PI);
      ctx.fillStyle = '#d95757';
      ctx.fill();
    }

    // Draw current end effector position
    ctx.beginPath();
    ctx.arc(x2, y2, 5, 0, 2 * Math.PI);
    ctx.fillStyle = '#57a0d9';
    ctx.fill();
  };

  // Handle mouse events for interactive mode
  const handleMouseDown = (e) => {
    if (!interactive) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setDragging(true);
    setDragPosition([x, y]);

    // Calculate new joint angles based on clicked position
    const newAngles = inverseKinematics(x, y);
    setCurrentAngles(newAngles);
  };

  const handleMouseMove = (e) => {
    if (!dragging || !interactive) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setDragPosition([x, y]);

    // Calculate new joint angles based on current mouse position
    const newAngles = inverseKinematics(x, y);
    setCurrentAngles(newAngles);
  };

  const handleMouseUp = () => {
    setDragging(false);
  };

  // Draw on canvas when state changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      drawRobot(ctx);
    }
  }, [currentAngles, targetPosition, showWorkspace]);

  // Update joint angles when prop changes
  useEffect(() => {
    setCurrentAngles(jointAngles);
  }, [jointAngles]);

  return (
    <div className={styles.visualizerContainer} role="region" aria-labelledby="visualizer-title">
      <h3 id="visualizer-title">Robot Kinematics Visualizer</h3>
      <p>2-DOF Planar Robot Arm Visualization</p>

      {/* Fallback content for non-JS environments */}
      <noscript>
        <div className={styles.fallbackContent}>
          <h4>Interactive Visualization Disabled</h4>
          <p>To see the interactive robot kinematics visualization, please enable JavaScript.</p>
          <p>In the meantime, here's a description of the 2-DOF robot arm:</p>
          <ul>
            <li>Base position: Center-bottom of visualization area</li>
            <li>Link 1: 100 units long, connected to base</li>
            <li>Link 2: 80 units long, connected to end of Link 1</li>
            <li>Forward kinematics: Calculates end-effector position from joint angles</li>
            <li>Inverse kinematics: Calculates joint angles from end-effector position</li>
          </ul>
        </div>
      </noscript>

      <div className={styles.controls} role="group" aria-label="Robot arm controls">
        <div className={styles.sliderContainer}>
          <label htmlFor="joint1-slider">
            Joint 1: {(currentAngles[0] * 180 / Math.PI).toFixed(1)}°
          </label>
          <input
            id="joint1-slider"
            type="range"
            min={-Math.PI}
            max={Math.PI}
            step={0.01}
            value={currentAngles[0]}
            onChange={(e) => setCurrentAngles([parseFloat(e.target.value), currentAngles[1]])}
            disabled={!interactive}
            aria-valuemin={-Math.PI}
            aria-valuemax={Math.PI}
            aria-valuenow={currentAngles[0]}
            aria-valuetext={`Joint 1 angle: ${(currentAngles[0] * 180 / Math.PI).toFixed(1)} degrees`}
          />
        </div>

        <div className={styles.sliderContainer}>
          <label htmlFor="joint2-slider">
            Joint 2: {(currentAngles[1] * 180 / Math.PI).toFixed(1)}°
          </label>
          <input
            id="joint2-slider"
            type="range"
            min={-Math.PI}
            max={Math.PI}
            step={0.01}
            value={currentAngles[1]}
            onChange={(e) => setCurrentAngles([currentAngles[0], parseFloat(e.target.value)])}
            disabled={!interactive}
            aria-valuemin={-Math.PI}
            aria-valuemax={Math.PI}
            aria-valuenow={currentAngles[1]}
            aria-valuetext={`Joint 2 angle: ${(currentAngles[1] * 180 / Math.PI).toFixed(1)} degrees`}
          />
        </div>
      </div>

      <div className={styles.canvasContainer} role="img" aria-label="Robot arm visualization">
        <canvas
          ref={canvasRef}
          width={canvasWidth}
          height={canvasHeight}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          className={styles.robotCanvas}
          aria-hidden="true"
        />
      </div>

      <div className={styles.infoPanel} role="status" aria-live="polite">
        <h4>Current Position</h4>
        <p>End Effector: ({calculateEndEffector(currentAngles).x2.toFixed(1)}, {calculateEndEffector(currentAngles).y2.toFixed(1)})</p>
        <p>Joint Angles: [{(currentAngles[0] * 180 / Math.PI).toFixed(1)}°, {(currentAngles[1] * 180 / Math.PI).toFixed(1)}°]</p>
      </div>
    </div>
  );
};

export default RobotKinematicsVisualizer;