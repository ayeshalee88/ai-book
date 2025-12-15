import React from 'react';
import styles from './RobotDiagram.module.css';

const RobotDiagram = ({ type = 'humanoid', title = 'Robot Diagram', description = '' }) => {
  const renderDiagram = () => {
    switch (type) {
      case 'humanoid':
        return (
          <div className={styles.humanoidDiagram}>
            <div className={styles.head}></div>
            <div className={styles.torso}></div>
            <div className={styles.arm}></div>
            <div className={styles.arm}></div>
            <div className={styles.leg}></div>
            <div className={styles.leg}></div>
          </div>
        );
      case 'sensorimotor':
        return (
          <div className={styles.sensorimotorDiagram}>
            <div className={styles.sensors}>
              <div className={styles.sensor}>Vision</div>
              <div className={styles.sensor}>Touch</div>
              <div className={styles.sensor}>Proprioception</div>
            </div>
            <div className={styles.processing}>Processing</div>
            <div className={styles.actuation}>Actuation</div>
            <div className={styles.environment}>Environment</div>
            <div className={styles.loop}>Sensorimotor Loop</div>
          </div>
        );
      case 'control':
        return (
          <div className={styles.controlDiagram}>
            <div className={styles.reference}>Reference</div>
            <div className={styles.controller}>Controller</div>
            <div className={styles.system}>Physical System</div>
            <div className={styles.sensor}>Sensor</div>
            <div className={styles.feedback}>Feedback</div>
          </div>
        );
      default:
        return (
          <div className={styles.defaultDiagram}>
            <div className={styles.robotBody}></div>
          </div>
        );
    }
  };

  return (
    <div className={styles.robotDiagramContainer}>
      <h3 className={styles.title}>{title}</h3>
      <div className={styles.diagram}>
        {renderDiagram()}
      </div>
      {description && <p className={styles.description}>{description}</p>}
    </div>
  );
};

export default RobotDiagram;