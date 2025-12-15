import React, { useState, useEffect } from 'react';
import styles from './SimulationViewer.module.css';

const SimulationViewer = ({
  title = 'Robot Simulation',
  description = 'Interactive robot simulation viewer',
  simulationType = 'simple-agent',
  initialData = null
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [timeStep, setTimeStep] = useState(0);
  const [simulationData, setSimulationData] = useState(initialData || {
    agentPosition: { x: 50, y: 50 },
    targetPosition: { x: 350, y: 300 },
    obstacles: [
      { x: 150, y: 150, width: 30, height: 30 },
      { x: 250, y: 200, width: 40, height: 40 },
      { x: 100, y: 250, width: 25, height: 25 }
    ],
    path: []
  });

  useEffect(() => {
    let interval = null;
    if (isPlaying) {
      interval = setInterval(() => {
        setTimeStep(prev => prev + 1);

        // Simple simulation logic for demonstration
        setSimulationData(prev => {
          const newAgentPos = { ...prev.agentPosition };

          // Simple movement toward target (avoiding obstacles is simplified)
          const dx = prev.targetPosition.x - newAgentPos.x;
          const dy = prev.targetPosition.y - newAgentPos.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance > 5) {
            newAgentPos.x += (dx / distance) * 2;
            newAgentPos.y += (dy / distance) * 2;
          }

          return {
            ...prev,
            agentPosition: newAgentPos,
            path: [...prev.path, { ...newAgentPos }]
          };
        });
      }, 100);
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  const toggleSimulation = () => {
    setIsPlaying(!isPlaying);
  };

  const resetSimulation = () => {
    setIsPlaying(false);
    setTimeStep(0);
    setSimulationData(initialData || {
      agentPosition: { x: 50, y: 50 },
      targetPosition: { x: 350, y: 300 },
      obstacles: [
        { x: 150, y: 150, width: 30, height: 30 },
        { x: 250, y: 200, width: 40, height: 40 },
        { x: 100, y: 250, width: 25, height: 25 }
      ],
      path: []
    });
  };

  return (
    <div className={styles.simulationContainer}>
      <div className={styles.simulationHeader}>
        <h3 className={styles.title}>{title}</h3>
        <div className={styles.controls}>
          <button
            className={`${styles.controlButton} ${isPlaying ? styles.pauseButton : styles.playButton}`}
            onClick={toggleSimulation}
          >
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button
            className={styles.controlButton}
            onClick={resetSimulation}
          >
            Reset
          </button>
        </div>
      </div>

      <p className={styles.description}>{description}</p>

      <div className={styles.simulationArea}>
        <svg
          width="400"
          height="400"
          className={styles.simulationSvg}
          viewBox="0 0 400 400"
        >
          {/* Background grid */}
          <defs>
            <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#f0f0f0" strokeWidth="0.5"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />

          {/* Obstacles */}
          {simulationData.obstacles.map((obs, index) => (
            <rect
              key={index}
              x={obs.x}
              y={obs.y}
              width={obs.width}
              height={obs.height}
              fill="#ff6b6b"
              stroke="#ff5252"
              strokeWidth="1"
              rx="4"
            />
          ))}

          {/* Path */}
          {simulationData.path.length > 1 && (
            <polyline
              points={simulationData.path.map(p => `${p.x},${p.y}`).join(' ')}
              fill="none"
              stroke="#4ecdc4"
              strokeWidth="2"
              strokeOpacity="0.7"
            />
          )}

          {/* Agent */}
          <circle
            cx={simulationData.agentPosition.x}
            cy={simulationData.agentPosition.y}
            r="8"
            fill="#45b7d1"
            stroke="#3a9dc1"
            strokeWidth="2"
          />

          {/* Target */}
          <circle
            cx={simulationData.targetPosition.x}
            cy={simulationData.targetPosition.y}
            r="10"
            fill="#96ceb4"
            stroke="#7eb693"
            strokeWidth="2"
            opacity="0.8"
          />
          <text
            x={simulationData.targetPosition.x}
            y={simulationData.targetPosition.y + 25}
            textAnchor="middle"
            fontSize="12"
            fill="#777"
          >
            Target
          </text>
        </svg>
      </div>

      <div className={styles.simulationInfo}>
        <div>Time Step: {timeStep}</div>
        <div>Agent Position: ({Math.round(simulationData.agentPosition.x)}, {Math.round(simulationData.agentPosition.y)})</div>
        <div>Path Length: {simulationData.path.length} steps</div>
      </div>
    </div>
  );
};

export default SimulationViewer;