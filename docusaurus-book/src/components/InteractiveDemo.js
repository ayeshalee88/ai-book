import React from 'react';
import styles from './InteractiveDemo.module.css';

const InteractiveDemo = ({ simulationType = "locomotion", parameters = {}, onSimulationStart, onSimulationEnd }) => {
  const [isRunning, setIsRunning] = React.useState(false);
  const [simulationData, setSimulationData] = React.useState(null);

  const startSimulation = () => {
    setIsRunning(true);
    if (onSimulationStart) onSimulationStart();

    // Simulate a basic simulation run
    setTimeout(() => {
      setSimulationData({
        status: 'completed',
        results: `Simulation of type "${simulationType}" completed successfully`,
        timestamp: new Date().toISOString()
      });
      setIsRunning(false);
      if (onSimulationEnd) onSimulationEnd();
    }, 2000);
  };

  const resetSimulation = () => {
    setIsRunning(false);
    setSimulationData(null);
  };

  return (
    <div className={styles.interactiveDemoContainer} role="region" aria-labelledby="interactive-demo-title">
      <h3 id="interactive-demo-title">Interactive Simulation Demo</h3>
      <p>Type: {simulationType}</p>

      {/* Fallback content for non-JS environments */}
      <noscript>
        <div className={styles.fallbackContent}>
          <h4>Interactive Simulation Disabled</h4>
          <p>To see the interactive simulation demo, please enable JavaScript.</p>
          <p>Simulation type: {simulationType}</p>
          <p>Parameters: {JSON.stringify(parameters, null, 2)}</p>
        </div>
      </noscript>

      <div className={styles.controls} role="group" aria-label="Simulation controls">
        <button
          onClick={startSimulation}
          disabled={isRunning}
          aria-busy={isRunning}
          className={styles.primaryButton}
        >
          {isRunning ? 'Running...' : 'Start Simulation'}
        </button>
        <button
          onClick={resetSimulation}
          className={styles.secondaryButton}
        >
          Reset
        </button>
      </div>
      {simulationData && (
        <div className={styles.results} role="status" aria-live="polite">
          <h4>Results:</h4>
          <p>{simulationData.results}</p>
          <p>Time: {simulationData.timestamp}</p>
        </div>
      )}
      {!simulationData && (
        <div className={styles.info} role="status" aria-live="polite">
          <p>Click "Start Simulation" to run the interactive demo.</p>
        </div>
      )}
    </div>
  );
};

export default InteractiveDemo;