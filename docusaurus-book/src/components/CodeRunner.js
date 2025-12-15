import React, { useState } from 'react';
import styles from './CodeRunner.module.css';

const CodeRunner = ({
  language = "python",
  code: initialCode = "",
  dependencies = [],
  readOnly = false,
  showLineNumbers = true,
  onRun
}) => {
  const [code, setCode] = useState(initialCode);
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState('');

  const runCode = async () => {
    setIsRunning(true);
    setError('');
    setOutput('');

    try {
      // Simulate code execution
      setTimeout(() => {
        const result = {
          output: `Executed ${language} code successfully\nResult: Sample output from code execution`,
          error: null,
          executionTime: 0.15,
          success: true
        };

        setOutput(result.output);
        setIsRunning(false);

        if (onRun) {
          onRun(result);
        }
      }, 1000);
    } catch (err) {
      setError(err.message);
      setIsRunning(false);
    }
  };

  const resetCode = () => {
    setCode(initialCode);
    setOutput('');
    setError('');
  };

  return (
    <div className={styles.codeRunnerContainer} role="region" aria-labelledby="code-runner-title">
      <div className={styles.header}>
        <h3 id="code-runner-title">Code Runner</h3>
        <span className={styles.languageBadge} aria-label={`Language: ${language}`}>{language}</span>
      </div>

      {/* Fallback content for non-JS environments */}
      <noscript>
        <div className={styles.fallbackContent}>
          <h4>Code Runner Disabled</h4>
          <p>To use the interactive code runner, please enable JavaScript.</p>
          <p>Language: {language}</p>
          <p>Dependencies: {dependencies.join(', ')}</p>
          <p>Code example:</p>
          <pre className={styles.fallbackCode}>{initialCode}</pre>
        </div>
      </noscript>

      <label htmlFor="code-runner-textarea" className={styles.visuallyHidden}>
        Code Editor: Edit the {language} code here
      </label>
      <textarea
        id="code-runner-textarea"
        className={`${styles.codeInput} ${showLineNumbers ? styles.withLineNumbers : ''}`}
        value={code}
        onChange={(e) => setCode(e.target.value)}
        readOnly={readOnly}
        rows={10}
        spellCheck={false}
        aria-describedby="code-runner-description"
      />
      <div id="code-runner-description" className={styles.visuallyHidden}>
        Interactive code editor for {language} code. Use the Run button to execute the code and see the output.
      </div>

      <div className={styles.controls} role="group" aria-label="Code runner controls">
        <button
          onClick={runCode}
          disabled={isRunning}
          aria-busy={isRunning}
          className={styles.primaryButton}
        >
          {isRunning ? 'Running...' : 'Run Code'}
        </button>
        <button
          onClick={resetCode}
          className={styles.secondaryButton}
        >
          Reset
        </button>
      </div>

      {(output || error) && (
        <div className={`${styles.output} ${error ? styles.error : ''}`} role="status" aria-live="polite">
          <h4>Output:</h4>
          <pre>{error || output}</pre>
        </div>
      )}
    </div>
  );
};

export default CodeRunner;