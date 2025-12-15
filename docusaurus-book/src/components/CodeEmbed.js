import React from 'react';
import styles from './CodeEmbed.module.css';
import BrowserOnly from '@docusaurus/BrowserOnly';

const CodeEmbed = ({ code, language = 'python', title = 'Code Example', description = '' }) => {
  const copyToClipboard = async (text) => {
    if (navigator.clipboard) {
      try {
        await navigator.clipboard.writeText(text);
        // You could add a visual feedback here
        console.log('Code copied to clipboard');
      } catch (err) {
        console.error('Failed to copy code: ', err);
      }
    }
  };

  return (
    <div className={styles.codeEmbedContainer}>
      <div className={styles.codeHeader}>
        <h3 className={styles.title}>{title}</h3>
        <button
          className={styles.copyButton}
          onClick={() => copyToClipboard(code)}
          title="Copy code to clipboard"
        >
          Copy
        </button>
      </div>
      {description && <p className={styles.description}>{description}</p>}
      <div className={styles.codeBlock}>
        <pre className={styles.pre}>
          <code className={`language-${language}`}>
            {code}
          </code>
        </pre>
      </div>
      <div className={styles.codeFooter}>
        <span className={styles.languageTag}>{language}</span>
      </div>
    </div>
  );
};

// For executable code examples (client-side only)
const ExecutableCode = ({ code, title = 'Interactive Example' }) => {
  const [output, setOutput] = React.useState('');
  const [isRunning, setIsRunning] = React.useState(false);

  const executeCode = async () => {
    setIsRunning(true);
    setOutput('Running...');

    // This is a simplified example - in practice, you'd want to use a proper
    // sandboxed environment for security reasons
    try {
      // In a real implementation, you would use something like:
      // - Pyodide for Python execution in the browser
      // - Web Workers for safer execution
      // - A backend service for actual execution
      setOutput('Code execution would happen here in a production environment.\n\nThis is a placeholder for interactive code execution.');
    } catch (error) {
      setOutput(`Error: ${error.message}`);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className={styles.executableContainer}>
      <div className={styles.codeHeader}>
        <h3 className={styles.title}>{title}</h3>
        <button
          className={styles.runButton}
          onClick={executeCode}
          disabled={isRunning}
        >
          {isRunning ? 'Running...' : 'Run Code'}
        </button>
      </div>
      <div className={styles.codeBlock}>
        <pre className={styles.pre}>
          <code className="language-python">
            {code}
          </code>
        </pre>
      </div>
      {output && (
        <div className={styles.outputBlock}>
          <h4>Output:</h4>
          <pre className={styles.outputPre}>{output}</pre>
        </div>
      )}
    </div>
  );
};

const CodeEmbedWithExecution = ({ code, language = 'python', title = 'Code Example', description = '', executable = false }) => {
  if (executable) {
    return (
      <BrowserOnly fallback={<div>Loading interactive example...</div>}>
        {() => <ExecutableCode code={code} title={title} />}
      </BrowserOnly>
    );
  }

  return <CodeEmbed code={code} language={language} title={title} description={description} />;
};

export default CodeEmbedWithExecution;