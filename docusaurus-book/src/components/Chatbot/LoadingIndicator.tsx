import React from 'react';

interface LoadingIndicatorProps {
  visible?: boolean;
  message?: string;
}

const LoadingIndicator: React.FC<LoadingIndicatorProps> = ({
  visible = true,
  message = 'Thinking...'
}) => {
  if (!visible) {
    return null;
  }

  return (
    <div className="loading-indicator" role="status" aria-label="Loading">
      <div className="loading-spinner">
        <div className="spinner-dot"></div>
        <div className="spinner-dot"></div>
        <div className="spinner-dot"></div>
      </div>
      <span className="loading-text">{message}</span>
    </div>
  );
};

export default LoadingIndicator;