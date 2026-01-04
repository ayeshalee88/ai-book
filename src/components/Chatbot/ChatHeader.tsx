import React from 'react';

interface ChatHeaderProps {
  title?: string;
  subtitle?: string;
  onMinimize?: () => void;
  onMaximize?: () => void;
}

const ChatHeader: React.FC<ChatHeaderProps> = ({
  title = 'AYI BUDDY',
  subtitle = 'Always here to help',
  onMinimize,
  onMaximize
}) => {
  return (
    <div className="chat-header">
      <div className="header-content">
        <div className="logo-placeholder">
          🤖
        </div>
        <div className="header-text">
          <h3>{title}</h3>
          {subtitle && <div className="header-subtitle">{subtitle}</div>}
        </div>
      </div>
      <div className="header-controls">
        {onMinimize && (
          <button 
            className="control-button" 
            onClick={onMinimize} 
            aria-label="Minimize"
            title="Minimize"
          >
            −
          </button>
        )}
        {onMaximize && (
          <button 
            className="control-button" 
            onClick={onMaximize} 
            aria-label="Close"
            title="Close"
          >
            ✕
          </button>
        )}
      </div>
    </div>
  );
};

export default ChatHeader;