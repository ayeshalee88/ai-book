import React, { useState } from 'react';

interface MessageBubbleProps {
  text: string;
  sender: 'user' | 'assistant';
  timestamp?: Date;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ text, sender, timestamp }) => {
  const [showSources, setShowSources] = useState(false);

  const toggleSources = () => {
    setShowSources(!showSources);
  };

  return (
    <div className={`message-bubble ${sender}`}>
      <div className="message-content">
        {sender === 'assistant' && (
          <div className="avatar">
            🤖
          </div>
        )}
        
        <div className="bubble">
          {text}
        </div>
        
        {sender === 'user' && (
          <div className="avatar">
            👤
          </div>
        )}
      </div>
      
      {timestamp && (
        <div className="timestamp">
          {timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </div>
      )}
      
      {sender === 'assistant' && (
        <>
          <button className="sources-toggle" onClick={toggleSources}>
            {showSources ? 'Hide Sources' : 'Show Sources'}
          </button>
          {showSources && (
            <div className="sources">
              <p>Sources will appear here</p>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default MessageBubble;