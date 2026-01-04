import React, { useState } from 'react';
import '../../styles/chatbot.css';
import ChatContainer from './ChatContainer';

interface ChatWidgetProps {
  // Add any props here
}

const ChatWidget: React.FC<ChatWidgetProps> = () => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleWidget = () => {
    setIsOpen(true);
  };

  const handleClose = () => {
    setIsOpen(false);
  };

  return (
    <div className="chat-widget">
      {!isOpen && (
        <button className="chat-toggle" onClick={toggleWidget}>
          <img
          src="/img/ebuddy.PNG"         // ← your image path
          alt="Chat"
          className="w-32 h-32 rounded-full object-cover"  // adjust size as needed
  />
        </button>
      )}
      
      {isOpen && (
        <ChatContainer onClose={handleClose} />
      )}
    </div>
  );
};

export default ChatWidget;