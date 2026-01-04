import React from 'react';
import MessageBubble from './MessageBubble';
import { Message } from './types';

interface MessageListProps {
  messages: Message[];
}

const MessageList: React.FC<MessageListProps> = ({ messages }) => {
  return (
    <div className="message-list-container">
      <div className="message-list">
        {messages.map((message, index) => (
          <MessageBubble
            key={index}
            text={message.text}
            sender={message.sender}
            timestamp={message.timestamp}
          />
        ))}
      </div>
    </div>
  );
};

export default MessageList;
