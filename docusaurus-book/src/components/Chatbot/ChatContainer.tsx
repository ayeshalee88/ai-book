import React, { useState, useCallback } from 'react';
import MessageList from './MessageList';
import ChatInput from './ChatInput';
import LoadingIndicator from './LoadingIndicator';
import TextSelectionListener from './TextSelectionListener';
import { Message } from './types';
import useApi from '../../hooks/useApi';
import { useChatHistory } from '../../hooks/useChatHistory';

interface ChatContainerProps {
  initialMessages?: Message[];
  onClose: () => void;
}

const ChatContainer: React.FC<ChatContainerProps> = ({ initialMessages = [], onClose }) => {
  console.log('CHAT CONTAINER RENDERED - USING useApi()');
  const { messages, addMessage, clearHistory, updateMessage } = useChatHistory();
  const [selectedText, setSelectedText] = useState<string>('');

  const API_URL = process.env.REACT_APP_API_URL || 'https://ayishaalee-backend.hf.space';
  const { loading, error, execute, reset } = useApi(`${API_URL}/api/v1/qa`);
  
  const handleSendMessage = useCallback(async (text: string) => {
    // Add user message to the chat
    const userMessage: Message = {
      text,
      sender: 'user',
      timestamp: new Date(),
    };

    addMessage(userMessage);

    try {
      const response = await execute({
        message: text,
        top_k: 3,
      });

      const assistantMessage: Message = {
        text: response.response,
        sender: 'assistant',
        timestamp: new Date(),
        sources: response.sources ?? [],
      };

      addMessage(assistantMessage);
    } catch (err) {
      console.error('Error getting response:', err);

      const errorMessage: Message = {
        text: 'Sorry, I encountered an error processing your request.',
        sender: 'assistant',
        timestamp: new Date(),
      };

      addMessage(errorMessage);
    }
  }, [addMessage, execute]);

  const handleTextSelected = (text: string) => {
    setSelectedText(text);
  };

  return (
    <div className="chat-container-full">
      <TextSelectionListener onTextSelected={handleTextSelected} />
      
      {/* Chat Header with Close Button */}
      <div className="chat-header">
        <div className="header-content">
          <div className="logo-placeholder">🤖</div>
          <div className="header-text">
            <h3>AYI BUDDY</h3>
            <div className="header-subtitle">Ask anything about our Book 💡📖</div>
          </div>
        </div>
        <div className="header-controls">
          <button 
            className="control-button" 
            onClick={onClose}
            aria-label="Close chat"
            title="Close chat"
          >
            ✕
          </button>
        </div>
      </div>

      {/* Chat Body */}
      <div className="chat-body">
        <div className="chat-messages">
          <MessageList messages={messages} />
          {loading && <LoadingIndicator visible={true} message="AI is thinking..." />}
          {error && (
            <div className="error-message">
              Error: {error}
            </div>
          )}
        </div>
        
        <div className="chat-input-form">
          <ChatInput
            onSendMessage={handleSendMessage}
            disabled={loading}
            placeholder={loading ? "Processing..." : "Type your message..."}
            selectedText={selectedText}
          />
        </div>
      </div>
    </div>
  );
};

export default ChatContainer;