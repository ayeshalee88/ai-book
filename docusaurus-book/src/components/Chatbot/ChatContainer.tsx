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
  const { loading, error, execute, reset } = useApi('http://127.0.0.1:8000/api/v1/qa');
  console.log('ChatContainer: FORCING API URL TO → http://127.0.0.1:8000/api/v1/qa');

  const handleSendMessage = useCallback(async (text: string) => {
    // Add user message to the chat
    const userMessage: Message = {
      text,
      sender: 'user',
      timestamp: new Date(),
    };

    addMessage(userMessage);

    try {
      // Call the API to get the response
      const response = await execute(text);

      // Add assistant message to the chat
      const assistantMessage: Message = {
        text: response.answer,
        sender: 'assistant',
        timestamp: new Date(),
      };

      addMessage(assistantMessage);
    } catch (err) {
      console.error('Error getting response:', err);

      // Add error message to the chat
      const errorMessage: Message = {
        text: 'Sorry, I encountered an error processing your request.',
        sender: 'assistant',
        timestamp: new Date(),
      };

      addMessage(errorMessage);
    }
  }, [execute, addMessage]);

  const handleTextSelected = useCallback((text: string) => {
    // Set the selected text so it can be injected into the input field
    setSelectedText(text);
  }, []);

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