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

const ChatContainer: React.FC<ChatContainerProps> = ({
  initialMessages = [],
  onClose,
  }) => {
  const { messages, addMessage } = useChatHistory();
  const [selectedText, setSelectedText] = useState('');

  const API_URL =
    process.env.REACT_APP_API_URL ??
    'https://ayishaalee-rag-chatbot-api-v2.hf.space';

  const { loading, error, execute } = useApi(`${API_URL}/chat`);

  const handleSendMessage = useCallback(
    async (text: string) => {
      addMessage({
        text,
        sender: 'user',
        timestamp: new Date(),
      });

      try {
        const response = await execute({
          message: text,
          top_k: 3,
        });

        addMessage({
          text: response.response,
          sender: 'assistant',
          timestamp: new Date(),
          sources: response.sources ?? [],
        });
      } catch (err) {
        addMessage({
          text: 'Sorry, I encountered an error processing your request.',
          sender: 'assistant',
          timestamp: new Date(),
        });
      }
    },
    [execute, addMessage]
  );

  const handleTextSelected = useCallback((text: string) => {
    setSelectedText(text);
  }, []);

  return (
    <div className="chat-container-full">
      <TextSelectionListener onTextSelected={handleTextSelected} />

      <div className="chat-header">
        <div className="header-content">
          <div className="logo-placeholder">🤖</div>
          <div className="header-text">
            <h3>AYI BUDDY</h3>
            <div className="header-subtitle">
              Ask anything about our Book 💡📖
            </div>
          </div>
        </div>

        <div className="header-controls">
          <button onClick={onClose} aria-label="Close chat">✕</button>
        </div>
      </div>

      <div className="chat-body">
        <MessageList messages={messages} />

        {loading && <LoadingIndicator visible message="AI is thinking..." />}

        {error && <div className="error-message">Error: {String(error)}</div>}

        <ChatInput
          onSendMessage={handleSendMessage}
          disabled={loading}
          selectedText={selectedText}
          placeholder="Type your message..."
        />
      </div>
    </div>
  );
};

export default ChatContainer;
