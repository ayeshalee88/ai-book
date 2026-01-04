import React, { useState, useRef, useEffect } from 'react';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  placeholder?: string;
  disabled?: boolean;
  selectedText?: string;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, placeholder = 'Type your message...', disabled = false, selectedText }) => {
  const [inputValue, setInputValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Handle selected text injection
  useEffect(() => {
    if (selectedText && !disabled) {
      setInputValue(prev => prev + selectedText + ' ');
    }
  }, [selectedText, disabled]);

  useEffect(() => {
    if (textareaRef.current) {
      // Auto-resize textarea based on content
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`;
    }
  }, [inputValue]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim() && !disabled) {
      onSendMessage(inputValue.trim());
      setInputValue('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(e.target.value);
  };

  // Function to inject text from selection
  const injectText = (text: string) => {
    if (!disabled) {
      setInputValue(prev => prev + text + ' ');
    }
  };

  return (
    <form className="chat-input-form" onSubmit={handleSubmit}>
      <div className="input-container">
        <textarea
          ref={textareaRef}
          value={inputValue}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          rows={1}
          className="chat-input-textarea"
        />
        <button
          type="submit"
          disabled={inputValue.trim() === '' || disabled}
          className="send-button"
          aria-label="Send message"
        >
          <span>➤</span>
        </button>
      </div>
    </form>
  );
};

export default ChatInput;