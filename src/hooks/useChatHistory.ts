import { useState, useEffect } from 'react';
import { Message } from '../components/Chatbot/types';

const CHAT_HISTORY_KEY = 'chatbot-history';
const MAX_HISTORY_SIZE = 100; // Limit the number of stored messages

interface ChatHistoryState {
  messages: Message[];
  addMessage: (message: Message) => void;
  clearHistory: () => void;
  updateMessage: (index: number, message: Message) => void;
}

export const useChatHistory = (): ChatHistoryState => {
  const [messages, setMessages] = useState<Message[]>([]);

  // Load chat history from localStorage on component mount
  useEffect(() => {
    try {
      const storedHistory = localStorage.getItem(CHAT_HISTORY_KEY);
      if (storedHistory) {
        const parsedMessages: Message[] = JSON.parse(storedHistory);
        // Validate that the parsed data has the correct structure
        if (Array.isArray(parsedMessages) && parsedMessages.every(msg =>
          typeof msg.text === 'string' &&
          ['user', 'assistant'].includes(msg.sender) &&
          msg.timestamp instanceof Date
        )) {
          setMessages(parsedMessages.map(msg => ({
            ...msg,
            timestamp: new Date(msg.timestamp) // Convert string back to Date
          })));
        } else {
          // If the stored data is invalid, clear it
          localStorage.removeItem(CHAT_HISTORY_KEY);
        }
      }
    } catch (error) {
      console.error('Error loading chat history from localStorage:', error);
      // If there's an error, clear the stored data
      localStorage.removeItem(CHAT_HISTORY_KEY);
    }
  }, []);

  // Save chat history to localStorage whenever messages change
  useEffect(() => {
    try {
      // Limit the history size to prevent localStorage from getting too large
      const messagesToStore = messages.slice(-MAX_HISTORY_SIZE);
      localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(messagesToStore));
    } catch (error) {
      console.error('Error saving chat history to localStorage:', error);
    }
  }, [messages]);

  const addMessage = (message: Message) => {
    setMessages(prev => [...prev, message]);
  };

  const clearHistory = () => {
    setMessages([]);
    try {
      localStorage.removeItem(CHAT_HISTORY_KEY);
    } catch (error) {
      console.error('Error clearing chat history from localStorage:', error);
    }
  };

  const updateMessage = (index: number, message: Message) => {
    setMessages(prev => {
      const newMessages = [...prev];
      newMessages[index] = message;
      return newMessages;
    });
  };

  return {
    messages,
    addMessage,
    clearHistory,
    updateMessage,
  };
};