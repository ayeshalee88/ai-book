import React, { useState } from 'react';

interface MessageBubbleProps {
  text: string;
  sender: 'user' | 'assistant';
  timestamp?: Date;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ text, sender, timestamp }) => {
  const [showSources, setShowSources] = useState(false);
  const isUser = sender === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-5 group`}>
      {/* AI avatar */}
      {!isUser && (
        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-500 to-blue-600 flex-shrink-0 mr-3 shadow-sm flex items-center justify-center">
          <img src="/img/chatbot-logo.svg" alt="AI" className="w-6 h-6" />
        </div>
      )}

      {/* Bubble */}
      <div className="max-w-[75%]">
        <div
          className={`
            px-5 py-3.5 rounded-2xl shadow-sm transition-all duration-200
            ${isUser
              ? 'bg-blue-600 text-white rounded-br-md'
              : 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100 border border-gray-200 dark:border-gray-700 rounded-bl-md'
            }
            group-hover:shadow-md
          `}
        >
          <div className="text-[15px] leading-relaxed whitespace-pre-wrap">
            {text}
          </div>

          {/* Sources toggle */}
          {!isUser && (
            <button
              onClick={() => setShowSources(!showSources)}
              className="
                mt-3 text-xs font-medium
                text-blue-600 hover:text-blue-800
                dark:text-blue-400 dark:hover:text-blue-300
                transition-colors
              "
            >
              {showSources ? 'Hide Sources' : 'Show Sources'}
            </button>
          )}

          {/* Sources area */}
          {showSources && !isUser && (
            <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 text-sm text-gray-600 dark:text-gray-400">
              Sources will appear here...
            </div>
          )}
        </div>

        {/* Timestamp */}
        {timestamp && (
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-1.5 opacity-80">
            {timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </div>
        )}
      </div>

      {/* User avatar */}
      {isUser && (
        <div className="w-10 h-10 rounded-full bg-gray-300 dark:bg-gray-600 flex-shrink-0 ml-3 shadow-sm flex items-center justify-center text-lg">
          👤
        </div>
      )}
    </div>
  );
};

export default MessageBubble;