import React, { useEffect } from 'react';
import useTextSelection from '../../hooks/useTextSelection';

interface TextSelectionListenerProps {
  onTextSelected: (text: string) => void;
}

const TextSelectionListener: React.FC<TextSelectionListenerProps> = ({ onTextSelected }) => {
  const { selectedText, clearSelection } = useTextSelection();

  useEffect(() => {
    if (selectedText) {
      onTextSelected(selectedText);
      // Clear the selection after processing
      clearSelection();
    }
  }, [selectedText, onTextSelected, clearSelection]);

  return null; // This component doesn't render anything
};

export default TextSelectionListener;