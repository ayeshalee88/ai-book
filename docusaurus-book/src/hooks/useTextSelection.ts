import { useState, useEffect, useCallback } from 'react';

interface TextSelectionResult {
  selectedText: string;
  clearSelection: () => void;
}

const useTextSelection = (): TextSelectionResult => {
  const [selectedText, setSelectedText] = useState('');

  const handleSelection = useCallback(() => {
    const selection = window.getSelection();
    const text = selection?.toString().trim() || '';

    if (text) {
      setSelectedText(text);
    }
  }, []);

  useEffect(() => {
    // Add event listener for mouseup to capture text selection
    document.addEventListener('mouseup', handleSelection);

    // Cleanup function to remove event listener
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, [handleSelection]);

  const clearSelection = useCallback(() => {
    setSelectedText('');
  }, []);

  return { selectedText, clearSelection };
};

export default useTextSelection;