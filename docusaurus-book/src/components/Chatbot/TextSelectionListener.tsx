import { useEffect, useRef } from "react";

interface TextSelectionListenerProps {
  onTextSelected: (text: string) => void;
}

const TextSelectionListener: React.FC<TextSelectionListenerProps> = ({ onTextSelected }) => {
  const lastTextRef = useRef("");

  useEffect(() => {
    const handleSelection = () => {
      // Mobile browsers need delay after long-press
      setTimeout(() => {
        const selection = window.getSelection();
        if (!selection) return;

        const text = selection.toString().trim();
        if (!text || text === lastTextRef.current) return;

        lastTextRef.current = text;
        onTextSelected(text);

        // Clear selection safely (mobile-friendly)
        setTimeout(() => {
          window.getSelection()?.removeAllRanges();
        }, 500);
      }, 300);
    };

    // Desktop + Mobile events
    document.addEventListener("mouseup", handleSelection);
    document.addEventListener("touchend", handleSelection);
    document.addEventListener("pointerup", handleSelection);

    return () => {
      document.removeEventListener("mouseup", handleSelection);
      document.removeEventListener("touchend", handleSelection);
      document.removeEventListener("pointerup", handleSelection);
    };
  }, [onTextSelected]);

  return null;
};

export default TextSelectionListener;
