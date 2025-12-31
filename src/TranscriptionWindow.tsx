import { useState, useEffect } from "react";
import { listen } from "@tauri-apps/api/event";
import "./TranscriptionWindow.css";

function TranscriptionWindow() {
  const [transcriptionText, setTranscriptionText] = useState("");
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    // Listen for transcription updates
    const unlistenTranscription = listen<string>(
      "transcription-update",
      (event) => {
        setTranscriptionText(event.payload);
      }
    );

    // Listen for show window events
    const unlistenShow = listen("show-window", () => {
      setIsVisible(true);
      setTranscriptionText("");
    });

    // Listen for hide window events
    const unlistenHide = listen("hide-window", () => {
      setIsVisible(false);
    });

    // Cleanup listeners on unmount
    return () => {
      unlistenTranscription.then((fn) => fn());
      unlistenShow.then((fn) => fn());
      unlistenHide.then((fn) => fn());
    };
  }, []);

  if (!isVisible) {
    return null;
  }

  return (
    <div className="transcription-container">
      <div className="transcription-content">
        {transcriptionText || "Listening..."}
      </div>
    </div>
  );
}

export default TranscriptionWindow;
