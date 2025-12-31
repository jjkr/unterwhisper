import { useState, useEffect } from "react";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import "./TranscriptionWindow.css";

function TranscriptionWindow() {
  const [transcriptionText, setTranscriptionText] = useState("");
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const appWindow = getCurrentWindow();

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
      appWindow.show();
    });

    // Listen for hide window events
    const unlistenHide = listen("hide-window", () => {
      setIsVisible(false);
      appWindow.hide();
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
