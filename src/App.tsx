import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";
import TranscriptionWindow from "./TranscriptionWindow";
import { SettingsDialog } from "./SettingsDialog";
import "./App.css";

function App() {
  const [windowLabel, setWindowLabel] = useState("");
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  useEffect(() => {
    const getLabel = async () => {
      const window = getCurrentWindow();
      setWindowLabel(window.label);
      
      // If this is the main window, automatically open settings
      if (window.label === "main") {
        setIsSettingsOpen(true);
        
        // Listen for window close event to hide from dock
        const unlisten = await window.onCloseRequested(async (event) => {
          // Prevent default close behavior
          event.preventDefault();
          
          // Hide the window instead of closing
          await window.hide();
          
          // Hide app from dock by invoking a backend command
          try {
            await invoke("hide_from_dock");
          } catch (e) {
            console.error("Failed to hide from dock:", e);
          }
        });
        
        return () => {
          unlisten();
        };
      }
    };
    getLabel();
  }, []);

  // Render transcription window if this is the transcription window
  if (windowLabel === "transcription") {
    return <TranscriptionWindow />;
  }

  // Otherwise render the main app (settings window)
  return (
    <main className="container">
      {/* Settings Dialog - auto-opens when window is shown */}
      <SettingsDialog 
        isOpen={isSettingsOpen} 
        onClose={() => {
          setIsSettingsOpen(false);
          // Close the window when settings dialog is closed
          getCurrentWindow().hide();
          invoke("hide_from_dock").catch(e => console.error("Failed to hide from dock:", e));
        }} 
      />
    </main>
  );
}

export default App;
