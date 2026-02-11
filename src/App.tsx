import { useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { SettingsDialog } from "./SettingsDialog";
import "./App.css";

function App() {
  useEffect(() => {
    const setup = async () => {
      const window = getCurrentWindow();

      // Listen for window close event to hide from dock
      const unlistenClose = await window.onCloseRequested(async (event) => {
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
        unlistenClose();
      };
    };
    setup();
  }, []);

  const hideWindow = () => {
    getCurrentWindow().hide();
    invoke("hide_from_dock").catch(e => console.error("Failed to hide from dock:", e));
  };

  // Render the main app (settings window)
  return (
    <main className="container">
      <SettingsDialog onClose={hideWindow} />
    </main>
  );
}

export default App;
