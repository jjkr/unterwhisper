# Unterwhisper

A background automatic speech recognition (ASR) application for macOS that runs in the system tray. Press a global hotkey to start recording, see live transcription in a popup window, and have the text automatically pasted into your active application.

## Features

- 🎤 **Global Hotkey**: Press `Cmd+Shift+Space` to start/stop recording from anywhere
- 🪟 **Live Transcription**: See real-time transcription in a floating window
- 📋 **Auto-Paste**: Transcribed text is automatically pasted into your active application
- 🔧 **Configurable**: Choose your Whisper model and language settings
- 🦉 **Background App**: Runs quietly in the system tray, no dock icon

## Setup

### Prerequisites

- macOS 10.13 or later
- Node.js and pnpm
- Rust toolchain

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pnpm install
   ```

3. Build and run:
   ```bash
   pnpm tauri dev
   ```

### Required Permissions

For the app to work properly, you need to grant the following macOS permissions:

1. **Microphone Access**: Required to capture audio for transcription
   - Go to: System Settings > Privacy & Security > Microphone
   - Enable access for Unterwhisper

2. **Accessibility Access**: Required for global hotkeys and auto-paste
   - Go to: System Settings > Privacy & Security > Accessibility
   - Enable access for Unterwhisper
   - **This is critical for global hotkeys to work!**

The app will show notifications if permissions are missing.

## Usage

1. **Launch the app**: It will appear in your system tray (menu bar)
2. **Start recording**: 
   - Press `Cmd+Shift+Space` (if Accessibility permission is granted)
   - Or click the tray icon and select "Start Recording"
3. **Speak**: The transcription window will appear showing live transcription
4. **Stop recording**: Release the hotkey or press it again
5. **Auto-paste**: The transcribed text will be automatically pasted into your active application

## Troubleshooting

### Global hotkey not working
- Make sure Accessibility permissions are granted (see Required Permissions above)
- Check if another app is using the same hotkey
- You can always use the tray menu as a fallback

### No transcription window appears
- This is a background app - no window appears on launch (this is normal!)
- The transcription window only appears when you start recording
- Check the system tray for the app icon

### "No keyWindow" error
- This typically means Accessibility permissions are not granted
- Grant the permission and restart the app

## Development

### Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)

### Project Structure

- `src/` - React frontend
- `src-tauri/` - Rust backend
  - `src/asr/` - Audio capture and Whisper transcription
  - `src/settings.rs` - Settings management
  - `src/lib.rs` - Main application logic

## License

[Add your license here]
