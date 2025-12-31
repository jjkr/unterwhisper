# Implementation Plan: Background Transcription App

## Overview

This plan implements a minimal viable product (MVP) for a background transcription app on macOS. The implementation leverages the existing ASR module and adds system tray integration, global hotkey support, and automatic text pasting.

## Tasks

- [x] 1. Set up project dependencies and configuration
  - Add required Cargo dependencies (arboard, enigo, dirs, anyhow)
  - Update tauri.conf.json with transcription window configuration
  - Add system tray icon configuration
  - Update Info.plist with microphone and accessibility permissions
  - _Requirements: 1.1, 2.1, 8.1, 8.2_

- [x] 2. Implement settings module
  - [x] 2.1 Create Settings struct and default implementation
    - Define Settings struct with model and language fields
    - Implement Default trait with sensible defaults
    - _Requirements: 6.2, 6.3, 6.4, 7.2_

  - [x] 2.2 Implement settings persistence functions
    - Implement load() function to read from config file
    - Implement save() function to write to config file
    - Handle config directory creation
    - _Requirements: 6.5, 6.6, 7.1, 7.5_

  - [ ]* 2.3 Write property test for settings round-trip
    - **Property 1: Settings Persistence Round-Trip**
    - **Validates: Requirements 6.5, 6.6, 7.2**

  - [ ]* 2.4 Write unit tests for settings error handling
    - Test corrupted JSON handling
    - Test missing config file handling
    - _Requirements: 7.3, 7.4_

- [x] 3. Implement core application state and setup
  - [x] 3.1 Create AppState struct
    - Define AppState with transcriber, is_recording, and settings fields
    - Use Arc<Mutex<>> for thread-safe access
    - _Requirements: 4.1, 4.4_

  - [x] 3.2 Implement Tauri setup function
    - Initialize AppState with default settings
    - Load settings from config file
    - Set up system tray with menu items
    - Register global hotkey (Cmd+Shift+Space)
    - _Requirements: 1.1, 1.2, 1.3, 2.1, 10.1_

  - [x] 3.3 Implement system tray menu handlers
    - Handle "Start Recording" menu item
    - Handle "Quit" menu item
    - _Requirements: 1.3, 1.4, 10.4_

- [ ] 4. Implement hotkey handling
  - [ ] 4.1 Create hotkey event polling loop
    - Spawn background thread to poll hotkey events
    - Handle press event (start recording)
    - Handle release event (stop recording)
    - _Requirements: 2.2, 2.3, 2.5_

  - [ ] 4.2 Implement hotkey error handling
    - Handle registration failures gracefully
    - Display error notifications if hotkey conflicts
    - _Requirements: 2.4, 9.2_

- [ ] 5. Implement recording control functions
  - [ ] 5.1 Implement start_recording function
    - Create RealtimeTranscriber with settings
    - Start transcriber
    - Set is_recording flag to true
    - Emit "show-window" event to frontend
    - _Requirements: 3.1, 4.1, 4.2_

  - [ ] 5.2 Implement stop_recording function
    - Stop transcriber
    - Get final transcription text
    - Set is_recording flag to false
    - Return transcription text
    - _Requirements: 4.4_

  - [ ] 5.3 Implement transcription polling loop
    - Spawn background thread to poll transcription updates
    - Emit "transcription-update" events to frontend
    - Continue until recording stops
    - _Requirements: 3.2, 3.3, 4.3_

  - [ ]* 5.4 Write property test for recording state consistency
    - **Property 2: Recording State Consistency**
    - **Validates: Requirements 4.1, 4.4**

  - [ ]* 5.5 Write property test for transcription text accumulation
    - **Property 3: Transcription Text Accumulation**
    - **Validates: Requirements 3.2, 3.3**

- [ ] 6. Implement clipboard and paste functionality
  - [ ] 6.1 Implement copy_and_paste function
    - Copy transcription text to clipboard
    - Simulate Cmd+V keypress using enigo
    - Handle clipboard errors gracefully
    - _Requirements: 5.1, 5.2, 5.5_

  - [ ] 6.2 Integrate paste into stop_recording flow
    - Call copy_and_paste after getting final transcription
    - Emit "hide-window" event after paste
    - Add 1 second delay before hiding window
    - _Requirements: 3.4, 5.1_

  - [ ]* 6.3 Write unit tests for clipboard operations
    - Test copy and paste with various text inputs
    - Test error handling for clipboard failures
    - _Requirements: 5.4_

- [ ] 7. Implement window control functions
  - [ ] 7.1 Implement show_window and hide_window functions
    - Get window by label
    - Call show() or hide() on window
    - Handle window not found errors
    - _Requirements: 3.1, 3.4_

  - [ ]* 7.2 Write unit tests for window control
    - Test show/hide operations
    - Test error handling for missing windows

- [ ] 8. Implement Tauri commands
  - [ ] 8.1 Implement get_settings command
    - Return current settings from AppState
    - _Requirements: 6.6_

  - [ ] 8.2 Implement update_settings command
    - Update settings in AppState
    - Save settings to config file
    - _Requirements: 6.5_

  - [ ] 8.3 Implement manual recording control commands
    - Implement manual_start_recording command
    - Implement manual_stop_recording command
    - _Requirements: 1.3_

- [ ] 9. Implement React transcription window
  - [ ] 9.1 Create TranscriptionWindow component
    - Display transcription text with proper styling
    - Semi-transparent background
    - Auto-sizing based on text length
    - _Requirements: 3.2, 3.6_

  - [ ] 9.2 Set up event listeners
    - Listen for "transcription-update" events
    - Listen for "show-window" events
    - Listen for "hide-window" events
    - Update component state accordingly
    - _Requirements: 3.2, 3.3_

- [ ] 10. Implement error handling and notifications
  - [ ] 10.1 Add microphone permission check
    - Check permissions before starting recording
    - Display notification if permissions denied
    - _Requirements: 4.5, 9.1_

  - [ ] 10.2 Add error logging
    - Set up log file in Application Support directory
    - Log errors with context
    - _Requirements: 9.5_

  - [ ] 10.3 Add error notifications for common failures
    - Whisper model load failures
    - Recording failures
    - No speech detected
    - _Requirements: 9.2, 9.3, 9.4_

  - [ ]* 10.4 Write property test for config validation
    - **Property 4: Configuration File Validation**
    - **Validates: Requirements 7.3, 7.4**

- [ ] 11. Integration and testing
  - [ ] 11.1 Wire all components together in main.rs
    - Initialize all state
    - Register all Tauri commands
    - Start hotkey polling thread
    - _Requirements: 10.1, 10.2_

  - [ ]* 11.2 Write integration test for end-to-end flow
    - Test hotkey trigger → window show → recording → transcription → paste
    - _Requirements: All_

  - [ ] 11.3 Manual testing on macOS
    - Test with real microphone input
    - Test paste into various applications
    - Test error scenarios
    - _Requirements: All_

- [ ] 12. Final checkpoint
  - Ensure all tests pass
  - Verify app runs without errors
  - Test on clean macOS system
  - Ask user for feedback

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- The existing ASR module (RealtimeTranscriber) handles audio capture and transcription
- Focus on minimal, composable functions rather than complex abstractions
