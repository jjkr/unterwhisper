# Requirements Document

## Introduction

This document specifies the requirements for adding microphone input device selection to the background ASR application. Users need the ability to choose which audio input device to use for transcription, rather than being limited to the system default microphone.

## Glossary

- **Audio_Device_Manager**: The component that enumerates and manages audio input devices
- **Settings_Manager**: The component that manages application settings persistence
- **Audio_Capture**: The component that captures microphone audio input
- **Device_Selector_UI**: The user interface component for selecting audio input devices
- **ASR_System**: The automatic speech recognition system that converts audio to text

## Requirements

### Requirement 1: Audio Device Enumeration

**User Story:** As a user, I want to see a list of available microphone devices, so that I can choose which one to use for transcription.

#### Acceptance Criteria

1. WHEN the settings dialog opens, THE Audio_Device_Manager SHALL enumerate all available audio input devices
2. THE Audio_Device_Manager SHALL retrieve the device name for each available input device
3. THE Audio_Device_Manager SHALL retrieve the device identifier for each available input device
4. WHEN no audio input devices are available, THE Audio_Device_Manager SHALL return an empty list
5. THE Audio_Device_Manager SHALL refresh the device list when the settings dialog is opened

### Requirement 2: Device Selection Interface

**User Story:** As a user, I want to select my preferred microphone from a dropdown menu, so that I can easily switch between input devices.

#### Acceptance Criteria

1. WHEN the settings dialog displays, THE Device_Selector_UI SHALL show a dropdown list of available audio input devices
2. THE Device_Selector_UI SHALL display each device by its human-readable name
3. WHEN the user selects a device from the dropdown, THE Device_Selector_UI SHALL update the selection
4. THE Device_Selector_UI SHALL indicate the currently selected device
5. WHEN no devices are available, THE Device_Selector_UI SHALL display a message indicating no input devices were found

### Requirement 3: Device Selection Persistence

**User Story:** As a user, I want my microphone selection to be remembered, so that I don't have to reconfigure it every time I restart the app.

#### Acceptance Criteria

1. WHEN the user selects a device, THE Settings_Manager SHALL persist the device identifier to the configuration file
2. WHEN the application starts, THE Settings_Manager SHALL load the saved device identifier from the configuration file
3. WHEN the saved device identifier is not found in available devices, THE Settings_Manager SHALL fall back to the system default device
4. THE Settings_Manager SHALL store the device identifier in the existing JSON configuration structure
5. WHEN no device has been previously selected, THE Settings_Manager SHALL use the system default device

### Requirement 4: Audio Capture Device Configuration

**User Story:** As a user, I want the app to use my selected microphone for recording, so that transcription captures audio from the correct source.

#### Acceptance Criteria

1. WHEN recording starts, THE Audio_Capture SHALL use the device identifier from Settings_Manager
2. WHEN the selected device is not available, THE Audio_Capture SHALL fall back to the system default device
3. WHEN the selected device becomes unavailable during recording, THE Audio_Capture SHALL stop recording and display an error notification
4. THE Audio_Capture SHALL validate that the selected device supports audio input before starting recording
5. WHEN using the fallback device, THE ASR_System SHALL display a notification informing the user

### Requirement 5: Device Change Handling

**User Story:** As a user, I want the app to handle device changes gracefully, so that I can plug/unplug devices without crashes.

#### Acceptance Criteria

1. WHEN an audio device is connected or disconnected, THE Audio_Device_Manager SHALL detect the change
2. WHEN the currently selected device is disconnected, THE Audio_Capture SHALL fall back to the system default device
3. WHEN a device change occurs during recording, THE Audio_Capture SHALL stop recording and notify the user
4. WHEN the settings dialog is open and devices change, THE Device_Selector_UI SHALL refresh the device list
5. THE Audio_Device_Manager SHALL handle device changes without crashing the application

### Requirement 6: Default Device Handling

**User Story:** As a user, I want an option to always use the system default device, so that the app automatically switches when I change my system audio settings.

#### Acceptance Criteria

1. THE Device_Selector_UI SHALL include a "System Default" option in the device dropdown
2. WHEN "System Default" is selected, THE Audio_Capture SHALL query the system for the current default input device at recording start
3. WHEN "System Default" is selected, THE Settings_Manager SHALL store a special identifier indicating default device preference
4. WHEN the system default device changes, THE Audio_Capture SHALL use the new default device for subsequent recordings
5. THE Device_Selector_UI SHALL display "System Default" as the first option in the dropdown

### Requirement 7: Device Information Display

**User Story:** As a user, I want to see which device is currently active, so that I can verify the correct microphone is being used.

#### Acceptance Criteria

1. THE Device_Selector_UI SHALL display the currently selected device name in the settings dialog
2. WHEN recording starts, THE ASR_System SHALL log the active device name
3. WHEN a fallback device is used, THE ASR_System SHALL display a notification showing which device is being used
4. THE Device_Selector_UI SHALL indicate if a previously selected device is no longer available
5. THE Settings_Manager SHALL provide an API to query the currently active device name

### Requirement 8: Error Handling

**User Story:** As a user, I want clear error messages when device issues occur, so that I can troubleshoot problems.

#### Acceptance Criteria

1. WHEN device enumeration fails, THE Audio_Device_Manager SHALL log the error and return an empty list
2. WHEN the selected device cannot be opened, THE Audio_Capture SHALL display an error notification with the device name
3. WHEN no audio input devices are available, THE ASR_System SHALL display a notification explaining the issue
4. IF device initialization fails during recording start, THEN THE Audio_Capture SHALL prevent recording and notify the user
5. THE ASR_System SHALL provide actionable error messages that guide users toward resolution

### Requirement 9: Settings Migration

**User Story:** As an existing user, I want my app to continue working after the update, so that I don't experience disruption.

#### Acceptance Criteria

1. WHEN the application starts with an old configuration file, THE Settings_Manager SHALL add the device selection field with a default value
2. THE Settings_Manager SHALL preserve all existing settings during migration
3. WHEN no device preference exists in the configuration, THE Settings_Manager SHALL default to "System Default"
4. THE Settings_Manager SHALL validate the configuration structure and handle missing fields gracefully
5. WHEN migration occurs, THE Settings_Manager SHALL not require user intervention

### Requirement 10: Platform-Specific Integration

**User Story:** As a macOS user, I want the device selection to use native macOS audio APIs, so that it integrates properly with the system.

#### Acceptance Criteria

1. THE Audio_Device_Manager SHALL use macOS Core Audio APIs to enumerate devices
2. THE Audio_Device_Manager SHALL respect macOS audio device permissions
3. THE Audio_Capture SHALL use the device identifier format expected by macOS Core Audio
4. WHEN microphone permissions are denied, THE Audio_Device_Manager SHALL return an empty device list
5. THE Audio_Device_Manager SHALL handle macOS-specific device properties correctly
