# Implementation Plan: Microphone Input Device Selection

## Overview

This implementation plan breaks down the microphone device selection feature into discrete coding tasks. The approach follows a bottom-up strategy: first extending the backend audio and settings infrastructure, then adding Tauri commands, and finally implementing the frontend UI components. Each task builds on previous work to ensure incremental progress with no orphaned code.

## Tasks

- [x] 1. Extend audio device management in Rust backend
  - Add DeviceId enum and AudioDeviceInfo struct to audio.rs
  - Implement list_input_devices_with_info() function
  - Implement find_device_by_id() and get_default_device() functions
  - Implement from_device_id() constructor for AudioRecorder
  - _Requirements: 1.1, 1.2, 1.3, 1.5, 4.1, 4.4_

- [ ]* 1.1 Write property test for device enumeration completeness
  - **Property 1: Device Enumeration Completeness**
  - **Validates: Requirements 1.1, 1.2, 1.3**

- [ ]* 1.2 Write property test for device input validation
  - **Property 10: Device Input Validation**
  - **Validates: Requirements 4.4**

- [ ]* 1.3 Write property test for enumeration robustness
  - **Property 11: Enumeration Robustness**
  - **Validates: Requirements 5.5, 8.1**

- [ ]* 1.4 Write unit tests for device management functions
  - Test get_default_device() returns a valid device
  - Test find_device_by_id() with SystemDefault
  - Test find_device_by_id() with specific device name
  - Test error handling for nonexistent device
  - _Requirements: 1.1, 1.2, 1.3, 4.1_

- [x] 2. Extend Settings struct with device selection
  - Add device_id field to Settings struct with #[serde(default)]
  - Update Default implementation to include DeviceId::SystemDefault
  - Implement validate_device() method
  - Implement get_device_name() method
  - _Requirements: 3.1, 3.2, 3.4, 3.5, 7.5_

- [ ]* 2.1 Write property test for settings persistence round-trip
  - **Property 6: Settings Persistence Round-Trip**
  - **Validates: Requirements 3.1, 3.2, 3.4**

- [ ]* 2.2 Write property test for default fallback behavior
  - **Property 7: Default Fallback Behavior**
  - **Validates: Requirements 3.3, 3.5, 9.1, 9.3**

- [ ]* 2.3 Write property test for SystemDefault serialization
  - **Property 12: SystemDefault Serialization**
  - **Validates: Requirements 6.3**

- [ ]* 2.4 Write property test for device name display
  - **Property 14: Device Name Display**
  - **Validates: Requirements 7.1, 7.5**

- [ ]* 2.5 Write property test for settings migration preservation
  - **Property 18: Settings Migration Preservation**
  - **Validates: Requirements 9.2, 9.4**

- [ ]* 2.6 Write unit tests for Settings struct
  - Test serialization of DeviceId::SystemDefault
  - Test serialization of DeviceId::Specific
  - Test deserialization of old config without device_id
  - Test get_device_name() for both variants
  - _Requirements: 3.1, 3.4, 7.5, 9.1, 9.2_

- [x] 3. Add Tauri command for device enumeration
  - Implement get_audio_devices() Tauri command
  - Add command to invoke_handler in lib.rs
  - Handle errors and return empty list on failure
  - _Requirements: 1.1, 1.2, 1.3, 8.1_

- [ ]* 3.1 Write unit test for get_audio_devices command
  - Test that command returns Vec<AudioDeviceInfo>
  - Test error handling returns empty list
  - _Requirements: 1.1, 8.1_

- [x] 4. Update recording logic to use selected device
  - Modify start_recording() to read device_id from settings
  - Implement fallback to system default on device error
  - Emit "device-fallback" event when fallback occurs
  - Emit "device-error" event when device initialization fails
  - _Requirements: 4.1, 4.2, 4.5, 8.2, 8.4_

- [ ]* 4.1 Write property test for device selection integration
  - **Property 8: Device Selection Integration**
  - **Validates: Requirements 4.1**

- [ ]* 4.2 Write property test for fallback with notification
  - **Property 9: Fallback with Notification**
  - **Validates: Requirements 4.2, 4.5**

- [ ]* 4.3 Write property test for SystemDefault device query
  - **Property 13: SystemDefault Device Query**
  - **Validates: Requirements 6.2**

- [ ]* 4.4 Write property test for device initialization error handling
  - **Property 17: Device Initialization Error Handling**
  - **Validates: Requirements 8.2, 8.4**

- [ ]* 4.5 Write unit tests for recording with device selection
  - Test recording starts with valid device_id
  - Test fallback to default when device unavailable
  - Test error notification emitted on device failure
  - _Requirements: 4.1, 4.2, 4.5, 8.2_

- [x] 5. Checkpoint - Ensure all backend tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Create DeviceSelector React component
  - Create DeviceSelector.tsx component file
  - Implement device list loading via get_audio_devices command
  - Implement dropdown rendering with System Default option first
  - Implement onChange handler to update DeviceId
  - Handle loading and error states
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 6.1, 6.5_

- [ ]* 6.1 Write property test for UI device display
  - **Property 3: UI Device Display**
  - **Validates: Requirements 2.2**

- [ ]* 6.2 Write property test for UI selection update
  - **Property 4: UI Selection Update**
  - **Validates: Requirements 2.3**

- [ ]* 6.3 Write property test for UI selection state
  - **Property 5: UI Selection State**
  - **Validates: Requirements 2.4**

- [ ]* 6.4 Write unit tests for DeviceSelector component
  - Test System Default option appears first
  - Test all devices rendered as options
  - Test onChange called with correct DeviceId
  - Test empty device list shows message
  - Test error state displays retry button
  - _Requirements: 2.1, 2.2, 2.3, 2.5, 6.1, 6.5_

- [x] 7. Create Settings dialog component
  - Create SettingsDialog.tsx component file
  - Integrate DeviceSelector component
  - Implement settings loading via get_settings command
  - Implement settings saving via update_settings command
  - Add model and language selectors (if not already present)
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1_

- [ ]* 7.1 Write unit tests for SettingsDialog component
  - Test settings load on mount
  - Test device selection updates settings state
  - Test save button calls update_settings
  - _Requirements: 2.1, 3.1_

- [ ] 8. Add device fallback notification handling
  - Listen for "device-fallback" event in frontend
  - Display notification with fallback message
  - Listen for "device-error" event in frontend
  - Display error dialog with device name
  - _Requirements: 4.5, 7.3, 8.2, 8.4_

- [ ]* 8.1 Write property test for fallback notification content
  - **Property 16: Fallback Notification Content**
  - **Validates: Requirements 7.3**

- [ ]* 8.2 Write unit tests for notification handling
  - Test device-fallback event displays notification
  - Test device-error event displays error dialog
  - _Requirements: 4.5, 7.3, 8.2_

- [ ] 9. Add unavailable device indication in UI
  - Check if selected device is in available devices list
  - Display warning icon/text if device not found
  - Suggest selecting a different device
  - _Requirements: 7.4_

- [ ]* 9.1 Write property test for unavailable device indication
  - **Property 15: Unavailable Device Indication**
  - **Validates: Requirements 7.4**

- [ ]* 9.2 Write unit test for unavailable device UI
  - Test warning displayed when device not in list
  - _Requirements: 7.4_

- [ ] 10. Add permission handling in device enumeration
  - Handle empty device list gracefully
  - Display "No devices - check permissions" message
  - Add retry button to re-enumerate devices
  - _Requirements: 8.3, 10.2, 10.4_

- [ ]* 10.1 Write property test for permission handling
  - **Property 19: Permission Handling**
  - **Validates: Requirements 10.2, 10.4**

- [ ]* 10.2 Write unit test for permission error UI
  - Test empty device list shows permission message
  - Test retry button calls get_audio_devices again
  - _Requirements: 8.3, 10.2_

- [ ] 11. Add CSS styling for device selector components
  - Style DeviceSelector dropdown
  - Style SettingsDialog layout
  - Style error and warning messages
  - Ensure accessibility (focus states, labels)
  - _Requirements: 2.1, 2.5_

- [ ] 12. Final checkpoint - Integration testing
  - Test complete flow: open settings → select device → save → start recording
  - Test fallback flow: select invalid device → start recording → verify fallback
  - Test migration: delete device_id from config → restart app → verify default
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- The implementation follows a bottom-up approach: backend → commands → frontend
- Checkpoints ensure incremental validation of functionality
