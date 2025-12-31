# Design Document: Microphone Input Device Selection

## Overview

This design extends the Unterwhisper background ASR application to support user-selectable audio input devices. Currently, the application uses the system default microphone. This feature will allow users to choose from available audio input devices through a settings interface, with the selection persisted across application restarts.

The implementation leverages the existing `cpal` (Cross-Platform Audio Library) infrastructure already used for audio capture, extending it with device enumeration and selection capabilities. The design integrates seamlessly with the existing settings management system and Tauri command architecture.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Settings Dialog                                        │ │
│  │  - Device Dropdown (DeviceSelector component)          │ │
│  │  - Model Selection                                      │ │
│  │  - Language Selection                                   │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Tauri Commands
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Backend (Rust/Tauri)                       │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Tauri Commands                                         │ │
│  │  - get_audio_devices()                                  │ │
│  │  - get_settings() [extended]                            │ │
│  │  - update_settings() [extended]                         │ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Settings Manager                                       │ │
│  │  - Load/Save device_id field                            │ │
│  │  - Validate device availability                         │ │
│  │  - Handle migration from old config                     │ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Audio Device Manager (audio.rs)                        │ │
│  │  - list_input_devices() [existing]                      │ │
│  │  - find_device_by_id() [new]                            │ │
│  │  - get_default_device() [new]                           │ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Audio Recorder                                         │ │
│  │  - AudioRecorder::with_device() [existing]              │ │
│  │  - start_streaming() [existing]                         │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ cpal API
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              macOS Core Audio (via cpal)                     │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

1. **Settings Dialog Opens**: Frontend calls `get_audio_devices()` to populate dropdown
2. **User Selects Device**: Frontend calls `update_settings()` with new device_id
3. **Settings Persisted**: Settings Manager saves device_id to config.json
4. **Recording Starts**: Audio Recorder uses device_id from settings to select device
5. **Device Unavailable**: Falls back to system default, notifies user

## Components and Interfaces

### 1. Audio Device Manager (Rust - audio.rs)

Extends the existing `AudioRecorder` implementation with device management functions.

#### New Functions

```rust
/// Device identifier that can be serialized and persisted
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum DeviceId {
    /// Use the system default input device
    SystemDefault,
    /// Use a specific device by its unique identifier
    Specific(String),
}

impl Default for DeviceId {
    fn default() -> Self {
        DeviceId::SystemDefault
    }
}

/// Information about an audio input device
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AudioDeviceInfo {
    /// Unique identifier for the device
    pub id: String,
    /// Human-readable device name
    pub name: String,
    /// Whether this is the system default device
    pub is_default: bool,
}

impl AudioRecorder {
    /// Get the system default input device
    pub fn get_default_device() -> Result<cpal::Device> {
        let host = cpal::default_host();
        host.default_input_device()
            .context("No default input device available")
    }
    
    /// List all available input devices with metadata
    pub fn list_input_devices_with_info() -> Result<Vec<AudioDeviceInfo>> {
        let host = cpal::default_host();
        let default_device = host.default_input_device();
        let default_name = default_device
            .as_ref()
            .and_then(|d| d.description().ok())
            .map(|desc| desc.name().to_string());
        
        let mut devices = Vec::new();
        
        for device in host.input_devices()? {
            if let Ok(desc) = device.description() {
                let name = desc.name().to_string();
                let is_default = Some(&name) == default_name.as_ref();
                
                devices.push(AudioDeviceInfo {
                    id: name.clone(), // Use name as ID for simplicity
                    name,
                    is_default,
                });
            }
        }
        
        Ok(devices)
    }
    
    /// Find a device by its identifier
    pub fn find_device_by_id(device_id: &DeviceId) -> Result<cpal::Device> {
        match device_id {
            DeviceId::SystemDefault => Self::get_default_device(),
            DeviceId::Specific(id) => {
                // Use existing find_device_by_name since we use name as ID
                Self::find_device_by_name(id)
            }
        }
    }
    
    /// Create a recorder with device from settings
    pub fn from_device_id(device_id: &DeviceId) -> Result<Self> {
        let device = Self::find_device_by_id(device_id)?;
        Ok(Self::with_device(device))
    }
}
```

### 2. Settings Manager (Rust - settings.rs)

Extends the existing `Settings` struct to include device selection.

#### Updated Settings Struct

```rust
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Settings {
    pub model: String,
    pub language: Option<String>,
    #[serde(default)]
    pub device_id: DeviceId,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            model: "tiny.en".to_string(),
            language: None,
            device_id: DeviceId::SystemDefault,
        }
    }
}

impl Settings {
    /// Validate that the selected device is available
    pub fn validate_device(&self) -> Result<()> {
        AudioRecorder::find_device_by_id(&self.device_id)?;
        Ok(())
    }
    
    /// Get the device name for display purposes
    pub fn get_device_name(&self) -> String {
        match &self.device_id {
            DeviceId::SystemDefault => "System Default".to_string(),
            DeviceId::Specific(id) => id.clone(),
        }
    }
}
```

The `#[serde(default)]` attribute ensures backward compatibility - old config files without the `device_id` field will automatically use `DeviceId::SystemDefault`.

### 3. Tauri Commands (Rust - lib.rs)

New command for device enumeration, existing commands remain unchanged.

#### New Command

```rust
/// Get list of available audio input devices
#[tauri::command]
fn get_audio_devices() -> Result<Vec<AudioDeviceInfo>, String> {
    info!("Getting available audio input devices");
    
    AudioRecorder::list_input_devices_with_info()
        .map_err(|e| {
            error!("Failed to enumerate audio devices: {}", e);
            format!("Failed to enumerate audio devices: {}", e)
        })
}
```

The existing `get_settings()` and `update_settings()` commands automatically handle the new `device_id` field through Rust's serialization.

### 4. Application State Integration (Rust - lib.rs)

Update the recording start logic to use the selected device.

#### Modified start_recording Function

```rust
fn start_recording(state: &AppState, app: &tauri::AppHandle) -> anyhow::Result<()> {
    info!("Starting recording");
    
    // ... existing permission checks ...
    
    // Get device from settings
    let device_id = {
        let settings = state.settings.lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock settings: {}", e))?;
        settings.device_id.clone()
    };
    
    // Create recorder with selected device
    let recorder = match AudioRecorder::from_device_id(&device_id) {
        Ok(rec) => rec,
        Err(e) => {
            warn!("Failed to use selected device: {}. Falling back to system default.", e);
            
            // Notify user about fallback
            let _ = app.emit("device-fallback", format!(
                "Selected device unavailable. Using system default."
            ));
            
            // Use system default
            AudioRecorder::from_device_id(&DeviceId::SystemDefault)?
        }
    };
    
    // Get transcriber and start it with the recorder
    let mut transcriber = state.transcriber.lock()
        .map_err(|e| anyhow::anyhow!("Failed to lock transcriber: {}", e))?;
    
    transcriber.start_with_recorder(recorder)?;
    
    // ... rest of existing logic ...
}
```

Note: This requires modifying `RealtimeTranscriber` to accept a custom `AudioRecorder` instance, or we can modify the transcriber initialization to use the device from settings.

### 5. Frontend Settings Dialog (React/TypeScript)

New component for device selection integrated into existing settings dialog.

#### DeviceSelector Component

```typescript
import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';

interface AudioDeviceInfo {
  id: string;
  name: string;
  is_default: boolean;
}

interface DeviceId {
  SystemDefault?: null;
  Specific?: string;
}

interface DeviceSelectorProps {
  value: DeviceId;
  onChange: (deviceId: DeviceId) -> void;
}

export function DeviceSelector({ value, onChange }: DeviceSelectorProps) {
  const [devices, setDevices] = useState<AudioDeviceInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDevices();
  }, []);

  const loadDevices = async () => {
    try {
      setLoading(true);
      setError(null);
      const deviceList = await invoke<AudioDeviceInfo[]>('get_audio_devices');
      setDevices(deviceList);
    } catch (err) {
      setError(err as string);
      console.error('Failed to load audio devices:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedValue = event.target.value;
    if (selectedValue === 'system-default') {
      onChange({ SystemDefault: null });
    } else {
      onChange({ Specific: selectedValue });
    }
  };

  const getCurrentValue = (): string => {
    if ('SystemDefault' in value) {
      return 'system-default';
    } else if ('Specific' in value) {
      return value.Specific;
    }
    return 'system-default';
  };

  if (loading) {
    return <div>Loading devices...</div>;
  }

  if (error) {
    return (
      <div>
        <p>Error loading devices: {error}</p>
        <button onClick={loadDevices}>Retry</button>
      </div>
    );
  }

  return (
    <div className="device-selector">
      <label htmlFor="audio-device">Microphone:</label>
      <select
        id="audio-device"
        value={getCurrentValue()}
        onChange={handleChange}
      >
        <option value="system-default">System Default</option>
        {devices.map((device) => (
          <option key={device.id} value={device.id}>
            {device.name} {device.is_default ? '(Default)' : ''}
          </option>
        ))}
      </select>
    </div>
  );
}
```

#### Integration into Settings Dialog

```typescript
import { DeviceSelector } from './DeviceSelector';

function SettingsDialog() {
  const [settings, setSettings] = useState<Settings | null>(null);

  // ... existing settings loading logic ...

  const handleDeviceChange = (deviceId: DeviceId) => {
    setSettings(prev => prev ? { ...prev, device_id: deviceId } : null);
  };

  return (
    <div className="settings-dialog">
      <h2>Settings</h2>
      
      <DeviceSelector
        value={settings?.device_id || { SystemDefault: null }}
        onChange={handleDeviceChange}
      />
      
      {/* Existing model and language selectors */}
      
      <button onClick={saveSettings}>Save</button>
    </div>
  );
}
```

## Data Models

### DeviceId (Rust Enum)

```rust
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "type")]
pub enum DeviceId {
    SystemDefault,
    Specific(String),
}
```

**Serialization Examples:**
- System Default: `{"type": "SystemDefault"}`
- Specific Device: `{"type": "Specific", "value": "MacBook Pro Microphone"}`

### AudioDeviceInfo (Rust Struct)

```rust
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AudioDeviceInfo {
    pub id: String,
    pub name: String,
    pub is_default: bool,
}
```

### Settings (Extended)

```rust
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Settings {
    pub model: String,
    pub language: Option<String>,
    #[serde(default)]
    pub device_id: DeviceId,
}
```

**Example config.json:**
```json
{
  "model": "tiny.en",
  "language": null,
  "device_id": {
    "type": "Specific",
    "value": "USB Microphone"
  }
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property 1: Device Enumeration Completeness
*For any* system with available audio input devices, enumerating devices should return a list where every device has both a non-empty name and a non-empty identifier.
**Validates: Requirements 1.1, 1.2, 1.3**

### Property 2: Device List Freshness
*For any* two consecutive calls to device enumeration, the second call should reflect the current system state, not cached data.
**Validates: Requirements 1.5**

### Property 3: UI Device Display
*For any* list of audio devices, the rendered UI should contain a select option for each device displaying its human-readable name.
**Validates: Requirements 2.2**

### Property 4: UI Selection Update
*For any* device selection change in the UI, the onChange handler should be called with the corresponding DeviceId value.
**Validates: Requirements 2.3**

### Property 5: UI Selection State
*For any* DeviceId value passed to the DeviceSelector component, the select element's value should match that device ID.
**Validates: Requirements 2.4**

### Property 6: Settings Persistence Round-Trip
*For any* valid DeviceId value, saving settings with that device ID and then loading them back should produce an equivalent DeviceId value.
**Validates: Requirements 3.1, 3.2, 3.4**

### Property 7: Default Fallback Behavior
*For any* settings configuration where the device_id field is missing, invalid, or references an unavailable device, the system should fall back to DeviceId::SystemDefault.
**Validates: Requirements 3.3, 3.5, 9.1, 9.3**

### Property 8: Device Selection Integration
*For any* valid DeviceId in settings, starting recording should initialize the AudioRecorder with the device corresponding to that DeviceId.
**Validates: Requirements 4.1**

### Property 9: Fallback with Notification
*For any* invalid or unavailable DeviceId, attempting to start recording should fall back to the system default device and emit a notification event.
**Validates: Requirements 4.2, 4.5**

### Property 10: Device Input Validation
*For any* device returned by enumeration, that device should support audio input (have at least one input channel).
**Validates: Requirements 4.4**

### Property 11: Enumeration Robustness
*For any* system state (including error conditions), device enumeration should not panic and should return a valid result (either a list or an empty list).
**Validates: Requirements 5.5, 8.1**

### Property 12: SystemDefault Serialization
*For any* Settings instance with device_id set to DeviceId::SystemDefault, serializing and deserializing should preserve the SystemDefault variant.
**Validates: Requirements 6.3**

### Property 13: SystemDefault Device Query
*For any* recording start with DeviceId::SystemDefault, the system should query for the current default input device at that moment.
**Validates: Requirements 6.2**

### Property 14: Device Name Display
*For any* DeviceId value, the Settings::get_device_name() function should return a non-empty string suitable for display.
**Validates: Requirements 7.1, 7.5**

### Property 15: Unavailable Device Indication
*For any* DeviceId that is not present in the current device list, the UI should indicate that the device is unavailable.
**Validates: Requirements 7.4**

### Property 16: Fallback Notification Content
*For any* fallback to default device, the notification should include the name of the device being used.
**Validates: Requirements 7.3**

### Property 17: Device Initialization Error Handling
*For any* device that fails to initialize, the system should prevent recording from starting and emit an error notification containing the device name.
**Validates: Requirements 8.2, 8.4**

### Property 18: Settings Migration Preservation
*For any* old settings JSON without a device_id field, deserializing should preserve all existing fields (model, language) while adding device_id with default value.
**Validates: Requirements 9.2, 9.4**

### Property 19: Permission Handling
*For any* system state where microphone permissions are denied, device enumeration should handle it gracefully and return an empty list.
**Validates: Requirements 10.2, 10.4**

## Error Handling

### Device Enumeration Errors

**Scenario**: cpal fails to enumerate devices (permissions denied, system error)
**Handling**:
1. Log the error with full context
2. Return empty device list
3. UI displays "No devices available" message
4. User can retry enumeration

**Code Pattern**:
```rust
pub fn list_input_devices_with_info() -> Result<Vec<AudioDeviceInfo>> {
    match host.input_devices() {
        Ok(devices) => { /* enumerate */ },
        Err(e) => {
            error!("Failed to enumerate devices: {}", e);
            Ok(Vec::new()) // Return empty list, don't propagate error
        }
    }
}
```

### Device Not Found Errors

**Scenario**: Selected device is not available when starting recording
**Handling**:
1. Log warning with device ID
2. Fall back to system default device
3. Emit "device-fallback" event to frontend
4. Frontend displays notification to user
5. Recording proceeds with default device

**Code Pattern**:
```rust
let recorder = match AudioRecorder::from_device_id(&device_id) {
    Ok(rec) => rec,
    Err(e) => {
        warn!("Device {} not available: {}. Using default.", device_id, e);
        app.emit("device-fallback", "Selected device unavailable")?;
        AudioRecorder::from_device_id(&DeviceId::SystemDefault)?
    }
};
```

### Device Initialization Errors

**Scenario**: Device exists but fails to initialize (in use, hardware error)
**Handling**:
1. Log error with device details
2. Prevent recording from starting
3. Emit "device-error" event with error message
4. Frontend displays error dialog
5. User can try again or select different device

**Code Pattern**:
```rust
match recorder.start_streaming(tx) {
    Ok(stream) => { /* proceed */ },
    Err(e) => {
        error!("Failed to start device {}: {}", device_name, e);
        app.emit("device-error", format!("Failed to start {}: {}", device_name, e))?;
        return Err(anyhow::anyhow!("Device initialization failed"));
    }
}
```

### Settings Deserialization Errors

**Scenario**: Config file has invalid device_id format
**Handling**:
1. Log warning about invalid format
2. Use default value (DeviceId::SystemDefault)
3. Continue loading other settings
4. Save corrected settings on next update

**Code Pattern**:
```rust
#[derive(Serialize, Deserialize)]
pub struct Settings {
    pub model: String,
    pub language: Option<String>,
    #[serde(default)] // Automatically uses Default::default() if missing/invalid
    pub device_id: DeviceId,
}
```

### Permission Denied Errors

**Scenario**: macOS microphone permissions not granted
**Handling**:
1. Device enumeration returns empty list
2. UI shows "No devices available - check permissions" message
3. Provide link/button to open System Settings
4. User grants permission and retries

**Frontend Code**:
```typescript
if (devices.length === 0) {
  return (
    <div className="no-devices">
      <p>No microphone devices found.</p>
      <p>Please check that microphone permissions are granted in System Settings.</p>
      <button onClick={loadDevices}>Retry</button>
    </div>
  );
}
```

## Testing Strategy

### Dual Testing Approach

This feature will be tested using both unit tests and property-based tests:

- **Unit tests**: Verify specific examples, edge cases (empty device lists, invalid IDs), and error conditions
- **Property tests**: Verify universal properties across all inputs (serialization round-trips, fallback behavior, enumeration completeness)

Both testing approaches are complementary and necessary for comprehensive coverage. Unit tests catch concrete bugs in specific scenarios, while property tests verify general correctness across a wide range of inputs.

### Property-Based Testing Configuration

**Library**: `proptest` (Rust's property-based testing library)

**Configuration**:
- Minimum 100 iterations per property test
- Each test tagged with feature name and property number
- Tag format: `Feature: microphone-selection, Property N: [property text]`

**Example Property Test**:
```rust
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // Feature: microphone-selection, Property 6: Settings Persistence Round-Trip
    proptest! {
        #[test]
        fn test_device_id_round_trip(device_name in "[a-zA-Z0-9 ]{1,50}") {
            let device_id = DeviceId::Specific(device_name.clone());
            let settings = Settings {
                model: "tiny.en".to_string(),
                language: None,
                device_id: device_id.clone(),
            };
            
            // Serialize
            let json = serde_json::to_string(&settings).unwrap();
            
            // Deserialize
            let loaded: Settings = serde_json::from_str(&json).unwrap();
            
            // Verify round-trip
            assert_eq!(settings.device_id, loaded.device_id);
        }
    }
}
```

### Unit Testing Focus

Unit tests will focus on:

1. **Specific Examples**:
   - DeviceId::SystemDefault serialization
   - DeviceId::Specific("MacBook Pro Microphone") serialization
   - Empty device list handling
   - Single device in list

2. **Edge Cases**:
   - Device name with special characters
   - Very long device names
   - Device ID that doesn't match any available device
   - Settings file without device_id field (migration)

3. **Error Conditions**:
   - Device enumeration failure
   - Device initialization failure
   - Invalid JSON in settings file
   - Permissions denied

4. **Integration Points**:
   - Settings Manager ↔ Audio Device Manager
   - Tauri Commands ↔ Frontend
   - Audio Recorder ↔ Device Selection

**Example Unit Test**:
```rust
#[test]
fn test_system_default_serialization() {
    let device_id = DeviceId::SystemDefault;
    let json = serde_json::to_string(&device_id).unwrap();
    assert_eq!(json, r#"{"type":"SystemDefault"}"#);
    
    let deserialized: DeviceId = serde_json::from_str(&json).unwrap();
    assert_eq!(device_id, deserialized);
}

#[test]
fn test_settings_migration_from_old_format() {
    // Old settings without device_id field
    let old_json = r#"{"model":"tiny.en","language":null}"#;
    
    let settings: Settings = serde_json::from_str(old_json).unwrap();
    
    // Should have default device_id
    assert_eq!(settings.device_id, DeviceId::SystemDefault);
    // Should preserve other fields
    assert_eq!(settings.model, "tiny.en");
    assert_eq!(settings.language, None);
}

#[test]
fn test_device_fallback_on_invalid_id() {
    let invalid_id = DeviceId::Specific("NonexistentDevice".to_string());
    
    // Should fall back to system default
    let result = AudioRecorder::find_device_by_id(&invalid_id);
    
    // If it fails, we should be able to fall back
    if result.is_err() {
        let fallback = AudioRecorder::find_device_by_id(&DeviceId::SystemDefault);
        assert!(fallback.is_ok());
    }
}
```

### Frontend Testing

**Component Tests** (using React Testing Library):
```typescript
describe('DeviceSelector', () => {
  it('renders system default option first', () => {
    const devices = [
      { id: 'device1', name: 'USB Mic', is_default: false },
      { id: 'device2', name: 'Built-in', is_default: true },
    ];
    
    const { container } = render(
      <DeviceSelector value={{ SystemDefault: null }} onChange={() => {}} />
    );
    
    const options = container.querySelectorAll('option');
    expect(options[0].value).toBe('system-default');
    expect(options[0].textContent).toBe('System Default');
  });
  
  it('calls onChange with correct DeviceId when device selected', () => {
    const onChange = jest.fn();
    const devices = [{ id: 'usb-mic', name: 'USB Microphone', is_default: false }];
    
    const { container } = render(
      <DeviceSelector value={{ SystemDefault: null }} onChange={onChange} />
    );
    
    const select = container.querySelector('select');
    fireEvent.change(select, { target: { value: 'usb-mic' } });
    
    expect(onChange).toHaveBeenCalledWith({ Specific: 'usb-mic' });
  });
});
```

### Test Coverage Goals

- **Rust Backend**: 80%+ line coverage, 100% of error paths tested
- **Frontend Components**: 80%+ branch coverage
- **Property Tests**: All 19 correctness properties implemented
- **Integration Tests**: End-to-end device selection flow

### Testing Execution

**Running Tests**:
```bash
# Rust unit tests
cargo test --lib

# Rust property tests (with more iterations)
PROPTEST_CASES=1000 cargo test --lib

# Frontend tests
pnpm test

# All tests
pnpm test:all
```

## Implementation Notes

### Backward Compatibility

The `#[serde(default)]` attribute on the `device_id` field ensures that:
1. Old config files without `device_id` will load successfully
2. The field will automatically be set to `DeviceId::SystemDefault`
3. No user intervention required
4. Next save will include the `device_id` field

### Device Identifier Strategy

We use device names as identifiers because:
1. cpal doesn't provide stable device IDs across sessions
2. Device names are human-readable and useful for debugging
3. Substring matching in `find_device_by_name()` provides flexibility
4. System default option handles cases where names change

### Performance Considerations

- Device enumeration is fast (< 10ms typically)
- Only enumerate when settings dialog opens (not on every render)
- No caching needed - cpal queries are efficient
- Device selection doesn't impact recording performance

### Platform-Specific Notes

**macOS**:
- Uses Core Audio via cpal
- Requires microphone permissions
- Device names may include manufacturer info
- System default device can change based on user preferences

**Future Platform Support**:
- Linux: PulseAudio/ALSA via cpal
- Windows: WASAPI via cpal
- Same Rust code should work across platforms

### Migration Path

**Phase 1** (Current Implementation):
- Add device selection to settings
- Use device names as IDs
- System default option

**Phase 2** (Future Enhancement):
- Device hotplug detection
- Automatic device switching
- Device preference profiles
- Per-application device selection

## Security Considerations

1. **Permission Handling**: Always check microphone permissions before device access
2. **Input Validation**: Validate device IDs from config file (handled by serde)
3. **Error Messages**: Don't expose system paths or sensitive info in error messages
4. **Config File**: Stored in user's Application Support directory (standard location)

## Accessibility Considerations

1. **Keyboard Navigation**: Dropdown fully keyboard accessible
2. **Screen Readers**: Proper labels on form elements
3. **Error Messages**: Clear, actionable error text
4. **Visual Indicators**: Show default device, unavailable devices
