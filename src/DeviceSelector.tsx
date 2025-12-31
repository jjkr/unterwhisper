import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';

interface AudioDeviceInfo {
  id: string;
  name: string;
  is_default: boolean;
}

interface DeviceId {
  type: 'SystemDefault' | 'Specific';
  value?: string;
}

interface DeviceSelectorProps {
  value: DeviceId;
  onChange: (deviceId: DeviceId) => void;
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
      onChange({ type: 'SystemDefault' });
    } else {
      onChange({ type: 'Specific', value: selectedValue });
    }
  };

  const getCurrentValue = (): string => {
    if (value.type === 'SystemDefault') {
      return 'system-default';
    } else if (value.type === 'Specific' && value.value) {
      return value.value;
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

  if (devices.length === 0) {
    return (
      <div className="no-devices">
        <p>No microphone devices found.</p>
        <p>Please check that microphone permissions are granted in System Settings.</p>
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
