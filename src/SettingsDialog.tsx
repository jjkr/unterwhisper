import { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { DeviceSelector } from './DeviceSelector';
import './SettingsDialog.css';

interface DeviceId {
  type: 'SystemDefault' | 'Specific';
  value?: string;
}

interface Settings {
  model_path: string;
  mode_idx: number;
  device_id: DeviceId;
}

interface SettingsDialogProps {
  onClose: () => void;
}

export function SettingsDialog({ onClose }: SettingsDialogProps) {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveSuccess, setSaveSuccess] = useState(false);

  const loadSettings = useCallback(async (showLoading: boolean) => {
    try {
      if (showLoading) setLoading(true);
      setError(null);
      setSaveSuccess(false);
      const loadedSettings = await invoke<Settings>('get_settings');
      setSettings(loadedSettings);
    } catch (err) {
      setError(err as string);
      console.error('Failed to load settings:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    loadSettings(true);
  }, [loadSettings]);

  // Silently refresh settings when the window regains focus
  // (e.g. reopened from tray menu after being hidden)
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    getCurrentWindow().onFocusChanged(({ payload: focused }) => {
      if (focused) {
        loadSettings(false);
      }
    }).then(fn => { unlisten = fn; });
    return () => { unlisten?.(); };
  }, [loadSettings]);

  const handleSave = async () => {
    if (!settings) return;

    try {
      setSaving(true);
      setError(null);
      setSaveSuccess(false);
      await invoke('update_settings', { settings });
      setSaveSuccess(true);

      // Auto-close after successful save
      setTimeout(() => {
        onClose();
      }, 1000);
    } catch (err) {
      setError(err as string);
      console.error('Failed to save settings:', err);
    } finally {
      setSaving(false);
    }
  };

  const handleDeviceChange = (deviceId: DeviceId) => {
    if (settings) {
      setSettings({ ...settings, device_id: deviceId });
    }
  };

  const handleModelPathChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (settings) {
      setSettings({ ...settings, model_path: event.target.value });
    }
  };

  const handleModeIdxChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    if (settings) {
      setSettings({ ...settings, mode_idx: parseInt(event.target.value) });
    }
  };

  return (
    <div className="settings-dialog-overlay" onClick={onClose}>
      <div className="settings-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="settings-dialog-header">
          <h2>Settings</h2>
          <button className="close-button" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        <div className="settings-dialog-content">
          {loading ? (
            <div className="loading">Loading settings...</div>
          ) : error && !settings ? (
            <div className="error">
              <p>Error loading settings: {error}</p>
              <button onClick={() => loadSettings(true)}>Retry</button>
            </div>
          ) : settings ? (
            <form onSubmit={(e) => { e.preventDefault(); handleSave(); }}>
              {/* Device Selection */}
              <div className="settings-section">
                <DeviceSelector
                  value={settings.device_id}
                  onChange={handleDeviceChange}
                />
              </div>

              {/* Model Path */}
              <div className="settings-section">
                <label htmlFor="model-path-input">Model Path:</label>
                <input
                  id="model-path-input"
                  type="text"
                  value={settings.model_path}
                  onChange={handleModelPathChange}
                />
                <p className="help-text">
                  Path to a .nemo model file or MLX model directory.
                </p>
              </div>

              {/* Streaming Mode */}
              <div className="settings-section">
                <label htmlFor="mode-idx-input">Streaming Mode:</label>
                <select
                  id="mode-idx-input"
                  value={settings.mode_idx}
                  onChange={handleModeIdxChange}
                >
                  <option value={0}>Balanced (1.1s chunks)</option>
                  <option value={1}>Faster (0.6s chunks)</option>
                  <option value={2}>Low Latency (0.16s chunks)</option>
                  <option value={3}>Realtime (0.08s chunks)</option>
                </select>
                <p className="help-text">
                  Lower latency responds faster but may reduce accuracy.
                </p>
              </div>

              {/* Error/Success Messages */}
              {error && (
                <div className="error-message">
                  {error}
                </div>
              )}
              {saveSuccess && (
                <div className="success-message">
                  Settings saved successfully!
                </div>
              )}

              {/* Action Buttons */}
              <div className="settings-dialog-actions">
                <button type="button" onClick={onClose} disabled={saving}>
                  Cancel
                </button>
                <button type="submit" disabled={saving} className="primary">
                  {saving ? 'Saving...' : 'Save'}
                </button>
              </div>
            </form>
          ) : null}
        </div>
      </div>
    </div>
  );
}
