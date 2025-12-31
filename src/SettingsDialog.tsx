import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { DeviceSelector } from './DeviceSelector';
import './SettingsDialog.css';

interface DeviceId {
  type: 'SystemDefault' | 'Specific';
  value?: string;
}

interface Settings {
  model: string;
  language: string | null;
  device_id: DeviceId;
}

interface SettingsDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

export function SettingsDialog({ isOpen, onClose }: SettingsDialogProps) {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveSuccess, setSaveSuccess] = useState(false);

  // Available models
  const models = [
    'tiny.en',
    'tiny',
    'base.en',
    'base',
    'small.en',
    'small',
    'medium.en',
    'medium',
    'large-v2',
    'large-v3',
    'large-v3-turbo',
    'large-v3-turbo-q41-gguf',
    'large-v3-turbo-q4k-gguf',
    'distil-small.en',
    'distil-medium.en',
    'distil-large-v3',
    'distil-large-v3.5',
    'parakeet-tdt-0.6b-v3',
  ];

  // Available languages (null means auto-detect)
  const languages = [
    { value: null, label: 'Auto-detect' },
    { value: 'en', label: 'English' },
    { value: 'es', label: 'Spanish' },
    { value: 'fr', label: 'French' },
    { value: 'de', label: 'German' },
    { value: 'it', label: 'Italian' },
    { value: 'pt', label: 'Portuguese' },
    { value: 'nl', label: 'Dutch' },
    { value: 'ja', label: 'Japanese' },
    { value: 'zh', label: 'Chinese' },
    { value: 'ko', label: 'Korean' },
  ];

  useEffect(() => {
    if (isOpen) {
      loadSettings();
    }
  }, [isOpen]);

  const loadSettings = async () => {
    try {
      setLoading(true);
      setError(null);
      const loadedSettings = await invoke<Settings>('get_settings');
      setSettings(loadedSettings);
    } catch (err) {
      setError(err as string);
      console.error('Failed to load settings:', err);
    } finally {
      setLoading(false);
    }
  };

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

  const handleModelChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    if (settings) {
      setSettings({ ...settings, model: event.target.value });
    }
  };

  const handleLanguageChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    if (settings) {
      const value = event.target.value === '' ? null : event.target.value;
      setSettings({ ...settings, language: value });
    }
  };

  if (!isOpen) return null;

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
              <button onClick={loadSettings}>Retry</button>
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

              {/* Model Selection */}
              <div className="settings-section">
                <label htmlFor="model-select">Model:</label>
                <select
                  id="model-select"
                  value={settings.model}
                  onChange={handleModelChange}
                >
                  {models.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
                <p className="help-text">
                  Smaller models are faster but less accurate. Larger models are more accurate but slower.
                </p>
              </div>

              {/* Language Selection */}
              <div className="settings-section">
                <label htmlFor="language-select">Language:</label>
                <select
                  id="language-select"
                  value={settings.language || ''}
                  onChange={handleLanguageChange}
                >
                  {languages.map((lang) => (
                    <option key={lang.value || 'auto'} value={lang.value || ''}>
                      {lang.label}
                    </option>
                  ))}
                </select>
                <p className="help-text">
                  Select a language for better accuracy, or use auto-detect.
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
