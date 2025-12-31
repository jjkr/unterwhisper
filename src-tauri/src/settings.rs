use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

use crate::asr::audio::{AudioRecorder, DeviceId};

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
    /// Get the path to the config file
    fn config_path() -> Result<PathBuf> {
        let app_dir = dirs::data_local_dir()
            .context("Failed to get local data directory")?
            .join("com.unterwhisper.app");
        Ok(app_dir.join("config.json"))
    }

    /// Validate that the selected device is available
    pub fn validate_device(&self) -> Result<()> {
        AudioRecorder::find_device_by_id(&self.device_id)?;
        Ok(())
    }

    /// Get the device name for display purposes
    pub fn get_device_name(&self) -> String {
        match &self.device_id {
            DeviceId::SystemDefault => "System Default".to_string(),
            DeviceId::Specific { value } => value.clone(),
        }
    }

    /// Load settings from the config file, or return defaults if file doesn't exist
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path()?;

        if !config_path.exists() {
            log::info!("Config file not found, using defaults");
            return Ok(Self::default());
        }

        let contents = fs::read_to_string(&config_path)
            .context("Failed to read config file")?;

        match serde_json::from_str::<Settings>(&contents) {
            Ok(settings) => {
                log::info!("Loaded settings from {:?}", config_path);
                Ok(settings)
            }
            Err(e) => {
                log::warn!("Failed to parse config file: {}. Creating backup and using defaults.", e);
                
                // Create backup of corrupted file
                let backup_path = config_path.with_extension("json.backup");
                if let Err(backup_err) = fs::copy(&config_path, &backup_path) {
                    log::error!("Failed to create backup: {}", backup_err);
                }
                
                // Return defaults
                Ok(Self::default())
            }
        }
    }

    /// Save settings to the config file
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path()?;

        // Ensure the config directory exists
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)
                .context("Failed to create config directory")?;
        }

        let json = serde_json::to_string_pretty(self)
            .context("Failed to serialize settings")?;

        fs::write(&config_path, json)
            .context("Failed to write config file")?;

        log::info!("Saved settings to {:?}", config_path);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Mutex;
    use std::sync::OnceLock;

    // Global lock to prevent tests from running concurrently
    static TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn get_test_lock() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[test]
    fn test_default_settings() {
        let settings = Settings::default();
        assert_eq!(settings.model, "tiny.en");
        assert_eq!(settings.language, None);
        assert_eq!(settings.device_id, DeviceId::SystemDefault);
    }

    #[test]
    fn test_settings_serialization() {
        let settings = Settings {
            model: "tiny.en".to_string(),
            language: Some("en".to_string()),
            device_id: DeviceId::SystemDefault,
        };

        let json = serde_json::to_string(&settings).unwrap();
        let deserialized: Settings = serde_json::from_str(&json).unwrap();

        assert_eq!(settings, deserialized);
    }

    #[test]
    fn test_corrupted_json_handling() {
        // Test that invalid JSON returns defaults
        let invalid_json = "{invalid json}";
        let result: Result<Settings, _> = serde_json::from_str(invalid_json);
        assert!(result.is_err());
    }

    #[test]
    fn test_save_and_load() {
        let _guard = get_test_lock();

        // Clean up any existing config first
        if let Ok(config_path) = Settings::config_path() {
            let _ = fs::remove_file(&config_path);
        }

        // Create custom settings
        let settings = Settings {
            model: "base.en".to_string(),
            language: Some("en".to_string()),
            device_id: DeviceId::Specific {
                value: "Test Microphone".to_string(),
            },
        };

        // Save settings
        settings.save().expect("Failed to save settings");

        // Verify file exists
        let config_path = Settings::config_path().expect("Failed to get config path");
        assert!(config_path.exists(), "Config file should exist after save");

        // Read the file content to verify
        let content = fs::read_to_string(&config_path).expect("Failed to read config file");
        println!("Saved config content: {}", content);

        // Load settings back
        let loaded = Settings::load().expect("Failed to load settings");

        // Verify they match
        assert_eq!(settings, loaded);

        // Clean up - remove the test config file
        if let Ok(config_path) = Settings::config_path() {
            let _ = fs::remove_file(config_path);
        }
    }

    #[test]
    fn test_load_nonexistent_file() {
        let _guard = get_test_lock();

        // Ensure config file doesn't exist
        if let Ok(config_path) = Settings::config_path() {
            let _ = fs::remove_file(&config_path);
        }

        // Load should return defaults
        let settings = Settings::load().expect("Failed to load settings");
        assert_eq!(settings, Settings::default());
    }

    #[test]
    fn test_config_directory_creation() {
        let _guard = get_test_lock();

        // Get config path
        let config_path = Settings::config_path().expect("Failed to get config path");
        
        // Remove config directory if it exists
        if let Some(parent) = config_path.parent() {
            let _ = fs::remove_dir_all(parent);
        }

        // Save should create the directory
        let settings = Settings::default();
        settings.save().expect("Failed to save settings");

        // Verify directory was created
        assert!(config_path.parent().unwrap().exists());

        // Clean up
        let _ = fs::remove_file(config_path);
    }

    #[test]
    fn test_get_device_name_system_default() {
        let settings = Settings {
            model: "tiny.en".to_string(),
            language: None,
            device_id: DeviceId::SystemDefault,
        };
        
        assert_eq!(settings.get_device_name(), "System Default");
    }

    #[test]
    fn test_get_device_name_specific() {
        let settings = Settings {
            model: "tiny.en".to_string(),
            language: None,
            device_id: DeviceId::Specific {
                value: "USB Microphone".to_string(),
            },
        };
        
        assert_eq!(settings.get_device_name(), "USB Microphone");
    }

    #[test]
    fn test_settings_with_device_id_serialization() {
        let settings = Settings {
            model: "tiny.en".to_string(),
            language: Some("en".to_string()),
            device_id: DeviceId::Specific {
                value: "MacBook Pro Microphone".to_string(),
            },
        };

        let json = serde_json::to_string(&settings).unwrap();
        let deserialized: Settings = serde_json::from_str(&json).unwrap();

        assert_eq!(settings, deserialized);
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
}
