use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::traits::Producer;
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use rubato::{Resampler, FftFixedInOut};
use std::sync::{Arc, Mutex};

pub const SAMPLE_RATE: u32 = 16000;
pub const CHANNELS: u16 = 1;

/// Device identifier that can be serialized and persisted
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "type")]
pub enum DeviceId {
    /// Use the system default input device
    SystemDefault,
    /// Use a specific device by its name
    Specific { value: String },
}

impl Default for DeviceId {
    fn default() -> Self {
        DeviceId::SystemDefault
    }
}

/// Information about an audio input device
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AudioDeviceInfo {
    /// Unique identifier for the device (using device name)
    pub id: String,
    /// Human-readable device name
    pub name: String,
    /// Whether this is the system default device
    pub is_default: bool,
}

/// Streaming audio recorder that sends chunks via channels
pub struct AudioRecorder {
    sample_rate: u32,
    device: Option<cpal::Device>,
}

/// Create a resampler for converting from one sample rate to another
fn create_resampler(from_rate: u32, to_rate: u32) -> Result<FftFixedInOut<f32>> {
    // Use a smaller chunk size that's more likely to match audio buffer sizes
    let chunk_size = 512; // Common audio buffer size
    
    let resampler = FftFixedInOut::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        chunk_size,
        1, // num_channels
    )?;
    
    Ok(resampler)
}

impl AudioRecorder {
    pub fn new() -> Self {
        Self {
            sample_rate: SAMPLE_RATE,
            device: None,
        }
    }

    /// Create a recorder with a specific device
    pub fn with_device(device: cpal::Device) -> Self {
        Self {
            sample_rate: SAMPLE_RATE,
            device: Some(device),
        }
    }

    /// List all available input devices
    pub fn list_input_devices() -> Result<Vec<(String, cpal::Device)>> {
        let host = cpal::default_host();
        let mut devices = Vec::new();

        for device in host.input_devices()? {
            if let Ok(desc) = device.description() {
                devices.push((desc.name().to_string(), device));
            }
        }

        Ok(devices)
    }

    /// Find a device by name (case-insensitive substring match)
    pub fn find_device_by_name(name: &str) -> Result<cpal::Device> {
        let devices = Self::list_input_devices()?;
        let name_lower = name.to_lowercase();

        for (device_name, device) in devices {
            if device_name.to_lowercase().contains(&name_lower) {
                info!("Found device: {}", device_name);
                return Ok(device);
            }
        }

        anyhow::bail!("No device found matching '{}'", name)
    }

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
            DeviceId::Specific { value } => {
                // Use existing find_device_by_name since we use name as ID
                Self::find_device_by_name(value)
            }
        }
    }

    /// Create a recorder with device from settings
    pub fn from_device_id(device_id: &DeviceId) -> Result<Self> {
        let device = Self::find_device_by_id(device_id)?;
        Ok(Self::with_device(device))
    }

    /// Start streaming audio samples to the provided ringbuf producer
    /// Returns the audio stream for proper lifecycle management
    pub fn start_streaming_ringbuf<P>(
        &self, 
        mut producer: P,
    ) -> Result<cpal::Stream> 
    where
        P: Producer<Item = f32> + Send + 'static,
    {
        info!("Starting continuous audio streaming with ringbuf");
        
        let host = cpal::default_host();
        let input_device = if let Some(ref device) = self.device {
            device.clone()
        } else {
            host.default_input_device()
                .context("No input device available")?
        };

        let device_name = input_device.description()
            .map(|d| d.name().to_string())
            .unwrap_or_else(|_| "Unknown".to_string());
        info!("Using input device: {}", device_name);

        // Get the device's default config to find native sample rate
        let default_config = input_device.default_input_config()
            .context("Failed to get default input config")?;
        let native_sample_rate = default_config.sample_rate();
        // native_sample_rate is already u32 in cpal 0.17
        
        info!("Device native sample rate: {}Hz, target: {}Hz", 
              native_sample_rate, self.sample_rate);

        // Log supported configs for debugging
        let supported_configs = input_device.supported_input_configs()
            .context("Failed to get supported input configs")?;
        info!("Supported input configs:");
        for config in supported_configs {
            info!("  - {:?}", config);
        }

        // Use the native sample rate instead of forcing 16kHz
        let config = cpal::StreamConfig {
            channels: CHANNELS,
            sample_rate: native_sample_rate,
            buffer_size: cpal::BufferSize::Default,
        };

        info!("Using config: {:?}", config);

        // Create resampler if needed
        let needs_resampling = native_sample_rate != self.sample_rate;
        let resampler: Option<Arc<Mutex<FftFixedInOut<f32>>>> = if needs_resampling {
            info!("Creating resampler: {}Hz -> {}Hz", native_sample_rate, self.sample_rate);
            Some(Arc::new(Mutex::new(create_resampler(native_sample_rate, self.sample_rate)?)))
        } else {
            info!("No resampling needed");
            None
        };
        
        // Buffer to accumulate samples for resampling (512 samples = chunk size)
        let input_buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

        let err_fn = |err| error!("Audio stream error: {}", err);

        let stream = input_device.build_input_stream(
            &config,
            move |data: &[f32], info: &cpal::InputCallbackInfo| {
                debug!("Received audio chunk with {} samples, timestamp: {:?}", data.len(), info.timestamp());
                
                // Resample if needed
                let samples_to_push = if let Some(ref resampler_arc) = resampler {
                    let mut buffer = match input_buffer.lock() {
                        Ok(b) => b,
                        Err(e) => {
                            error!("Failed to lock input buffer: {}", e);
                            return;
                        }
                    };
                    
                    let mut resampler = match resampler_arc.lock() {
                        Ok(r) => r,
                        Err(e) => {
                            error!("Failed to lock resampler: {}", e);
                            return;
                        }
                    };
                    
                    // Add new samples to buffer
                    buffer.extend_from_slice(data);
                    
                    let chunk_size = resampler.input_frames_next();
                    let mut output_samples = Vec::new();
                    
                    // Process complete chunks
                    while buffer.len() >= chunk_size {
                        let chunk: Vec<f32> = buffer.drain(..chunk_size).collect();
                        let input_frames = vec![chunk];
                        
                        match resampler.process(&input_frames, None) {
                            Ok(output_frames) => {
                                output_samples.extend_from_slice(&output_frames[0]);
                            }
                            Err(e) => {
                                error!("Resampling error: {}", e);
                                return;
                            }
                        }
                    }
                    
                    output_samples
                } else {
                    // No resampling needed, use data as-is
                    data.to_vec()
                };
                
                // Push samples to ringbuf (non-blocking)
                if !samples_to_push.is_empty() {
                    let pushed = producer.push_slice(&samples_to_push);
                    
                    if pushed < samples_to_push.len() {
                        warn!("Ringbuf full: dropped {} samples out of {}", samples_to_push.len() - pushed, samples_to_push.len());
                    }
                }
            },
            err_fn,
            None,
        )?;

        stream.play()?;
        info!("Audio stream started successfully");

        // Return the stream
        Ok(stream)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_recorder_creation() {
        let recorder = AudioRecorder::new();
        assert_eq!(recorder.sample_rate, SAMPLE_RATE);
    }
}
