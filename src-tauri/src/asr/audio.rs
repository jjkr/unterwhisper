use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::Sender;
use ringbuf::traits::Producer;
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction};
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

/// Audio chunk containing raw PCM samples
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub timestamp: std::time::Instant,
}

impl AudioChunk {
    pub fn new(samples: Vec<f32>) -> Self {
        Self {
            samples,
            timestamp: std::time::Instant::now(),
        }
    }
}

/// Streaming audio recorder that sends chunks via channels
pub struct AudioRecorder {
    sample_rate: u32,
    device: Option<cpal::Device>,
}

/// Create a resampler for converting from one sample rate to another
fn create_resampler(from_rate: u32, to_rate: u32) -> Result<SincFixedIn<f32>> {
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    
    let resampler = SincFixedIn::<f32>::new(
        to_rate as f64 / from_rate as f64,
        2.0, // max_resample_ratio_relative
        params,
        1024, // chunk_size
        1,    // num_channels
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
        let resampler: Option<Arc<Mutex<SincFixedIn<f32>>>> = if needs_resampling {
            info!("Creating resampler: {}Hz -> {}Hz", native_sample_rate, self.sample_rate);
            Some(Arc::new(Mutex::new(create_resampler(native_sample_rate, self.sample_rate)?)))
        } else {
            info!("No resampling needed");
            None
        };

        let err_fn = |err| error!("Audio stream error: {}", err);

        let stream = input_device.build_input_stream(
            &config,
            move |data: &[f32], info: &cpal::InputCallbackInfo| {
                debug!("Received audio chunk with {} samples, timestamp: {:?}", data.len(), info.timestamp());
                
                // Resample if needed
                let samples_to_push = if let Some(ref resampler_arc) = resampler {
                    let mut resampler = match resampler_arc.lock() {
                        Ok(r) => r,
                        Err(e) => {
                            error!("Failed to lock resampler: {}", e);
                            return;
                        }
                    };
                    
                    // Prepare input for resampler (needs Vec<Vec<f32>> for multi-channel)
                    let input_frames = vec![data.to_vec()];
                    
                    // Process resampling
                    match resampler.process(&input_frames, None) {
                        Ok(output_frames) => {
                            // Extract the resampled data from the first channel
                            output_frames[0].clone()
                        }
                        Err(e) => {
                            error!("Resampling error: {}", e);
                            return;
                        }
                    }
                } else {
                    // No resampling needed, use data as-is
                    data.to_vec()
                };
                
                // Push samples to ringbuf (non-blocking)
                let pushed = producer.push_slice(&samples_to_push);
                
                if pushed < samples_to_push.len() {
                    warn!("Ringbuf full: dropped {} samples out of {}", samples_to_push.len() - pushed, samples_to_push.len());
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

    /// Start streaming audio chunks to the provided channel (legacy interface)
    /// Returns the audio stream for proper lifecycle management
    pub fn start_streaming(&self, tx: Sender<AudioChunk>) -> Result<cpal::Stream> {
        info!("Starting continuous audio streaming (no buffering)");
        
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
        let resampler: Option<Arc<Mutex<SincFixedIn<f32>>>> = if needs_resampling {
            info!("Creating resampler: {}Hz -> {}Hz", native_sample_rate, self.sample_rate);
            Some(Arc::new(Mutex::new(create_resampler(native_sample_rate, self.sample_rate)?)))
        } else {
            info!("No resampling needed");
            None
        };

        let err_fn = |err| error!("Audio stream error: {}", err);

        let stream = input_device.build_input_stream(
            &config,
            move |data: &[f32], info: &cpal::InputCallbackInfo| {
                debug!("Received audio chunk with {} samples, timestamp: {:?}", data.len(), info.timestamp());
                
                // Resample if needed
                let samples = if let Some(ref resampler_arc) = resampler {
                    let mut resampler = match resampler_arc.lock() {
                        Ok(r) => r,
                        Err(e) => {
                            error!("Failed to lock resampler: {}", e);
                            return;
                        }
                    };
                    
                    // Prepare input for resampler (needs Vec<Vec<f32>> for multi-channel)
                    let input_frames = vec![data.to_vec()];
                    
                    // Process resampling
                    match resampler.process(&input_frames, None) {
                        Ok(output_frames) => {
                            // Extract the resampled data from the first channel
                            output_frames[0].clone()
                        }
                        Err(e) => {
                            error!("Resampling error: {}", e);
                            return;
                        }
                    }
                } else {
                    // No resampling needed, use data as-is
                    data.to_vec()
                };
                
                // Send audio data
                let chunk = AudioChunk::new(samples);
                
                if let Err(e) = tx.send(chunk) {
                    error!("Failed to send audio chunk: {}", e);
                    return;
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

/// Calculate RMS (Root Mean Square) values for audio data in time windows
pub fn calculate_rms_over_time(audio_data: &[f32], window_size_ms: u32) -> Vec<f32> {
    let window_size_samples = (SAMPLE_RATE * window_size_ms / 1000) as usize;
    let mut rms_values = Vec::new();
    
    if audio_data.is_empty() || window_size_samples == 0 {
        return rms_values;
    }
    
    // Calculate RMS for each window
    for chunk in audio_data.chunks(window_size_samples) {
        let sum_squares: f32 = chunk.iter().map(|&x| x * x).sum();
        let rms = (sum_squares / chunk.len() as f32).sqrt();
        rms_values.push(rms);
    }
    
    rms_values
}

/// Display a histogram of RMS values over time
pub fn display_rms_histogram(rms_values: &[f32], window_size_ms: u32) {
    if rms_values.is_empty() {
        println!("📊 RMS Histogram: No data");
        return;
    }
    
    let max_rms = rms_values.iter().fold(0.0f32, |a, &b| a.max(b));
    let min_rms = rms_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    
    println!("\n📊 RMS Audio Level Histogram ({}ms windows):", window_size_ms);
    println!("   Range: {:.4} - {:.4} | Duration: {:.1}s", 
             min_rms, max_rms, 
             (rms_values.len() * window_size_ms as usize) as f32 / 1000.0);
    
    // Create histogram bars
    const HISTOGRAM_WIDTH: usize = 200;
    const HISTOGRAM_HEIGHT: usize = 10;
    
    // Normalize RMS values to histogram height
    let scale = if max_rms > 0.0 { HISTOGRAM_HEIGHT as f32 / max_rms } else { 1.0 };
    
    // Print histogram from top to bottom
    for row in (0..HISTOGRAM_HEIGHT).rev() {
        print!("   ");
        let threshold = (row + 1) as f32 / scale;
        
        for &rms in rms_values.iter().take(HISTOGRAM_WIDTH) {
            if rms >= threshold {
                print!("█");
            } else {
                print!(" ");
            }
        }
        
        // Show scale on the right
        if row == HISTOGRAM_HEIGHT - 1 {
            println!(" {:.3}", max_rms);
        } else if row == 0 {
            println!(" 0.000");
        } else {
            println!();
        }
    }
    
    // Print time axis
    print!("   ");
    for i in 0..HISTOGRAM_WIDTH.min(rms_values.len()) {
        if i % 10 == 0 {
            print!("|");
        } else if i % 5 == 0 {
            print!("·");
        } else {
            print!("-");
        }
    }
    println!();
    
    // Print time labels
    print!("   ");
    for i in 0..HISTOGRAM_WIDTH.min(rms_values.len()) {
        if i % 10 == 0 {
            let time_s = (i * window_size_ms as usize) as f32 / 1000.0;
            print!("{:.0}", time_s);
            // Add padding for multi-digit numbers
            for _ in 1..format!("{:.0}", time_s).len() {
                print!(" ");
            }
        } else {
            print!(" ");
        }
    }
    println!(" (seconds)");
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_recorder_creation() {
        let recorder = AudioRecorder::new();
        assert_eq!(recorder.sample_rate, SAMPLE_RATE);
    }

    #[test]
    fn test_calculate_rms_over_time_empty() {
        let audio_data: Vec<f32> = vec![];
        let rms_values = calculate_rms_over_time(&audio_data, 100);
        assert!(rms_values.is_empty());
    }

    #[test]
    fn test_calculate_rms_over_time_single_window() {
        // Create 100ms of audio data at 16kHz (1600 samples)
        let window_size_ms = 100;
        let samples_per_window = (SAMPLE_RATE * window_size_ms / 1000) as usize;
        let audio_data: Vec<f32> = vec![0.5; samples_per_window];
        
        let rms_values = calculate_rms_over_time(&audio_data, window_size_ms);
        assert_eq!(rms_values.len(), 1);
        assert!((rms_values[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_rms_over_time_multiple_windows() {
        // Create 300ms of audio data at 16kHz (4800 samples)
        let window_size_ms = 100;
        let samples_per_window = (SAMPLE_RATE * window_size_ms / 1000) as usize;
        let mut audio_data = Vec::new();
        
        // First window: amplitude 0.1
        audio_data.extend(vec![0.1; samples_per_window]);
        // Second window: amplitude 0.5
        audio_data.extend(vec![0.5; samples_per_window]);
        // Third window: amplitude 0.9
        audio_data.extend(vec![0.9; samples_per_window]);
        
        let rms_values = calculate_rms_over_time(&audio_data, window_size_ms);
        assert_eq!(rms_values.len(), 3);
        assert!((rms_values[0] - 0.1).abs() < 1e-5);
        assert!((rms_values[1] - 0.5).abs() < 1e-5);
        assert!((rms_values[2] - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_calculate_rms_zero_window_size() {
        let audio_data: Vec<f32> = vec![0.5; 1000];
        let rms_values = calculate_rms_over_time(&audio_data, 0);
        assert!(rms_values.is_empty());
    }
}
