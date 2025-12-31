//! Real-time Audio Transcription System
//!
//! This module provides a complete real-time transcription pipeline that:
//! 1. Captures audio from a microphone and sends chunks via crossbeam channels
//! 2. Accumulates and processes the last 30 seconds of audio into mel spectrograms
//! 3. Runs Whisper inference as fast as possible on complete windows
//! 4. Merges overlapping transcriptions into a coherent output
//!
//! # Architecture
//!
//! ```text
//! Audio Input → PCM Channel → Mel Encoder → Mel Channel → Whisper → Text Merger → Final Text
//!     (cpal)    (crossbeam)   (blocking)    (crossbeam)   (async)    (overlap)
//! ```
//!
//! All communication uses crossbeam channels with proper blocking select semantics
//! for efficient CPU usage and immediate shutdown response.
//!
//! # Example Usage
//!
//! ```no_run
//! use whisper_transcribe::transcribe::{RealtimeTranscriber, TranscriberConfig};
//! use candle_core::Device;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = TranscriberConfig::default();
//!     let mut transcriber = RealtimeTranscriber::new(config, Device::Cpu).await?;
//!     
//!     // Start transcription
//!     transcriber.start().await?;
//!     
//!     // Get transcription updates
//!     while let Some(text) = transcriber.next_transcription().await {
//!         println!("Transcription: {}", text);
//!     }
//!     
//!     Ok(())
//! }
//! ```

use crate::asr::audio::{AudioChunk, AudioRecorder, SAMPLE_RATE};
use crate::asr::dsp::{self, fft};
use crate::asr::whisper::WhisperTransformer;
use anyhow::Result;
use crossbeam_channel::{unbounded, Receiver, Sender};
use core::f32;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use log::{debug, info, warn};

// Whisper constants
const N_FFT: usize = 400; // Frame size 400 samples @ 16kHz = 25ms
const HOP_LENGTH: usize = 160; // Frame hop 160 samples @ 16kHz = 10ms
const N_FRAMES: usize = 3000; // Whisper expects 3000 frames at 10ms hops, total 30s
const N_SAMPLES: usize = 30 * SAMPLE_RATE as usize; // 480,000 samples for 30 seconds

/// Configuration for the real-time transcriber
#[derive(Debug, Clone)]
pub struct TranscriberConfig {
    /// Whisper model to use (e.g., "tiny.en", "base", "large-v3-turbo")
    pub model_name: String,

    /// Size of PCM audio ringbuffer in samples (default: 60 seconds)
    pub pcm_buffer_size: usize,

    /// Minimum interval between transcription runs (to avoid overloading)
    pub min_transcription_interval: Duration,

    /// Audio device name (None = default device)
    pub audio_device: Option<String>,

    /// Language for transcription (None = auto-detect)
    pub language: Option<String>,
}

impl Default for TranscriberConfig {
    fn default() -> Self {
        Self {
            model_name: "distil-large-v3.5".to_string(),
            pcm_buffer_size: SAMPLE_RATE as usize * 60, // 60 seconds
            min_transcription_interval: Duration::from_millis(500),
            audio_device: None,
            language: None,
        }
    }
}

/// Statistics about the transcription process
#[derive(Debug, Clone, Default)]
pub struct TranscriptionStats {
    /// Total number of transcriptions completed
    pub transcriptions_completed: usize,
    
    /// Average transcription time in milliseconds
    pub avg_transcription_time_ms: f64,
    
    /// Current PCM buffer fill level (0.0 to 1.0)
    pub pcm_buffer_fill: f32,
    
    /// Number of audio samples processed
    pub samples_processed: usize,
    
    /// Number of mel frames generated
    pub mel_frames_generated: usize,
}

/// A single transcription result with timing information
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    /// The transcribed text
    pub text: String,
    
    /// Timestamp when this transcription was generated
    pub timestamp: Instant,
    
    /// Audio window start time (relative to stream start)
    pub audio_start: Duration,
    
    /// Audio window end time (relative to stream start)
    pub audio_end: Duration,
    
    /// Confidence score (if available)
    pub confidence: Option<f32>,
}

/// Real-time audio transcriber with overlapping windows
pub struct RealtimeTranscriber {
    // Audio capture
    audio_recorder: AudioRecorder,
    audio_stream: Option<cpal::Stream>,

    // Whisper model config (for initialization)
    model_name: String,
    device: candle_core::Device,
    language: Option<String>,

    // PCM audio channel (raw audio chunks)
    pcm_tx: Option<Sender<AudioChunk>>,
    pcm_rx: Option<Receiver<AudioChunk>>,

    // Mel spectrogram channel (complete 30-second windows)
    // mel_tx: Sender<Vec<f32>>,
    // mel_rx: Option<Receiver<Vec<f32>>>,
    mel_buffer: Arc<Mutex<VecDeque<Vec<f32>>>>,

    // Transcription results channel
    transcription_tx: Sender<TranscriptionResult>,
    transcription_rx: Option<Receiver<TranscriptionResult>>,

    // Statistics (shared for thread access)
    stats: Arc<Mutex<TranscriptionStats>>,

    // Thread handles for cleanup
    mel_thread: Option<thread::JoinHandle<()>>,
    whisper_thread: Option<thread::JoinHandle<()>>,

    // Stream start time
    stream_start: Option<Instant>,
    
    // Shutdown signals (one for mel thread, whisper thread has its own)
    shutdown_tx: Option<Sender<()>>,
    whisper_shutdown_tx: Option<Sender<()>>,
}

impl RealtimeTranscriber {
    /// Create a new real-time transcriber
    pub fn new(
        config: TranscriberConfig,
        device: candle_core::Device,
    ) -> Result<Self> {
        info!("Initializing RealtimeTranscriber with model: {}", config.model_name);

        // Create audio recorder
        let audio_recorder = if let Some(ref device_name) = config.audio_device {
            let device = AudioRecorder::find_device_by_name(device_name)?;
            AudioRecorder::with_device(device)
        } else {
            AudioRecorder::new()
        };

        // Create channels
        let (pcm_tx, pcm_rx) = unbounded();
        // let (mel_tx, mel_rx) = unbounded();
        let (transcription_tx, transcription_rx) = unbounded();

        Ok(Self {
            audio_recorder,
            audio_stream: None,
            model_name: config.model_name,
            device,
            language: config.language,
            pcm_tx: Some(pcm_tx),
            pcm_rx: Some(pcm_rx),
            // mel_tx,
            // mel_rx: Some(mel_rx),
            mel_buffer: Arc::new(Mutex::new(VecDeque::new())),
            transcription_tx,
            transcription_rx: Some(transcription_rx),
            stats: Arc::new(Mutex::new(TranscriptionStats::default())),
            mel_thread: None,
            whisper_thread: None,
            stream_start: None,
            shutdown_tx: None,
            whisper_shutdown_tx: None,
        })
    }
    
    /// Start the transcription pipeline
    pub fn start(&mut self) -> Result<()> {
        info!("Starting transcription pipeline");
        
        self.stream_start = Some(Instant::now());

        // Get the PCM sender for audio capture
        let pcm_tx = self.pcm_tx.take().expect("pcm_tx already taken");

        // Load Whisper model to get config
        let whisper = WhisperTransformer::new(
            &self.model_name,
            self.device.clone(),
            self.language.clone(),
        )?;
        let whisper_config = whisper.config().clone();

        // Start audio capture directly with crossbeam channel
        self.audio_stream = Some(self.audio_recorder.start_streaming(pcm_tx)?);

        // Create separate shutdown channels for each thread
        let (mel_shutdown_tx, mel_shutdown_rx) = unbounded();
        let (whisper_shutdown_tx, whisper_shutdown_rx) = unbounded();
        
        // Store both shutdown senders
        self.shutdown_tx = Some(mel_shutdown_tx);
        self.whisper_shutdown_tx = Some(whisper_shutdown_tx);

        // Start mel encoding thread
        let pcm_rx = self.pcm_rx.take().expect("pcm_rx already taken");
        self.start_mel_encoding_thread(pcm_rx, mel_shutdown_rx, whisper_config.num_mel_bins);

        // Start transcription thread
        self.start_transcription_thread(whisper, whisper_shutdown_rx);

        info!("Transcription pipeline started successfully");
        Ok(())
    }

    /// Process an audio chunk and generate mel spectrogram if enough samples are available
    fn process_audio_chunk(
        chunk: &AudioChunk,
        pcm_buffer: &mut VecDeque<f32>,
        mel_buffer: &Arc<Mutex<VecDeque<Vec<f32>>>>,
        n_mel_bins: usize,
    ) -> Result<()> {

        // Accumulate PCM samples
        pcm_buffer.extend(&chunk.samples);

        let mut mel_buffer_locked = mel_buffer.lock().expect("Mel buffer lock poisoned");

        let hann_window = dsp::hann_window(N_FFT).expect("Failed to create Hann window");
        while pcm_buffer.len() > N_FFT {
            let windowed_frame = pcm_buffer.iter().take(N_FFT).zip(&hann_window).map(|(x, h)| x * h);
            let fft_frame = match fft(windowed_frame.collect()) {
                Ok(output) => output,
                Err(e) => {
                    warn!("Mel computation failed: {}", e);
                    continue;
                }
            };
            // Compute mel features
            let mel_features = match dsp::mel(&fft_frame, n_mel_bins) {
                Ok(features) => features,
                Err(e) => {
                    warn!("Mel computation failed: {}", e);
                    continue;
                }
            };

            //debug!("Computed MEL features");
            // mel_tx.send(mel_features);
            mel_buffer_locked.push_back(mel_features);

            pcm_buffer.drain(0..HOP_LENGTH);
            // // Store in column-major order (mel_bin, frame)
            // for (mel_bin, &value) in mel_features.iter().enumerate() {
            //     mel_spectrogram[mel_bin * N_FRAMES + frame_idx] = value;
            // }
        }

        // // Only keep last 30s of mel data
        if mel_buffer_locked.len() > N_FRAMES {
            let extra_frames = mel_buffer_locked.len() - N_FRAMES;
            mel_buffer_locked.drain(extra_frames..);
        }

        Ok(())

        //// Keep only the most recent 30 seconds
        //if pcm_buffer.len() > N_SAMPLES {
        //    let excess = pcm_buffer.len() - N_SAMPLES;
        //    pcm_buffer.drain(0..excess);
        //}

        //// Process when we have enough samples
        //if pcm_buffer.len() < N_SAMPLES {
        //    return None;
        //}

        //// Process each frame
        //for frame_idx in 0..N_FRAMES {
        //    let start_sample = frame_idx * HOP_LENGTH;
        //    let end_sample = (start_sample + N_FFT).min(N_SAMPLES);
        //    
        //    // Extract frame and apply window
        //    for i in 0..(end_sample - start_sample) {
        //        frame_buffer[i] = pcm_buffer[start_sample + i] * hann_window[i];
        //    }
        //    // Zero-pad if needed
        //    for i in (end_sample - start_sample)..N_FFT {
        //        frame_buffer[i] = 0.0;
        //    }
        //    
        //    // Compute FFT
        //    let fft_output = match dsp::fft(frame_buffer) {
        //        Ok(output) => output,
        //        Err(e) => {
        //            warn!("FFT failed: {}", e);
        //            continue;
        //        }
        //    };
        //    
        //    // Compute mel features
        //    let mel_features = match dsp::mel(&fft_output, n_mel_bins) {
        //        Ok(features) => features,
        //        Err(e) => {
        //            warn!("Mel computation failed: {}", e);
        //            continue;
        //        }
        //    };
        //    
        //    // Store in column-major order (mel_bin, frame)
        //    for (mel_bin, &value) in mel_features.iter().enumerate() {
        //        mel_spectrogram[mel_bin * N_FRAMES + frame_idx] = value;
        //    }
        //}

        // // Apply log and normalization
        // for val in mel_spectrogram.iter_mut() {
        //     *val = (*val).max(1e-10).log10();
        // }

        // // Apply final normalization (similar to Whisper preprocessing)
        // let max_val = mel_spectrogram.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) - 8.0;
        // for val in mel_spectrogram.iter_mut() {
        //     *val = (*val).max(max_val) / 4.0 + 1.0;
        // }

        // Some(mel_spectrogram)
    }

    /// Start the mel encoding background thread
    fn start_mel_encoding_thread(
        &mut self,
        pcm_rx: Receiver<AudioChunk>,
        shutdown_rx: Receiver<()>,
        n_mel_bins: usize,
    ) {
        let stats = self.stats.clone();
        let mel_buffer = self.mel_buffer.clone();

        let handle = thread::Builder::new()
            .name("mel-encoder".to_string())
            .spawn(move || {
                info!("Mel encoding thread started");

                // Pre-allocate buffers
                let mut pcm_buffer = VecDeque::with_capacity(N_FFT * 2);
                // let mut fft_buffer = vec![0.0f32; N_FFT];

                loop {
                    // Block on either shutdown signal or PCM data
                    crossbeam_channel::select! {
                        recv(shutdown_rx) -> _ => {
                            info!("Mel encoding thread shutting down");
                            break;
                        }
                        recv(pcm_rx) -> result => {
                            match result {
                                Ok(chunk) => {
                                    debug!("Got audio chunk: {}", chunk.samples.len());
                                    Self::process_audio_chunk(
                                        &chunk,
                                        &mut pcm_buffer,
                                        &mel_buffer,
                                        n_mel_bins,
                                    ).unwrap_or_else(|e| warn!("Failed to process audio chunk: {}", e));
                                }
                                Err(_) => {
                                    info!("PCM channel disconnected");
                                    break;
                                }
                            }
                        }
                    }
                }

                info!("Mel encoding thread stopped");
            })
            .expect("Failed to spawn mel encoding thread");

        self.mel_thread = Some(handle);
    }

    /// Start the transcription background thread
    fn start_transcription_thread(&mut self, whisper: WhisperTransformer, shutdown_rx: Receiver<()>) {
        let transcription_tx = self.transcription_tx.clone();
        let stats = self.stats.clone();
        let stream_start = self.stream_start.unwrap();
        let mel_buffer = self.mel_buffer.clone();

        let handle = thread::Builder::new()
            .name("whisper-inference".to_string())
            .spawn(move || {
                info!("Transcription thread started");

                // Rebind whisper as mut
                let mut whisper = whisper;
                let n_mel_bins = whisper.config().num_mel_bins;

                loop {
                    // Check for shutdown signal (non-blocking)
                    if shutdown_rx.try_recv().is_ok() {
                        info!("Transcription thread shutting down");
                        break;
                    }

                    // Try to get the most recent mel spectrogram from buffer
                    // We need to construct a proper mel spectrogram with shape (n_mels, 3000)
                    let mel_data: Vec<f32> = {
                        let buffer = mel_buffer.lock().expect("Mel buffer lock poisoned");

                        if buffer.is_empty() {
                            Vec::new()
                        } else {
                            let n_frames = buffer.len().min(N_FRAMES);

                            // Construct mel spectrogram in row-major order: (n_mels, n_frames)
                            // Each row contains all time steps for one mel bin
                            let mut mel_spec = vec![0.0f32; N_FRAMES * n_mel_bins];

                            let mut max_mel = f32::NEG_INFINITY;
                            for (frame_idx, frame) in buffer.iter().take(n_frames).enumerate() {
                                for (mel_bin, &value) in frame.iter().enumerate() {
                                // for mel_bin in 0..n_mel_bins {
                                    // Store in row-major: mel_spec[mel_bin][frame_idx]
                                    mel_spec[mel_bin * N_FRAMES + frame_idx] = value;
                                    max_mel = value.max(max_mel);
                                }
                            }
                            max_mel -= 8.0;

                            for val in mel_spec.iter_mut() {
                                *val = (*val).max(max_mel) / 4.0 + 1.0;
                            }

                            mel_spec
                        }
                    };

                    debug!("MEL data size: {} frames", mel_data.len() / 3000);
                    if mel_data.len() == 0 {
                        debug!("Sleeping for 20ms");
                        thread::sleep(Duration::from_millis(20));
                        continue;
                    }

                    let start_time = Instant::now();

                    // Run Whisper transcription (blocking)
                    let transcription_result = whisper.transcribe_from_mel(&mel_data);

                    match transcription_result {
                        Ok(text) => {
                            let elapsed = start_time.elapsed();

                            // Update stats
                            if let Ok(mut s) = stats.lock() {
                                s.transcriptions_completed += 1;
                                let n = s.transcriptions_completed as f64;
                                s.avg_transcription_time_ms = 
                                    (s.avg_transcription_time_ms * (n - 1.0) + elapsed.as_millis() as f64) / n;
                            }

                            // Send result
                            let result = TranscriptionResult {
                                text,
                                timestamp: Instant::now(),
                                audio_start: stream_start.elapsed().saturating_sub(Duration::from_secs(30)),
                                audio_end: stream_start.elapsed(),
                                confidence: None,
                            };

                            if let Err(e) = transcription_tx.send(result) {
                                warn!("Failed to send transcription result: {}", e);
                            }

                            debug!("Transcription completed in {:?}", elapsed);
                        }
                        Err(e) => {
                            warn!("Transcription failed: {}", e);
                        }
                    }
                }

                info!("Transcription thread stopped");
            })
            .expect("Failed to spawn transcription thread");

        self.whisper_thread = Some(handle);
    }

    /// Get the next transcription result (blocking)
    pub fn next_transcription(&mut self) -> Option<TranscriptionResult> {
        self.transcription_rx.as_ref()?.recv().ok()
    }

    /// Try to get the next transcription result without blocking
    pub fn try_next_transcription(&mut self) -> Option<TranscriptionResult> {
        self.transcription_rx.as_ref()?.try_recv().ok()
    }

    /// Get current statistics
    pub fn stats(&self) -> TranscriptionStats {
        self.stats.lock().unwrap().clone()
    }

    /// Stop the transcription pipeline
    pub fn stop(&mut self) {
        info!("Stopping transcription pipeline");

        // Send shutdown signals to both threads
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
        if let Some(whisper_shutdown_tx) = self.whisper_shutdown_tx.take() {
            let _ = whisper_shutdown_tx.send(());
        }

        // Drop the audio stream to stop capture
        self.audio_stream = None;

        // Wait for threads to finish (with timeout)
        if let Some(handle) = self.mel_thread.take() {
            let _ = handle.join();
        }

        if let Some(handle) = self.whisper_thread.take() {
            let _ = handle.join();
        }

        // Recreate channels for next start
        let (pcm_tx, pcm_rx) = unbounded();
        self.pcm_tx = Some(pcm_tx);
        self.pcm_rx = Some(pcm_rx);

        // Clear mel buffer for next session
        self.mel_buffer.lock().unwrap().clear();

        info!("Transcription pipeline stopped and reset for next session");
    }
}

impl Drop for RealtimeTranscriber {
    fn drop(&mut self) {
        // Send shutdown signals to both threads
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
        if let Some(whisper_shutdown_tx) = self.whisper_shutdown_tx.take() {
            let _ = whisper_shutdown_tx.send(());
        }
        
        // Note: We don't join threads in Drop to avoid blocking
        // The threads will exit when they detect the shutdown signal
    }
}

/// Text merger that combines overlapping transcriptions into coherent output
pub struct TranscriptionMerger {
    /// Window of recent transcriptions
    recent_transcriptions: Vec<TranscriptionResult>,
    
    /// Maximum number of transcriptions to keep in window
    window_size: usize,
    
    /// Merged output text
    merged_text: String,
}

impl TranscriptionMerger {
    /// Create a new transcription merger
    pub fn new(window_size: usize) -> Self {
        Self {
            recent_transcriptions: Vec::new(),
            window_size,
            merged_text: String::new(),
        }
    }
    
    /// Add a new transcription and update the merged output
    pub fn add_transcription(&mut self, result: TranscriptionResult) {
        self.recent_transcriptions.push(result);
        
        // Keep only the most recent transcriptions
        if self.recent_transcriptions.len() > self.window_size {
            self.recent_transcriptions.remove(0);
        }
        
        // Merge transcriptions
        self.merge();
    }
    
    /// Merge overlapping transcriptions using a simple voting/consensus approach
    fn merge(&mut self) {
        if self.recent_transcriptions.is_empty() {
            return;
        }
        
        // Simple approach: use the most recent transcription
        // TODO: Implement more sophisticated merging with:
        // - Word-level alignment
        // - Confidence scores
        // - Longest common subsequence
        // - Voting across overlapping windows
        
        let latest = &self.recent_transcriptions[self.recent_transcriptions.len() - 1];
        self.merged_text = latest.text.clone();
    }
    
    /// Get the current merged transcription
    pub fn get_merged_text(&self) -> &str {
        &self.merged_text
    }
    
    /// Clear all transcriptions and reset
    pub fn reset(&mut self) {
        self.recent_transcriptions.clear();
        self.merged_text.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_default() {
        let config = TranscriberConfig::default();
        assert_eq!(config.model_name, "tiny.en");
    }
    
    #[test]
    fn test_merger_creation() {
        let merger = TranscriptionMerger::new(5);
        assert_eq!(merger.get_merged_text(), "");
    }
    
    #[test]
    fn test_merger_add_transcription() {
        let mut merger = TranscriptionMerger::new(5);
        
        let result = TranscriptionResult {
            text: "Hello world".to_string(),
            timestamp: Instant::now(),
            audio_start: Duration::from_secs(0),
            audio_end: Duration::from_secs(30),
            confidence: None,
        };
        
        merger.add_transcription(result);
        assert_eq!(merger.get_merged_text(), "Hello world");
    }
    
    #[test]
    fn test_merger_window_size() {
        let mut merger = TranscriptionMerger::new(3);
        
        for i in 0..5 {
            let result = TranscriptionResult {
                text: format!("Text {}", i),
                timestamp: Instant::now(),
                audio_start: Duration::from_secs(i * 30),
                audio_end: Duration::from_secs((i + 1) * 30),
                confidence: None,
            };
            merger.add_transcription(result);
        }
        
        // Should only keep the last 3
        assert_eq!(merger.recent_transcriptions.len(), 3);
    }
}
