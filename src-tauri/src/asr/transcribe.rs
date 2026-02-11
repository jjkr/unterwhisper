use crate::asr::audio::AudioRecorder;
use anyhow::{Context, Result};
use cpal::traits::StreamTrait;
use crossbeam_channel::{unbounded, Receiver};
use log::{debug, info, warn};
use mlx_nemo::asr::CacheAwareRNNTPipeline;
use ringbuf::{traits::*, HeapCons, HeapRb};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Configuration for the NeMo-based transcriber
#[derive(Debug, Clone)]
pub struct TranscriberConfig {
    /// Path to .nemo file or MLX model directory
    pub model_path: PathBuf,
    /// Streaming mode index (default: 0)
    pub mode_idx: usize,
}

impl Default for TranscriberConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("/Users/jokr/work/mlx-nemo-rs/models/animaslabs/nemotron-speech-streaming-en-0.6b-mlx-8bit"),
            mode_idx: 0,
        }
    }
}

/// A single transcription result
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub is_final: bool,
}

/// Real-time audio transcriber using NeMo streaming pipeline.
///
/// The audio stream is initialized once and persists across recording sessions
/// (paused when not recording, resumed on start). This avoids the cost of
/// re-opening the audio device and re-creating the resampler on every keypress.
pub struct NemoTranscriber {
    /// The NeMo pipeline (owned when not recording)
    pipeline: Option<CacheAwareRNNTPipeline>,

    /// Persistent audio stream (paused between recording sessions)
    audio_stream: Option<cpal::Stream>,

    /// Ringbuf consumer — lives here between sessions, moves into the
    /// transcription thread during recording, returned on stop
    ring_consumer: Option<HeapCons<f32>>,

    /// Transcription results channel
    result_rx: Option<Receiver<TranscriptionResult>>,

    /// Shutdown signal
    shutdown: Option<Arc<AtomicBool>>,

    /// Transcription thread handle (returns pipeline + consumer on exit)
    thread_handle: Option<thread::JoinHandle<(CacheAwareRNNTPipeline, HeapCons<f32>)>>,
}

impl NemoTranscriber {
    /// Create a new NeMo transcriber by loading a model
    pub fn new(config: TranscriberConfig) -> Result<Self> {
        info!("Loading NeMo model from: {:?} (mode_idx={})", config.model_path, config.mode_idx);

        let pipeline = CacheAwareRNNTPipeline::restore_from_directory(
            &config.model_path,
            config.mode_idx,
        )?;

        info!("NeMo model loaded successfully");

        Ok(Self {
            pipeline: Some(pipeline),
            audio_stream: None,
            ring_consumer: None,
            result_rx: None,
            shutdown: None,
            thread_handle: None,
        })
    }

    /// Initialize the audio stream if not already set up.
    ///
    /// Only performs the expensive device open / resampler creation on the
    /// first call. Subsequent calls are a no-op unless `reset_audio()` was
    /// called (e.g. after a device change in Settings).
    pub fn ensure_audio_ready(&mut self, recorder: AudioRecorder) -> Result<()> {
        if self.audio_stream.is_some() {
            info!("Audio stream already initialized, reusing");
            return Ok(());
        }

        info!("Initializing audio stream (first time or after device change)");

        // Create ringbuf for audio transfer (2 seconds buffer at 16kHz)
        let ringbuf = HeapRb::<f32>::new(16000 * 2);
        let (producer, consumer) = ringbuf.split();

        // Build and start the audio stream, then immediately pause it
        let stream = recorder.start_streaming_ringbuf(producer)?;
        stream.pause().context("Failed to pause audio stream after creation")?;

        self.audio_stream = Some(stream);
        self.ring_consumer = Some(consumer);

        info!("Audio stream initialized and paused");
        Ok(())
    }

    /// Start transcription. The audio stream must already be initialized via
    /// `ensure_audio_ready`.
    pub fn start(&mut self) -> Result<()> {
        info!("Starting NeMo transcription pipeline");

        // Take ownership of the pipeline for the transcription thread
        let mut pipeline = self
            .pipeline
            .take()
            .ok_or_else(|| anyhow::anyhow!("Pipeline not available (already running?)"))?;

        // Take the ringbuf consumer for the transcription thread
        let mut consumer = self
            .ring_consumer
            .take()
            .ok_or_else(|| anyhow::anyhow!("Audio not initialized — call ensure_audio_ready first"))?;

        // Drain any stale samples left over from the previous session
        while consumer.try_pop().is_some() {}

        // Resume the audio stream
        if let Some(ref stream) = self.audio_stream {
            stream.play().context("Failed to resume audio stream")?;
        }

        // Create result channel
        let (result_tx, result_rx) = unbounded::<TranscriptionResult>();
        self.result_rx = Some(result_rx);

        // Create shutdown signal
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_flag = shutdown.clone();
        self.shutdown = Some(shutdown);

        // Spawn transcription thread
        let handle = thread::Builder::new()
            .name("nemo-transcription".to_string())
            .spawn(move || {
                info!("NeMo transcription thread started");

                let mut audio_buf = Vec::with_capacity(16000);

                loop {
                    if shutdown_flag.load(Ordering::SeqCst) {
                        info!("NeMo transcription thread shutting down — flushing remaining audio");

                        // Drain ALL remaining samples from ringbuf
                        audio_buf.clear();
                        let available = consumer.occupied_len();
                        if available > 0 {
                            audio_buf.resize(available, 0.0f32);
                            let popped = consumer.pop_slice(&mut audio_buf);
                            audio_buf.truncate(popped);
                            if popped > 0 {
                                debug!("Flush: fed {} remaining samples to NeMo pipeline", popped);
                                pipeline.add_audio(&audio_buf);
                            }
                        }

                        // Signal end of audio stream so pipeline processes remaining buffered audio
                        pipeline.end_stream();

                        // Loop transcribe_step() until pipeline is fully done
                        loop {
                            match pipeline.transcribe_step() {
                                Ok(Some(output)) => {
                                    if output.has_new_tokens() {
                                        debug!("Flush: NeMo transcription: '{}' (final={})", output.text, output.is_final);
                                        let result = TranscriptionResult {
                                            text: output.text,
                                            is_final: output.is_final,
                                        };
                                        if let Err(e) = result_tx.send(result) {
                                            warn!("Flush: failed to send transcription result: {}", e);
                                            break;
                                        }
                                    }
                                }
                                Ok(None) => {
                                    // No more data to process
                                }
                                Err(e) => {
                                    warn!("Flush: NeMo transcription step failed: {}", e);
                                }
                            }
                            if pipeline.is_done() {
                                info!("NeMo pipeline flush complete");
                                break;
                            }
                        }

                        break;
                    }

                    // Pull available audio from ringbuf
                    audio_buf.clear();
                    let available = consumer.occupied_len();
                    if available > 0 {
                        audio_buf.resize(available, 0.0f32);
                        let popped = consumer.pop_slice(&mut audio_buf);
                        audio_buf.truncate(popped);

                        if popped > 0 {
                            debug!("Fed {} samples to NeMo pipeline", popped);
                            pipeline.add_audio(&audio_buf);
                        }
                    }

                    // Run transcription step
                    match pipeline.transcribe_step() {
                        Ok(Some(output)) => {
                            if output.has_new_tokens() {
                                debug!("NeMo transcription: '{}' (final={})", output.text, output.is_final);
                                let result = TranscriptionResult {
                                    text: output.text,
                                    is_final: output.is_final,
                                };
                                if let Err(e) = result_tx.send(result) {
                                    warn!("Failed to send transcription result: {}", e);
                                    break;
                                }
                            }
                        }
                        Ok(None) => {
                            // No audio available yet, wait a bit
                            thread::sleep(Duration::from_millis(10));
                        }
                        Err(e) => {
                            warn!("NeMo transcription step failed: {}", e);
                        }
                    }
                }

                info!("NeMo transcription thread stopped");
                (pipeline, consumer)
            })?;

        self.thread_handle = Some(handle);

        info!("NeMo transcription pipeline started");
        Ok(())
    }

    /// Try to get the next transcription result without blocking
    pub fn try_next_transcription(&mut self) -> Option<TranscriptionResult> {
        self.result_rx.as_ref()?.try_recv().ok()
    }

    /// Stop the transcription pipeline.
    ///
    /// Pauses the audio stream (keeps it alive for reuse) and waits for the
    /// transcription thread to flush remaining audio and exit.
    pub fn stop(&mut self) {
        info!("Stopping NeMo transcription pipeline");

        // Pause the audio stream to stop new samples flowing into ringbuf.
        // The stream stays alive for reuse on the next start().
        if let Some(ref stream) = self.audio_stream {
            if let Err(e) = stream.pause() {
                warn!("Failed to pause audio stream: {}", e);
            }
        }

        // Signal shutdown — thread will drain remaining ringbuf samples,
        // call end_stream(), and flush pipeline before exiting
        if let Some(ref shutdown) = self.shutdown {
            shutdown.store(true, Ordering::SeqCst);
        }

        // Wait for transcription thread to finish flushing and recover
        // both the pipeline and the ringbuf consumer for reuse
        if let Some(handle) = self.thread_handle.take() {
            match handle.join() {
                Ok((mut pipeline, consumer)) => {
                    pipeline.reset();
                    self.pipeline = Some(pipeline);
                    self.ring_consumer = Some(consumer);
                    info!("Pipeline and audio consumer recovered for reuse");
                }
                Err(_) => {
                    warn!("Transcription thread panicked, pipeline and consumer lost");
                    // Audio stream is now orphaned (consumer gone) — force reinit
                    self.audio_stream = None;
                }
            }
        }

        // Clean up shutdown flag but keep result_rx alive so callers can
        // drain remaining results. It gets overwritten on next start().
        self.shutdown = None;

        info!("NeMo transcription pipeline stopped");
    }

    /// Drop the audio stream so the next `ensure_audio_ready` reinitializes it.
    /// Call this when the audio device setting changes.
    pub fn reset_audio(&mut self) {
        info!("Resetting audio stream (will reinitialize on next recording)");
        self.audio_stream = None;
        self.ring_consumer = None;
    }
}

impl Drop for NemoTranscriber {
    fn drop(&mut self) {
        if let Some(ref shutdown) = self.shutdown {
            shutdown.store(true, Ordering::SeqCst);
        }
        // Drop the audio stream and consumer
        self.audio_stream = None;
        self.ring_consumer = None;
    }
}
