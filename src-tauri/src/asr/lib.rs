//! Whisper Transcribe Library
//! 
//! A high-performance Rust library for real-time audio transcription using OpenAI's Whisper model.

pub mod audio;
pub mod dsp;
pub mod transcribe;
pub mod whisper;

pub use audio::{AudioChunk, AudioRecorder, SAMPLE_RATE, CHANNELS};
pub use transcribe::{RealtimeTranscriber, TranscriberConfig, TranscriptionMerger, TranscriptionResult};
pub use whisper::WhisperTransformer;
