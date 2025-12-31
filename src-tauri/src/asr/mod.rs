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

//    // Configure the transcriber with default settings
//    let config = TranscriberConfig::default();
//
//    // Create transcriber with Metal device for macOS
//    let device = Device::new_metal(0)?;
//    let mut transcriber = RealtimeTranscriber::new(config, device)?;
//
//    // Start transcription
//    transcriber.start();
//
//    println!("🎙️  Transcription started. Speak into your microphone...\n");
//
//    // Print transcribed text as it arrives
//    while let Some(result) = transcriber.next_transcription() {
//        println!("{}", result.text);
//    }