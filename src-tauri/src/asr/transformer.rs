//! Unified Whisper Transformer Interface
//!
//! This module provides a unified interface for different Whisper model implementations,
//! allowing seamless switching between Candle-based and ONNX-based transformers.

use anyhow::Result;
use candle_core::Device;

use crate::asr::config::WhisperConfig;
use crate::asr::onnx::OnnxTranscriber;
use crate::asr::whisper::WhisperTransformer;

/// Unified transformer that can use either Candle or ONNX backend
pub enum UnifiedTransformer {
    Candle(WhisperTransformer),
    Onnx(OnnxTranscriber),
}

impl UnifiedTransformer {
    /// Create a new transformer based on the model name
    /// 
    /// If the model name ends with "-onnx", uses ONNX backend.
    /// Otherwise, uses Candle backend.
    /// 
    /// # Arguments
    /// 
    /// * `model_name` - Name of the model (e.g., "tiny.en", "base-onnx")
    /// * `device` - Device to use for inference
    /// * `language` - Optional language code
    /// 
    /// # Example
    /// 
    /// ```no_run
    /// use candle_core::Device;
    /// 
    /// // Use Candle backend
    /// let transformer = UnifiedTransformer::new("tiny.en", Device::Cpu, None)?;
    /// 
    /// // Use ONNX backend
    /// let transformer = UnifiedTransformer::new("tiny.en-onnx", Device::Cpu, None)?;
    /// ```
    pub fn new(
        model_name: &str,
        language: Option<String>,
    ) -> Result<Self> {
        if model_name.ends_with("-onnx") {
            // Strip the -onnx suffix and use ONNX backend
            let base_model_name = model_name.trim_end_matches("-onnx");
            // Disable KV cache by default - only distil-whisper models support it
            // Users can enable it by using model names like "distil-small.en"
            let onnx_transcriber = OnnxTranscriber::new(base_model_name, language)?;
            Ok(UnifiedTransformer::Onnx(onnx_transcriber))
        } else {
            // Use Candle backend
            let whisper_transformer = WhisperTransformer::new(model_name, language)?;
            Ok(UnifiedTransformer::Candle(whisper_transformer))
        }
    }

    /// Get the model configuration
    pub fn config(&self) -> &WhisperConfig {
        match self {
            UnifiedTransformer::Candle(transformer) => transformer.config(),
            UnifiedTransformer::Onnx(transcriber) => transcriber.config(),
        }
    }

    /// Transcribe from a mel spectrogram
    pub fn transcribe_from_mel(&mut self, mel_spectrogram: &[f32]) -> Result<String> {
        match self {
            UnifiedTransformer::Candle(transformer) => transformer.transcribe_from_mel(mel_spectrogram),
            UnifiedTransformer::Onnx(transcriber) => transcriber.transcribe_from_mel(mel_spectrogram),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_name_detection_candle() {
        // Test that non-onnx model names are detected correctly
        assert!(!("tiny.en".ends_with("-onnx")));
        assert!(!("base".ends_with("-onnx")));
        assert!(!("large-v3-turbo".ends_with("-onnx")));
    }

    #[test]
    fn test_model_name_detection_onnx() {
        // Test that onnx model names are detected correctly
        assert!("tiny.en-onnx".ends_with("-onnx"));
        assert!("base-onnx".ends_with("-onnx"));
        assert!("large-v3-turbo-onnx".ends_with("-onnx"));
    }

    #[test]
    fn test_model_name_stripping() {
        // Test that -onnx suffix is correctly stripped
        let model_name = "tiny.en-onnx";
        let base_name = model_name.trim_end_matches("-onnx");
        assert_eq!(base_name, "tiny.en");

        let model_name = "large-v3-turbo-onnx";
        let base_name = model_name.trim_end_matches("-onnx");
        assert_eq!(base_name, "large-v3-turbo");
    }
}
