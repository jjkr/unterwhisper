use serde::{Deserialize, Serialize};

/// Shared configuration for Whisper models
/// 
/// This configuration is used by both WhisperTransformer (Candle-based)
/// and OnnxTranscriber (ONNX Runtime-based) to maintain compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperConfig {
    /// Number of mel frequency bins in the input spectrogram
    pub num_mel_bins: usize,
    
    /// Size of the vocabulary (number of tokens)
    pub vocab_size: usize,
    
    /// Maximum sequence length for generated tokens
    pub max_length: usize,
    
    /// Number of encoder layers in the model
    pub num_encoder_layers: usize,
    
    /// Number of decoder layers in the model
    pub num_decoder_layers: usize,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        // Default configuration based on Whisper tiny model
        Self {
            num_mel_bins: 80,
            vocab_size: 51865,
            max_length: 448,
            num_encoder_layers: 4,
            num_decoder_layers: 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = WhisperConfig::default();
        assert_eq!(config.num_mel_bins, 80);
        assert_eq!(config.vocab_size, 51865);
        assert_eq!(config.max_length, 448);
        assert_eq!(config.num_encoder_layers, 4);
        assert_eq!(config.num_decoder_layers, 4);
    }

    #[test]
    fn test_json_serialization() {
        let config = WhisperConfig {
            num_mel_bins: 128,
            vocab_size: 51866,
            max_length: 224,
            num_encoder_layers: 6,
            num_decoder_layers: 6,
        };

        // Serialize to JSON
        let json = serde_json::to_string(&config).unwrap();
        
        // Deserialize back
        let deserialized: WhisperConfig = serde_json::from_str(&json).unwrap();
        
        assert_eq!(deserialized.num_mel_bins, 128);
        assert_eq!(deserialized.vocab_size, 51866);
        assert_eq!(deserialized.max_length, 224);
        assert_eq!(deserialized.num_encoder_layers, 6);
        assert_eq!(deserialized.num_decoder_layers, 6);
    }

    #[test]
    fn test_json_deserialization_from_config_format() {
        // Test deserialization from a typical config.json format
        let json = r#"{
            "num_mel_bins": 80,
            "vocab_size": 51865,
            "max_length": 448,
            "num_encoder_layers": 4,
            "num_decoder_layers": 4
        }"#;

        let config: WhisperConfig = serde_json::from_str(json).unwrap();
        
        assert_eq!(config.num_mel_bins, 80);
        assert_eq!(config.vocab_size, 51865);
        assert_eq!(config.max_length, 448);
        assert_eq!(config.num_encoder_layers, 4);
        assert_eq!(config.num_decoder_layers, 4);
    }
}
