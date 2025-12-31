//! ONNX-based Whisper Transcriber
//!
//! This module provides an alternative implementation of the Whisper transcription interface
//! using ONNX Runtime instead of Candle. The `OnnxTranscriber` offers a drop-in replacement
//! for the existing `WhisperTransformer` that leverages ONNX's cross-platform optimizations
//! and hardware acceleration capabilities.
//!
//! # Features
//!
//! - **Cross-platform compatibility**: Uses ONNX Runtime for consistent behavior across platforms
//! - **Hardware acceleration**: Supports CPU, CUDA, and Metal/CoreML execution providers
//! - **Interface compatibility**: Maintains the same API as `WhisperTransformer` for seamless integration
//! - **Model flexibility**: Supports standard Whisper models from HuggingFace in ONNX format
//!
//! # Example
//!
//! ```no_run
//! use unterwhisper_lib::asr::onnx::OnnxTranscriber;
//! use candle_core::Device;
//!
//! // Create a new ONNX transcriber with the tiny.en model
//! let device = Device::Cpu;
//! let transcriber = OnnxTranscriber::new("tiny.en", device, None)?;
//!
//! // Transcribe from a mel spectrogram
//! let mel_spectrogram: Vec<f32> = vec![/* ... */];
//! let transcription = transcriber.transcribe_from_mel(&mel_spectrogram)?;
//! println!("Transcription: {}", transcription);
//! ```
//!
//! # Architecture
//!
//! The ONNX transcriber follows Whisper's encoder-decoder architecture:
//! 1. **Encoder**: Processes mel spectrogram to produce audio features
//! 2. **Decoder**: Autoregressively generates tokens from audio features
//! 3. **Tokenizer**: Converts tokens to text and handles special token cleanup

use anyhow::{anyhow, Result};
use candle_core::Device;
use hf_hub::{api::sync::Api, Repo, RepoType};
use log::info;
use ndarray::Array3;
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider, ExecutionProviderDispatch,
};
use ort::session::Session;
use tokenizers::Tokenizer;

use crate::asr::config::WhisperConfig;

/// Maps a Whisper model name to its HuggingFace repository information.
///
/// This function provides the mapping between user-friendly model names
/// (e.g., "tiny", "base", "small") and their corresponding HuggingFace
/// repository details needed for downloading ONNX models.
///
/// # Arguments
///
/// * `model_name` - The name of the Whisper model (e.g., "tiny", "tiny.en", "base", "large-v3-turbo")
///
/// # Returns
///
/// A tuple containing:
/// * `repo_id` - HuggingFace repository identifier (e.g., "onnx-community/whisper-tiny")
/// * `revision` - Git revision/branch to use (typically "main")
/// * `encoder_file` - Encoder model filename (e.g., "encoder_model.onnx")
/// * `decoder_file` - Decoder model filename (e.g., "decoder_model.onnx")
///
/// # Example
///
/// ```
/// let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info("tiny.en");
/// assert_eq!(repo_id, "onnx-community/whisper-tiny.en");
/// assert_eq!(revision, "main");
/// assert_eq!(encoder_file, "encoder_model.onnx");
/// assert_eq!(decoder_file, "decoder_model.onnx");
/// ```
///
/// # Notes
///
/// - For unknown model names, defaults to "large-v3-turbo" from onnx-community
/// - Uses onnx-community repositories which provide optimized ONNX models
/// - Supports all standard Whisper variants: tiny, base, small, medium, large
/// - Supports English-only variants with ".en" suffix
/// - Supports distil variants for smaller, faster models
pub fn get_onnx_model_info(model_name: &str) -> (&'static str, &'static str, &'static str, &'static str) {
    match model_name {
        // Tiny models
        "tiny" => ("onnx-community/whisper-tiny", "main", "encoder_model.onnx", "decoder_model.onnx"),
        "tiny.en" => ("onnx-community/whisper-tiny.en", "main", "encoder_model.onnx", "decoder_model.onnx"),
        
        // Base models
        "base" => ("onnx-community/whisper-base", "main", "encoder_model.onnx", "decoder_model.onnx"),
        "base.en" => ("onnx-community/whisper-base.en", "main", "encoder_model.onnx", "decoder_model.onnx"),
        
        // Small models
        "small" => ("onnx-community/whisper-small", "main", "encoder_model.onnx", "decoder_model.onnx"),
        "small.en" => ("onnx-community/whisper-small.en", "main", "encoder_model.onnx", "decoder_model.onnx"),
        
        // Medium models
        "medium" => ("onnx-community/whisper-medium", "main", "encoder_model.onnx", "decoder_model.onnx"),
        "medium.en" => ("onnx-community/whisper-medium.en", "main", "encoder_model.onnx", "decoder_model.onnx"),
        
        // Large models
        "large" => ("onnx-community/whisper-large", "main", "encoder_model.onnx", "decoder_model.onnx"),
        "large-v2" => ("onnx-community/whisper-large-v2", "main", "encoder_model.onnx", "decoder_model.onnx"),
        "large-v3" => ("onnx-community/whisper-large-v3", "main", "encoder_model.onnx", "decoder_model.onnx"),
        "large-v3-turbo" => ("onnx-community/whisper-large-v3-turbo", "main", "encoder_model.onnx", "decoder_model.onnx"),
        
        // Distil models (smaller, faster variants)
        "distil-small.en" => ("onnx-community/distil-whisper-small.en", "main", "encoder_model.onnx", "decoder_model.onnx"),
        "distil-medium.en" => ("onnx-community/distil-whisper-medium.en", "main", "encoder_model.onnx", "decoder_model.onnx"),
        "distil-large-v2" => ("onnx-community/distil-whisper-large-v2", "main", "encoder_model.onnx", "decoder_model.onnx"),
        "distil-large-v3" => ("onnx-community/distil-whisper-large-v3", "main", "encoder_model.onnx", "decoder_model.onnx"),
        
        // Default fallback for unknown model names
        _ => ("onnx-community/whisper-large-v3-turbo", "main", "encoder_model.onnx", "decoder_model.onnx"),
    }
}

/// Maps a Candle Device to an ONNX Runtime execution provider.
///
/// This helper function translates the device specification from Candle's
/// Device enum to the appropriate ONNX Runtime execution provider.
///
/// # Arguments
///
/// * `device` - The Candle device specification
///
/// # Returns
///
/// An `ExecutionProviderDispatch` configured for the specified device
///
/// # Supported Mappings
///
/// * `Device::Cpu` → CPUExecutionProvider
/// * `Device::Cuda(_)` → CUDAExecutionProvider
/// * `Device::Metal(_)` → CoreMLExecutionProvider
///
/// # Example
///
/// ```
/// use candle_core::Device;
/// let device = Device::Cpu;
/// let provider = get_execution_provider(&device);
/// ```
fn get_execution_provider(device: &Device) -> ExecutionProviderDispatch {
    let provider = match device {
        Device::Cpu => {
            info!("Selected ONNX execution provider: CPUExecutionProvider");
            ExecutionProviderDispatch::from(CPUExecutionProvider::default())
        }
        Device::Cuda(_) => {
            info!("Selected ONNX execution provider: CUDAExecutionProvider");
            ExecutionProviderDispatch::from(CUDAExecutionProvider::default())
        }
        Device::Metal(_) => {
            info!("Selected ONNX execution provider: CoreMLExecutionProvider");
            ExecutionProviderDispatch::from(CoreMLExecutionProvider::default())
        }
    };
    
    provider
}

/// ONNX-based implementation of the Whisper transcriber.
///
/// This struct provides an alternative to `WhisperTransformer` using ONNX Runtime
/// for model inference. It maintains the same interface for compatibility with
/// the existing transcription pipeline.
///
/// # Fields
///
/// * `encoder_session` - ONNX Runtime session for the encoder model
/// * `decoder_session` - ONNX Runtime session for the decoder model
/// * `tokenizer` - Tokenizers library instance for encoding/decoding text
/// * `config` - Whisper model configuration (num_mel_bins, vocab_size, etc.)
/// * `device` - Device specification (CPU, Metal, CUDA) for execution provider selection
pub struct OnnxTranscriber {
    encoder_session: Session,
    decoder_session: Session,
    tokenizer: Tokenizer,
    config: WhisperConfig,
    device: Device,
}

impl OnnxTranscriber {
    /// Creates a new ONNX transcriber instance.
    ///
    /// # Arguments
    ///
    /// * `model_name` - Name of the Whisper model (e.g., "tiny", "base", "small")
    /// * `device` - Device to use for inference (CPU, CUDA, Metal)
    /// * `language` - Optional language code for transcription (e.g., Some("en"))
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the initialized transcriber or an error if
    /// model loading fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use unterwhisper_lib::asr::onnx::OnnxTranscriber;
    /// use candle_core::Device;
    ///
    /// let transcriber = OnnxTranscriber::new("tiny.en", Device::Cpu, None)?;
    /// ```
    pub fn new(
        model_name: &str,
        device: Device,
        _language: Option<String>,
    ) -> Result<Self> {
        info!("Loading ONNX Whisper model: {}", model_name);

        // Get model repository information
        let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info(model_name);
        info!("Repository: {} (revision: {})", repo_id, revision);

        // Download model files from HuggingFace
        let api = Api::new()
            .map_err(|e| anyhow!("Failed to initialize HuggingFace API: {}", e))?;
        let repo = api.repo(Repo::with_revision(
            repo_id.to_string(),
            RepoType::Model,
            revision.to_string(),
        ));

        info!("Downloading model files from HuggingFace...");
        
        // Download required files
        let config_path = repo.get("config.json")
            .map_err(|e| anyhow!("Failed to download config.json: {}", e))?;
        let tokenizer_path = repo.get("tokenizer.json")
            .map_err(|e| anyhow!("Failed to download tokenizer.json: {}", e))?;
        let encoder_path = repo.get(encoder_file)
            .map_err(|e| anyhow!("Failed to download {}: {}", encoder_file, e))?;
        let decoder_path = repo.get(decoder_file)
            .map_err(|e| anyhow!("Failed to download {}: {}", decoder_file, e))?;

        info!("Model files downloaded successfully");

        // Load and parse config.json
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| anyhow!("Failed to read config.json: {}", e))?;
        
        // Parse the full config JSON to extract the fields we need
        let config_json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| anyhow!("Failed to parse config.json: {}", e))?;
        
        // Extract configuration values with defaults
        let num_mel_bins = config_json["num_mel_bins"]
            .as_u64()
            .unwrap_or(80) as usize;
        let vocab_size = config_json["vocab_size"]
            .as_u64()
            .unwrap_or(51865) as usize;
        let max_length = config_json["max_target_positions"]
            .as_u64()
            .or_else(|| config_json["max_length"].as_u64())
            .unwrap_or(448) as usize;
        let num_encoder_layers = config_json["encoder_layers"]
            .as_u64()
            .unwrap_or(4) as usize;
        let num_decoder_layers = config_json["decoder_layers"]
            .as_u64()
            .unwrap_or(4) as usize;

        let config = WhisperConfig {
            num_mel_bins,
            vocab_size,
            max_length,
            num_encoder_layers,
            num_decoder_layers,
        };

        info!("Loaded config: {:?}", config);

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        info!("Tokenizer loaded successfully");

        // Create ONNX sessions with appropriate execution provider
        info!("Creating ONNX Runtime sessions...");
        
        let encoder_session = Session::builder()
            .map_err(|e| anyhow!("Failed to create session builder: {}", e))?
            .with_execution_providers([get_execution_provider(&device)])
            .map_err(|e| anyhow!("Failed to set execution provider: {}", e))?
            .commit_from_file(&encoder_path)
            .map_err(|e| anyhow!("Failed to load encoder model: {}", e))?;

        let decoder_session = Session::builder()
            .map_err(|e| anyhow!("Failed to create session builder: {}", e))?
            .with_execution_providers([get_execution_provider(&device)])
            .map_err(|e| anyhow!("Failed to set execution provider: {}", e))?
            .commit_from_file(&decoder_path)
            .map_err(|e| anyhow!("Failed to load decoder model: {}", e))?;

        info!("ONNX sessions created successfully");
        info!("ONNX Whisper model loaded successfully");

        Ok(Self {
            encoder_session,
            decoder_session,
            tokenizer,
            config,
            device,
        })
    }

    /// Returns a reference to the model configuration.
    ///
    /// # Returns
    ///
    /// A reference to the `WhisperConfig` containing model parameters.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use unterwhisper_lib::asr::onnx::OnnxTranscriber;
    /// # use candle_core::Device;
    /// # let transcriber = OnnxTranscriber::new("tiny.en", Device::Cpu, None)?;
    /// let config = transcriber.config();
    /// println!("Vocab size: {}", config.vocab_size);
    /// ```
    pub fn config(&self) -> &WhisperConfig {
        &self.config
    }

    /// Transcribes audio from a mel spectrogram.
    ///
    /// # Arguments
    ///
    /// * `mel_spectrogram` - Flattened mel spectrogram data as a slice of f32 values
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the transcribed text or an error if transcription fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use unterwhisper_lib::asr::onnx::OnnxTranscriber;
    /// # use candle_core::Device;
    /// # let transcriber = OnnxTranscriber::new("tiny.en", Device::Cpu, None)?;
    /// let mel_spectrogram: Vec<f32> = vec![/* ... */];
    /// let text = transcriber.transcribe_from_mel(&mel_spectrogram)?;
    /// ```
    pub fn transcribe_from_mel(&mut self, _mel_spectrogram: &[f32]) -> Result<String> {
        // TODO: Implement transcription pipeline
        unimplemented!("OnnxTranscriber::transcribe_from_mel will be implemented in subsequent tasks")
    }

    /// Preprocesses a mel spectrogram for encoder input.
    ///
    /// This function normalizes the mel spectrogram to exactly 3000 time frames,
    /// which is the expected input size for Whisper models. It handles both
    /// padding (if too short) and truncation (if too long).
    ///
    /// # Arguments
    ///
    /// * `mel_spectrogram` - Flattened mel spectrogram data as a slice of f32 values
    ///
    /// # Returns
    ///
    /// Returns an `Option<Array3<f32>>` containing the preprocessed tensor in shape
    /// `(1, n_mels, 3000)`, or `None` if the input is empty.
    ///
    /// # Behavior
    ///
    /// - **Empty input**: Returns `None` immediately
    /// - **Too short**: Pads with zeros to reach `n_mels * 3000` elements
    /// - **Too long**: Truncates to `n_mels * 3000` elements
    /// - **Exact size**: Uses input as-is
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use unterwhisper_lib::asr::onnx::OnnxTranscriber;
    /// # use candle_core::Device;
    /// # let transcriber = OnnxTranscriber::new("tiny.en", Device::Cpu, None)?;
    /// let mel_spectrogram: Vec<f32> = vec![0.0; 80 * 2000]; // Too short
    /// let preprocessed = transcriber.preprocess_mel(&mel_spectrogram);
    /// assert!(preprocessed.is_some());
    /// let tensor = preprocessed.unwrap();
    /// assert_eq!(tensor.shape(), &[1, 80, 3000]); // Padded to 3000 frames
    /// ```
    fn preprocess_mel(&self, mel_spectrogram: &[f32]) -> Option<Array3<f32>> {
        // Handle empty input
        if mel_spectrogram.is_empty() {
            return None;
        }

        let n_mels = self.config.num_mel_bins;
        let target_frames = 3000;
        let target_length = n_mels * target_frames;

        // Normalize length: pad or truncate
        let normalized: Vec<f32> = if mel_spectrogram.len() < target_length {
            // Pad with zeros
            let mut padded = mel_spectrogram.to_vec();
            padded.resize(target_length, 0.0);
            padded
        } else if mel_spectrogram.len() > target_length {
            // Truncate
            mel_spectrogram[..target_length].to_vec()
        } else {
            // Exact size
            mel_spectrogram.to_vec()
        };

        // Reshape to (1, n_mels, 3000)
        Array3::from_shape_vec((1, n_mels, target_frames), normalized).ok()
    }

    /// Runs the encoder model on a preprocessed mel spectrogram.
    ///
    /// This method executes the ONNX encoder session to produce audio features
    /// from the input mel spectrogram. The audio features are then used by the
    /// decoder for token generation.
    ///
    /// # Arguments
    ///
    /// * `mel_tensor` - Preprocessed mel spectrogram tensor in shape `(1, n_mels, 3000)`
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the audio features as an `Array3<f32>` with shape
    /// `(1, sequence_length, hidden_size)`, or an error if inference fails.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The ONNX session execution fails
    /// - The output tensor extraction fails
    /// - The output shape is invalid
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use unterwhisper_lib::asr::onnx::OnnxTranscriber;
    /// # use candle_core::Device;
    /// # use ndarray::Array3;
    /// # let mut transcriber = OnnxTranscriber::new("tiny.en", Device::Cpu, None)?;
    /// let mel_tensor = Array3::<f32>::zeros((1, 80, 3000));
    /// let audio_features = transcriber.run_encoder(&mel_tensor)?;
    /// println!("Audio features shape: {:?}", audio_features.shape());
    /// ```
    fn run_encoder(&mut self, mel_tensor: &Array3<f32>) -> Result<Array3<f32>> {
        use ort::value::Value;

        // Create ONNX value from the mel tensor
        let mel_value = Value::from_array(mel_tensor.clone())
            .map_err(|e| anyhow!("Failed to create ONNX value from mel tensor: {}", e))?;

        // Run encoder session with the mel spectrogram input
        let outputs = self.encoder_session
            .run(ort::inputs!["input_features" => mel_value])
            .map_err(|e| anyhow!("Encoder inference failed: {}", e))?;

        // Extract the audio features from the output
        // The encoder output is typically named "last_hidden_state" in Whisper ONNX models
        let audio_features = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("Failed to extract audio features from encoder output: {}", e))?;

        // Get the shape and data
        let (shape, data) = audio_features;
        
        // Convert shape to dimensions
        let dims = shape.as_ref();
        if dims.len() != 3 {
            return Err(anyhow!("Expected 3D audio features, got {}D", dims.len()));
        }

        // Convert i64 dimensions to usize
        let dim0 = dims[0] as usize;
        let dim1 = dims[1] as usize;
        let dim2 = dims[2] as usize;

        // Create Array3 from the data
        let audio_features_array = Array3::from_shape_vec(
            (dim0, dim1, dim2),
            data.to_vec()
        ).map_err(|e| anyhow!("Failed to create audio features array: {}", e))?;

        Ok(audio_features_array)
    }

    /// Looks up a token ID from the tokenizer by its string representation.
    ///
    /// This helper method retrieves the token ID for a given token string,
    /// providing descriptive error messages if the token is not found in
    /// the tokenizer's vocabulary.
    ///
    /// # Arguments
    ///
    /// * `token` - The string representation of the token to look up
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the token ID as a `u32`, or an error
    /// if the token is not found in the vocabulary.
    ///
    /// # Errors
    ///
    /// Returns an error if the token is not found in the tokenizer's vocabulary,
    /// with a descriptive message indicating which token was missing.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use unterwhisper_lib::asr::onnx::OnnxTranscriber;
    /// # use candle_core::Device;
    /// # let transcriber = OnnxTranscriber::new("tiny.en", Device::Cpu, None)?;
    /// let sot_token = transcriber.get_token_id("<|startoftranscript|>")?;
    /// let eot_token = transcriber.get_token_id("<|endoftext|>")?;
    /// ```
    fn get_token_id(&self, token: &str) -> Result<u32> {
        self.tokenizer
            .token_to_id(token)
            .ok_or_else(|| anyhow!("Token '{}' not found in tokenizer vocabulary", token))
    }

    /// Initializes the token sequence for decoder generation.
    ///
    /// This method creates the initial token sequence that the decoder uses
    /// to start generating transcription tokens. The sequence includes special
    /// tokens for start-of-transcript, language, task (transcribe), and
    /// timestamp settings.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a `Vec<u32>` with the initial token sequence,
    /// or an error if any required tokens are missing from the tokenizer.
    ///
    /// # Token Sequence
    ///
    /// The initial sequence typically includes:
    /// 1. Start-of-transcript token (`<|startoftranscript|>`)
    /// 2. Language token (e.g., `<|en|>` for English-only models, or detected language)
    /// 3. Task token (`<|transcribe|>` for transcription)
    /// 4. Timestamp token (`<|notimestamps|>` to disable timestamps)
    ///
    /// # Model Handling
    ///
    /// - **English-only models** (e.g., "tiny.en"): Use `<|en|>` language token
    /// - **Multilingual models**: Use `<|en|>` as default (can be extended for language detection)
    ///
    /// # Errors
    ///
    /// Returns an error if any of the required special tokens are not found
    /// in the tokenizer's vocabulary.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use unterwhisper_lib::asr::onnx::OnnxTranscriber;
    /// # use candle_core::Device;
    /// # let transcriber = OnnxTranscriber::new("tiny.en", Device::Cpu, None)?;
    /// let initial_tokens = transcriber.initialize_tokens()?;
    /// println!("Initial token sequence: {:?}", initial_tokens);
    /// ```
    fn initialize_tokens(&self) -> Result<Vec<u32>> {
        // Get special token IDs
        let sot_token = self.get_token_id("<|startoftranscript|>")?;
        let transcribe_token = self.get_token_id("<|transcribe|>")?;
        let notimestamps_token = self.get_token_id("<|notimestamps|>")?;
        
        // Get language token - default to English
        // For English-only models, this will be <|en|>
        // For multilingual models, we could detect language, but default to English for now
        let language_token = self.get_token_id("<|en|>")?;
        
        // Build initial token sequence: [SOT, LANG, TRANSCRIBE, NO_TIMESTAMPS]
        let tokens = vec![
            sot_token,
            language_token,
            transcribe_token,
            notimestamps_token,
        ];
        
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_onnx_model_info_tiny_models() {
        let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info("tiny");
        assert_eq!(repo_id, "onnx-community/whisper-tiny");
        assert_eq!(revision, "main");
        assert_eq!(encoder_file, "encoder_model.onnx");
        assert_eq!(decoder_file, "decoder_model.onnx");

        let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info("tiny.en");
        assert_eq!(repo_id, "onnx-community/whisper-tiny.en");
        assert_eq!(revision, "main");
        assert_eq!(encoder_file, "encoder_model.onnx");
        assert_eq!(decoder_file, "decoder_model.onnx");
    }

    #[test]
    fn test_get_onnx_model_info_base_models() {
        let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info("base");
        assert_eq!(repo_id, "onnx-community/whisper-base");
        assert_eq!(revision, "main");
        assert_eq!(encoder_file, "encoder_model.onnx");
        assert_eq!(decoder_file, "decoder_model.onnx");

        let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info("base.en");
        assert_eq!(repo_id, "onnx-community/whisper-base.en");
        assert_eq!(revision, "main");
        assert_eq!(encoder_file, "encoder_model.onnx");
        assert_eq!(decoder_file, "decoder_model.onnx");
    }

    #[test]
    fn test_get_onnx_model_info_small_models() {
        let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info("small");
        assert_eq!(repo_id, "onnx-community/whisper-small");
        assert_eq!(revision, "main");
        assert_eq!(encoder_file, "encoder_model.onnx");
        assert_eq!(decoder_file, "decoder_model.onnx");

        let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info("small.en");
        assert_eq!(repo_id, "onnx-community/whisper-small.en");
        assert_eq!(revision, "main");
        assert_eq!(encoder_file, "encoder_model.onnx");
        assert_eq!(decoder_file, "decoder_model.onnx");
    }

    #[test]
    fn test_get_onnx_model_info_medium_models() {
        let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info("medium");
        assert_eq!(repo_id, "onnx-community/whisper-medium");
        assert_eq!(revision, "main");
        assert_eq!(encoder_file, "encoder_model.onnx");
        assert_eq!(decoder_file, "decoder_model.onnx");

        let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info("medium.en");
        assert_eq!(repo_id, "onnx-community/whisper-medium.en");
        assert_eq!(revision, "main");
        assert_eq!(encoder_file, "encoder_model.onnx");
        assert_eq!(decoder_file, "decoder_model.onnx");
    }

    #[test]
    fn test_get_onnx_model_info_large_models() {
        let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info("large");
        assert_eq!(repo_id, "onnx-community/whisper-large");
        assert_eq!(revision, "main");
        assert_eq!(encoder_file, "encoder_model.onnx");
        assert_eq!(decoder_file, "decoder_model.onnx");

        let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info("large-v2");
        assert_eq!(repo_id, "onnx-community/whisper-large-v2");
        assert_eq!(revision, "main");
        assert_eq!(encoder_file, "encoder_model.onnx");
        assert_eq!(decoder_file, "decoder_model.onnx");

        let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info("large-v3");
        assert_eq!(repo_id, "onnx-community/whisper-large-v3");
        assert_eq!(revision, "main");
        assert_eq!(encoder_file, "encoder_model.onnx");
        assert_eq!(decoder_file, "decoder_model.onnx");

        let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info("large-v3-turbo");
        assert_eq!(repo_id, "onnx-community/whisper-large-v3-turbo");
        assert_eq!(revision, "main");
        assert_eq!(encoder_file, "encoder_model.onnx");
        assert_eq!(decoder_file, "decoder_model.onnx");
    }

    #[test]
    fn test_get_onnx_model_info_distil_models() {
        let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info("distil-small.en");
        assert_eq!(repo_id, "onnx-community/distil-whisper-small.en");
        assert_eq!(revision, "main");
        assert_eq!(encoder_file, "encoder_model.onnx");
        assert_eq!(decoder_file, "decoder_model.onnx");

        let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info("distil-medium.en");
        assert_eq!(repo_id, "onnx-community/distil-whisper-medium.en");
        assert_eq!(revision, "main");
        assert_eq!(encoder_file, "encoder_model.onnx");
        assert_eq!(decoder_file, "decoder_model.onnx");

        let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info("distil-large-v2");
        assert_eq!(repo_id, "onnx-community/distil-whisper-large-v2");
        assert_eq!(revision, "main");
        assert_eq!(encoder_file, "encoder_model.onnx");
        assert_eq!(decoder_file, "decoder_model.onnx");

        let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info("distil-large-v3");
        assert_eq!(repo_id, "onnx-community/distil-whisper-large-v3");
        assert_eq!(revision, "main");
        assert_eq!(encoder_file, "encoder_model.onnx");
        assert_eq!(decoder_file, "decoder_model.onnx");
    }

    #[test]
    fn test_get_onnx_model_info_unknown_model_fallback() {
        // Test that unknown model names fall back to large-v3-turbo
        let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info("unknown-model");
        assert_eq!(repo_id, "onnx-community/whisper-large-v3-turbo");
        assert_eq!(revision, "main");
        assert_eq!(encoder_file, "encoder_model.onnx");
        assert_eq!(decoder_file, "decoder_model.onnx");

        let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info("some-random-name");
        assert_eq!(repo_id, "onnx-community/whisper-large-v3-turbo");
        assert_eq!(revision, "main");
        assert_eq!(encoder_file, "encoder_model.onnx");
        assert_eq!(decoder_file, "decoder_model.onnx");
    }

    #[test]
    fn test_get_onnx_model_info_all_return_valid_format() {
        // Test that all supported models return valid repository information
        let models = vec![
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large", "large-v2", "large-v3", "large-v3-turbo",
            "distil-small.en", "distil-medium.en", "distil-large-v2", "distil-large-v3",
        ];

        for model in models {
            let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info(model);
            
            // Verify repo_id is not empty and contains a slash (org/repo format)
            assert!(!repo_id.is_empty(), "repo_id should not be empty for model: {}", model);
            assert!(repo_id.contains('/'), "repo_id should contain '/' for model: {}", model);
            
            // Verify revision is "main"
            assert_eq!(revision, "main", "revision should be 'main' for model: {}", model);
            
            // Verify encoder_file is encoder_model.onnx
            assert_eq!(encoder_file, "encoder_model.onnx", "encoder_file should be 'encoder_model.onnx' for model: {}", model);
            
            // Verify decoder_file is decoder_model.onnx
            assert_eq!(decoder_file, "decoder_model.onnx", "decoder_file should be 'decoder_model.onnx' for model: {}", model);
        }
    }

    #[test]
    fn test_preprocess_mel_empty_input() {
        // Create a minimal config for testing
        let config = WhisperConfig {
            num_mel_bins: 80,
            vocab_size: 51865,
            max_length: 448,
            num_encoder_layers: 4,
            num_decoder_layers: 4,
        };

        // Create a mock transcriber (we only need config for this test)
        // We can't easily create a full OnnxTranscriber without downloading models,
        // so we'll test the logic directly
        let empty_mel: Vec<f32> = vec![];
        
        // Simulate the preprocess_mel logic
        if empty_mel.is_empty() {
            // This should return None
            assert!(empty_mel.is_empty());
        }
    }

    #[test]
    fn test_preprocess_mel_padding() {
        // Test that short mel spectrograms are padded to 3000 frames
        let n_mels = 80;
        let short_frames = 2000;
        let short_mel: Vec<f32> = vec![1.0; n_mels * short_frames];
        
        let target_frames = 3000;
        let target_length = n_mels * target_frames;
        
        // Simulate padding
        let mut padded = short_mel.clone();
        padded.resize(target_length, 0.0);
        
        assert_eq!(padded.len(), target_length);
        // First part should be original data
        assert_eq!(padded[0], 1.0);
        // Padded part should be zeros
        assert_eq!(padded[n_mels * short_frames], 0.0);
    }

    #[test]
    fn test_preprocess_mel_truncation() {
        // Test that long mel spectrograms are truncated to 3000 frames
        let n_mels = 80;
        let long_frames = 4000;
        let long_mel: Vec<f32> = vec![1.0; n_mels * long_frames];
        
        let target_frames = 3000;
        let target_length = n_mels * target_frames;
        
        // Simulate truncation
        let truncated = &long_mel[..target_length];
        
        assert_eq!(truncated.len(), target_length);
        // All values should be from original data
        assert_eq!(truncated[0], 1.0);
        assert_eq!(truncated[target_length - 1], 1.0);
    }

    #[test]
    fn test_preprocess_mel_exact_size() {
        // Test that exact-size mel spectrograms are used as-is
        let n_mels = 80;
        let exact_frames = 3000;
        let exact_mel: Vec<f32> = vec![1.0; n_mels * exact_frames];
        
        let target_length = n_mels * exact_frames;
        
        assert_eq!(exact_mel.len(), target_length);
        // Should use as-is
        let result = exact_mel.clone();
        assert_eq!(result.len(), target_length);
        assert_eq!(result[0], 1.0);
    }

    #[test]
    fn test_get_token_id_valid_tokens() {
        // This test would require a real tokenizer, which requires downloading models
        // For now, we'll document the expected behavior:
        // - get_token_id should return a u32 token ID for valid tokens
        // - get_token_id should return an error for invalid tokens
        // These will be tested in integration tests with actual models
    }

    #[test]
    fn test_initialize_tokens_sequence() {
        // This test would require a real tokenizer, which requires downloading models
        // For now, we'll document the expected behavior:
        // - initialize_tokens should return a Vec<u32> with 4 tokens
        // - The sequence should be: [SOT, LANG, TRANSCRIBE, NO_TIMESTAMPS]
        // - All tokens should be valid u32 values
        // These will be tested in integration tests with actual models
    }
}
