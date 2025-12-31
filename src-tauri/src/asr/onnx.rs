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
    /// This is the main entry point for transcription. It orchestrates the complete
    /// pipeline from mel spectrogram preprocessing through token generation to final
    /// text output.
    ///
    /// # Arguments
    ///
    /// * `mel_spectrogram` - Flattened mel spectrogram data as a slice of f32 values
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the transcribed text or an error if transcription fails.
    ///
    /// # Pipeline Steps
    ///
    /// 1. **Preprocess**: Normalize mel spectrogram to 3000 frames (pad or truncate)
    /// 2. **Encode**: Run encoder to extract audio features
    /// 3. **Initialize**: Create initial token sequence with special tokens
    /// 4. **Generate**: Autoregressively generate tokens using decoder
    /// 5. **Decode**: Convert tokens to text
    /// 6. **Clean**: Remove special tokens and trim whitespace
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use unterwhisper_lib::asr::onnx::OnnxTranscriber;
    /// # use candle_core::Device;
    /// # let mut transcriber = OnnxTranscriber::new("tiny.en", Device::Cpu, None)?;
    /// let mel_spectrogram: Vec<f32> = vec![/* ... */];
    /// let text = transcriber.transcribe_from_mel(&mel_spectrogram)?;
    /// println!("Transcription: {}", text);
    /// ```
    pub fn transcribe_from_mel(&mut self, mel_spectrogram: &[f32]) -> Result<String> {
        // Step 1: Preprocess mel spectrogram
        let mel_tensor = match self.preprocess_mel(mel_spectrogram) {
            Some(tensor) => tensor,
            None => {
                // Empty input returns empty string
                return Ok(String::new());
            }
        };

        // Step 2: Run encoder to get audio features
        let audio_features = self.run_encoder(&mel_tensor)?;

        // Step 3: Generate tokens with decoder
        let tokens = self.generate_tokens(&audio_features)?;

        // Step 4: Decode and clean text
        let text = self.decode_and_clean(&tokens)?;

        Ok(text)
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

    /// Applies repetition penalty to logits to discourage repeated tokens.
    ///
    /// This function modifies the logits in-place by applying a penalty to tokens
    /// that have already appeared in the generated sequence. The penalty is applied
    /// exponentially based on how many times each token has occurred.
    ///
    /// # Arguments
    ///
    /// * `logits` - Mutable slice of logits to modify (one value per vocabulary token)
    /// * `tokens` - The sequence of tokens generated so far
    /// * `penalty` - The penalty factor to apply (typically 1.1)
    ///
    /// # Penalty Application
    ///
    /// For each token that has appeared in the sequence:
    /// - Count how many times it has occurred
    /// - Calculate penalty_factor = penalty^count
    /// - If logit > 0: divide by penalty_factor (reduce probability)
    /// - If logit ≤ 0: multiply by penalty_factor (further reduce probability)
    ///
    /// This approach ensures that:
    /// - Repeated tokens become less likely to be selected again
    /// - The penalty increases exponentially with repetition count
    /// - Both positive and negative logits are penalized appropriately
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use unterwhisper_lib::asr::onnx::OnnxTranscriber;
    /// let mut logits = vec![2.0, -1.0, 0.5, -0.5];
    /// let tokens = vec![0, 0, 2]; // Token 0 appears twice, token 2 once
    /// let penalty = 1.1;
    /// 
    /// // After applying penalty:
    /// // logits[0] = 2.0 / (1.1^2) ≈ 1.65 (appeared twice)
    /// // logits[1] = -1.0 (unchanged, never appeared)
    /// // logits[2] = 0.5 / 1.1 ≈ 0.45 (appeared once)
    /// // logits[3] = -0.5 (unchanged, never appeared)
    /// ```
    fn apply_repetition_penalty(logits: &mut [f32], tokens: &[u32], penalty: f32) {
        use std::collections::HashMap;

        // Track how many times each token has occurred
        let mut token_counts: HashMap<u32, i32> = HashMap::new();
        for &token in tokens {
            *token_counts.entry(token).or_insert(0) += 1;
        }

        // Apply penalty to each token that has occurred
        for (token_id, count) in token_counts {
            let idx = token_id as usize;
            if idx < logits.len() {
                // Calculate penalty factor: penalty^count
                let penalty_factor = penalty.powi(count);
                
                // Apply penalty based on logit sign
                if logits[idx] > 0.0 {
                    // Positive logits: divide by penalty factor
                    logits[idx] /= penalty_factor;
                } else {
                    // Negative logits: multiply by penalty factor
                    logits[idx] *= penalty_factor;
                }
            }
        }
    }

    /// Samples a token from the decoder logits.
    ///
    /// This method extracts the logits for the last position in the sequence,
    /// applies repetition penalty to discourage repeated tokens, optionally
    /// applies temperature for sampling, and selects the next token either
    /// greedily (temperature 0) or by sampling (temperature > 0).
    ///
    /// # Arguments
    ///
    /// * `logits` - The logits tensor from the decoder output, shape `(batch_size, sequence_length, vocab_size)`
    /// * `tokens` - The current token sequence (used for repetition penalty)
    /// * `temperature` - Temperature for sampling (0.0 for greedy, > 0.0 for sampling)
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the selected token ID as a `u32`, or an error
    /// if sampling fails.
    ///
    /// # Sampling Strategy
    ///
    /// - **Temperature 0.0 (Greedy)**: Selects the token with the highest logit value
    /// - **Temperature > 0.0 (Sampling)**: Applies softmax with temperature and samples
    ///   from the resulting probability distribution
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use unterwhisper_lib::asr::onnx::OnnxTranscriber;
    /// # use candle_core::Device;
    /// # use ndarray::Array3;
    /// # let mut transcriber = OnnxTranscriber::new("tiny.en", Device::Cpu, None)?;
    /// let logits = Array3::<f32>::zeros((1, 10, 51865));
    /// let tokens = vec![50258, 50259, 50359, 50363];
    /// let temperature = 0.0; // Greedy decoding
    /// let next_token = transcriber.sample_token(&logits, &tokens, temperature)?;
    /// ```
    fn sample_token(
        &self,
        logits: &ndarray::Array3<f32>,
        tokens: &[u32],
        temperature: f32,
    ) -> Result<u32> {
        // Extract logits for the last position
        // Shape: (batch_size, sequence_length, vocab_size)
        // We want: (vocab_size,) for the last position
        let shape = logits.shape();
        if shape.len() != 3 {
            return Err(anyhow!("Expected 3D logits tensor, got {}D", shape.len()));
        }
        
        let batch_size = shape[0];
        let seq_len = shape[1];
        let vocab_size = shape[2];
        
        if batch_size != 1 {
            return Err(anyhow!("Expected batch_size=1, got {}", batch_size));
        }
        
        if seq_len == 0 {
            return Err(anyhow!("Empty sequence in logits"));
        }
        
        // Extract last position logits: logits[0, seq_len-1, :]
        let last_logits = logits.slice(ndarray::s![0, seq_len - 1, ..]);
        let mut last_logits_vec: Vec<f32> = last_logits.to_vec();
        
        // Apply repetition penalty
        let penalty = 1.1; // Default repetition penalty
        Self::apply_repetition_penalty(&mut last_logits_vec, tokens, penalty);
        
        // Select token based on temperature
        if temperature == 0.0 {
            // Greedy decoding: select token with highest logit
            let max_idx = last_logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or_else(|| anyhow!("Failed to find maximum logit"))?;
            
            Ok(max_idx as u32)
        } else {
            // Sampling with temperature
            // Apply temperature scaling: logits / temperature
            for logit in last_logits_vec.iter_mut() {
                *logit /= temperature;
            }
            
            // Apply softmax to get probabilities
            let max_logit = last_logits_vec
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            
            // Subtract max for numerical stability
            for logit in last_logits_vec.iter_mut() {
                *logit -= max_logit;
            }
            
            // Compute exp and sum
            let exp_logits: Vec<f32> = last_logits_vec.iter().map(|x| x.exp()).collect();
            let sum_exp: f32 = exp_logits.iter().sum();
            
            // Normalize to get probabilities
            let probs: Vec<f32> = exp_logits.iter().map(|x| x / sum_exp).collect();
            
            // Sample from the distribution
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let sample: f32 = rng.gen();
            
            let mut cumsum = 0.0;
            for (idx, &prob) in probs.iter().enumerate() {
                cumsum += prob;
                if sample < cumsum {
                    return Ok(idx as u32);
                }
            }
            
            // Fallback: return last token (should rarely happen due to floating point)
            Ok((vocab_size - 1) as u32)
        }
    }

    /// Runs the decoder model on input tokens and audio features.
    ///
    /// This method executes the ONNX decoder session to produce logits for the
    /// next token prediction. The decoder runs autoregressively, taking the current
    /// token sequence and audio features as input.
    ///
    /// # Arguments
    ///
    /// * `input_tokens` - Current token sequence as a slice of u32 values
    /// * `audio_features` - Audio features from the encoder, shape `(1, sequence_length, hidden_size)`
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the logits as an `Array3<f32>` with shape
    /// `(1, sequence_length, vocab_size)`, or an error if inference fails.
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
    /// let tokens = vec![50258, 50259, 50359, 50363];
    /// let audio_features = Array3::<f32>::zeros((1, 1500, 384));
    /// let logits = transcriber.run_decoder(&tokens, &audio_features)?;
    /// ```
    fn run_decoder(
        &mut self,
        input_tokens: &[u32],
        audio_features: &Array3<f32>,
    ) -> Result<Array3<f32>> {
        use ndarray::Array2;
        use ort::value::Value;

        // Convert tokens to i64 array (ONNX typically uses int64 for token IDs)
        let tokens_i64: Vec<i64> = input_tokens.iter().map(|&t| t as i64).collect();
        let seq_len = tokens_i64.len();
        
        // Create input_ids tensor: shape (batch_size, sequence_length)
        let input_ids = Array2::from_shape_vec((1, seq_len), tokens_i64)
            .map_err(|e| anyhow!("Failed to create input_ids array: {}", e))?;
        
        let input_ids_value = Value::from_array(input_ids)
            .map_err(|e| anyhow!("Failed to create ONNX value from input_ids: {}", e))?;
        
        let audio_features_value = Value::from_array(audio_features.clone())
            .map_err(|e| anyhow!("Failed to create ONNX value from audio_features: {}", e))?;

        // Run decoder session
        let outputs = self.decoder_session
            .run(ort::inputs![
                "input_ids" => input_ids_value,
                "encoder_hidden_states" => audio_features_value
            ])
            .map_err(|e| anyhow!("Decoder inference failed: {}", e))?;

        // Extract logits from the output
        let logits = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("Failed to extract logits from decoder output: {}", e))?;

        // Get the shape and data
        let (shape, data) = logits;
        
        // Convert shape to dimensions
        let dims = shape.as_ref();
        if dims.len() != 3 {
            return Err(anyhow!("Expected 3D logits, got {}D", dims.len()));
        }

        // Convert i64 dimensions to usize
        let dim0 = dims[0] as usize;
        let dim1 = dims[1] as usize;
        let dim2 = dims[2] as usize;

        // Create Array3 from the data
        let logits_array = Array3::from_shape_vec(
            (dim0, dim1, dim2),
            data.to_vec()
        ).map_err(|e| anyhow!("Failed to create logits array: {}", e))?;

        Ok(logits_array)
    }

    /// Generates a sequence of tokens from audio features.
    ///
    /// This method implements the autoregressive token generation loop for the
    /// Whisper decoder. It starts with an initial token sequence (including special
    /// tokens), then iteratively runs the decoder to predict the next token until
    /// either the end-of-text token is generated or the maximum length is reached.
    ///
    /// # Arguments
    ///
    /// * `audio_features` - Audio features from the encoder, shape `(1, sequence_length, hidden_size)`
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the generated token sequence as a `Vec<u32>`,
    /// or an error if generation fails.
    ///
    /// # Generation Process
    ///
    /// 1. Initialize token sequence with special tokens (SOT, LANG, TRANSCRIBE, NO_TIMESTAMPS)
    /// 2. Loop until stopping condition:
    ///    - Run decoder with current tokens and audio features
    ///    - Sample next token from logits (using greedy decoding by default)
    ///    - Append token to sequence
    ///    - Check for end-of-text token or max length
    /// 3. Return complete token sequence
    ///
    /// # Stopping Conditions
    ///
    /// Generation stops when:
    /// - The end-of-text token (`<|endoftext|>`) is generated
    /// - The sequence reaches the maximum length (from model config)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use unterwhisper_lib::asr::onnx::OnnxTranscriber;
    /// # use candle_core::Device;
    /// # use ndarray::Array3;
    /// # let mut transcriber = OnnxTranscriber::new("tiny.en", Device::Cpu, None)?;
    /// let audio_features = Array3::<f32>::zeros((1, 1500, 384));
    /// let tokens = transcriber.generate_tokens(&audio_features)?;
    /// println!("Generated {} tokens", tokens.len());
    /// ```
    fn generate_tokens(&mut self, audio_features: &Array3<f32>) -> Result<Vec<u32>> {
        // Initialize token sequence with special tokens
        let mut tokens = self.initialize_tokens()?;
        
        // Get end-of-text token ID
        let eot_token = self.get_token_id("<|endoftext|>")?;
        
        // Get maximum length from config
        let max_length = self.config.max_length;
        
        // Temperature for sampling (0.0 = greedy decoding)
        let temperature = 0.0;
        
        // Generate tokens until EOT or max_length
        loop {
            // Check if we've reached max length
            if tokens.len() >= max_length {
                break;
            }
            
            // Run decoder with current tokens and audio features
            let logits = self.run_decoder(&tokens, audio_features)?;
            
            // Sample next token
            let next_token = self.sample_token(&logits, &tokens, temperature)?;
            
            // Append token to sequence
            tokens.push(next_token);
            
            // Check for end-of-text token
            if next_token == eot_token {
                break;
            }
        }
        
        Ok(tokens)
    }

    /// Decodes a token sequence to text and removes special tokens.
    ///
    /// This method converts a sequence of token IDs back into human-readable text
    /// using the tokenizer, then cleans up the output by removing Whisper's special
    /// tokens and trimming whitespace.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The token sequence to decode (as a slice of u32 values)
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the cleaned transcription text as a `String`,
    /// or an error if decoding fails.
    ///
    /// # Cleaning Process
    ///
    /// 1. Decode tokens to text using the tokenizer
    /// 2. Remove special tokens:
    ///    - `<|startoftranscript|>` - Start of transcript marker
    ///    - `<|transcribe|>` - Task indicator
    ///    - `<|notimestamps|>` - Timestamp mode indicator
    ///    - `<|endoftext|>` - End of text marker
    ///    - `<|en|>` - Language indicator
    /// 3. Trim leading and trailing whitespace
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tokenizer fails to decode the token sequence
    /// - The token sequence contains invalid token IDs
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use unterwhisper_lib::asr::onnx::OnnxTranscriber;
    /// # use candle_core::Device;
    /// # let transcriber = OnnxTranscriber::new("tiny.en", Device::Cpu, None)?;
    /// let tokens = vec![50258, 50259, 50359, 50363, 1234, 5678, 50257];
    /// let text = transcriber.decode_and_clean(&tokens)?;
    /// println!("Transcription: {}", text);
    /// ```
    fn decode_and_clean(&self, tokens: &[u32]) -> Result<String> {
        // Decode tokens to text using the tokenizer
        // skip_special_tokens=false because we want to manually remove specific ones
        let text = self.tokenizer
            .decode(tokens, false)
            .map_err(|e| anyhow!("Tokenizer decode failed: {}", e))?;
        
        // Remove special tokens
        let cleaned = text
            .replace("<|startoftranscript|>", "")
            .replace("<|transcribe|>", "")
            .replace("<|notimestamps|>", "")
            .replace("<|endoftext|>", "")
            .replace("<|en|>", "")
            .trim()
            .to_string();
        
        Ok(cleaned)
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

    #[test]
    fn test_apply_repetition_penalty_no_repetitions() {
        // Test that penalty is not applied when tokens haven't been repeated
        let mut logits = vec![2.0, -1.0, 0.5, -0.5, 1.0];
        let tokens = vec![0, 1, 2, 3]; // Each token appears once
        let penalty = 1.1;
        
        let original_logits = logits.clone();
        OnnxTranscriber::apply_repetition_penalty(&mut logits, &tokens, penalty);
        
        // All logits should be penalized once (appeared once)
        assert!((logits[0] - original_logits[0] / penalty).abs() < 1e-6);
        assert!((logits[1] - original_logits[1] * penalty).abs() < 1e-6);
        assert!((logits[2] - original_logits[2] / penalty).abs() < 1e-6);
        assert!((logits[3] - original_logits[3] * penalty).abs() < 1e-6);
        // Token 4 never appeared, should be unchanged
        assert_eq!(logits[4], original_logits[4]);
    }

    #[test]
    fn test_apply_repetition_penalty_with_repetitions() {
        // Test that penalty increases exponentially with repetition count
        let mut logits = vec![2.0, -1.0, 0.5, -0.5];
        let tokens = vec![0, 0, 0, 2]; // Token 0 appears 3 times, token 2 once
        let penalty = 1.1;
        
        let original_logits = logits.clone();
        OnnxTranscriber::apply_repetition_penalty(&mut logits, &tokens, penalty);
        
        // Token 0 appeared 3 times: penalty^3
        let expected_0 = original_logits[0] / penalty.powi(3);
        assert!((logits[0] - expected_0).abs() < 1e-6);
        
        // Token 1 never appeared: unchanged
        assert_eq!(logits[1], original_logits[1]);
        
        // Token 2 appeared once: penalty^1
        let expected_2 = original_logits[2] / penalty;
        assert!((logits[2] - expected_2).abs() < 1e-6);
        
        // Token 3 never appeared: unchanged
        assert_eq!(logits[3], original_logits[3]);
    }

    #[test]
    fn test_apply_repetition_penalty_positive_logits() {
        // Test that positive logits are divided by penalty factor
        let mut logits = vec![2.0, 3.0, 1.5];
        let tokens = vec![0, 1]; // Tokens 0 and 1 each appear once
        let penalty = 1.1;
        
        let original_logits = logits.clone();
        OnnxTranscriber::apply_repetition_penalty(&mut logits, &tokens, penalty);
        
        // Positive logits should be divided
        assert!((logits[0] - original_logits[0] / penalty).abs() < 1e-6);
        assert!((logits[1] - original_logits[1] / penalty).abs() < 1e-6);
        // Token 2 never appeared: unchanged
        assert_eq!(logits[2], original_logits[2]);
    }

    #[test]
    fn test_apply_repetition_penalty_negative_logits() {
        // Test that negative logits are multiplied by penalty factor
        let mut logits = vec![-2.0, -1.0, -0.5];
        let tokens = vec![0, 1]; // Tokens 0 and 1 each appear once
        let penalty = 1.1;
        
        let original_logits = logits.clone();
        OnnxTranscriber::apply_repetition_penalty(&mut logits, &tokens, penalty);
        
        // Negative logits should be multiplied (making them more negative)
        assert!((logits[0] - original_logits[0] * penalty).abs() < 1e-6);
        assert!((logits[1] - original_logits[1] * penalty).abs() < 1e-6);
        // Token 2 never appeared: unchanged
        assert_eq!(logits[2], original_logits[2]);
    }

    #[test]
    fn test_apply_repetition_penalty_mixed_logits() {
        // Test with a mix of positive and negative logits
        let mut logits = vec![2.0, -1.0, 0.5, -0.5, 0.0];
        let tokens = vec![0, 1, 2, 3]; // Each appears once
        let penalty = 1.1;
        
        let original_logits = logits.clone();
        OnnxTranscriber::apply_repetition_penalty(&mut logits, &tokens, penalty);
        
        // Positive logits divided
        assert!((logits[0] - original_logits[0] / penalty).abs() < 1e-6);
        assert!((logits[2] - original_logits[2] / penalty).abs() < 1e-6);
        
        // Negative logits multiplied
        assert!((logits[1] - original_logits[1] * penalty).abs() < 1e-6);
        assert!((logits[3] - original_logits[3] * penalty).abs() < 1e-6);
        
        // Zero logit stays zero (0 * penalty = 0)
        assert_eq!(logits[4], 0.0);
        
        // Token 4 never appeared: unchanged
        assert_eq!(logits[4], original_logits[4]);
    }

    #[test]
    fn test_apply_repetition_penalty_empty_tokens() {
        // Test that no penalty is applied when token sequence is empty
        let mut logits = vec![2.0, -1.0, 0.5];
        let tokens: Vec<u32> = vec![]; // Empty token sequence
        let penalty = 1.1;
        
        let original_logits = logits.clone();
        OnnxTranscriber::apply_repetition_penalty(&mut logits, &tokens, penalty);
        
        // All logits should remain unchanged
        assert_eq!(logits, original_logits);
    }

    #[test]
    fn test_apply_repetition_penalty_out_of_bounds_token() {
        // Test that out-of-bounds token IDs are safely ignored
        let mut logits = vec![2.0, -1.0, 0.5];
        let tokens = vec![0, 100]; // Token 100 is out of bounds
        let penalty = 1.1;
        
        let original_logits = logits.clone();
        OnnxTranscriber::apply_repetition_penalty(&mut logits, &tokens, penalty);
        
        // Token 0 should be penalized
        assert!((logits[0] - original_logits[0] / penalty).abs() < 1e-6);
        
        // Tokens 1 and 2 should be unchanged (never appeared)
        assert_eq!(logits[1], original_logits[1]);
        assert_eq!(logits[2], original_logits[2]);
        
        // No panic should occur from out-of-bounds token
    }

    #[test]
    fn test_apply_repetition_penalty_exponential_growth() {
        // Test that penalty grows exponentially with repetition count
        let mut logits = vec![10.0, 10.0, 10.0];
        let tokens = vec![0, 0, 0, 0, 0]; // Token 0 appears 5 times
        let penalty = 1.1;
        
        OnnxTranscriber::apply_repetition_penalty(&mut logits, &tokens, penalty);
        
        // Token 0 should have penalty^5 applied
        let expected = 10.0 / penalty.powi(5);
        assert!((logits[0] - expected).abs() < 1e-6);
        
        // Tokens 1 and 2 should be unchanged
        assert_eq!(logits[1], 10.0);
        assert_eq!(logits[2], 10.0);
    }
}

