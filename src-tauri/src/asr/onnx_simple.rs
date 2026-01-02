//! Simplified ONNX Whisper implementation
//! 
//! This is a simplified version that works with both standard and merged decoders
//! but doesn't yet implement full KV caching. It provides a foundation for adding
//! KV cache support later.

use anyhow::{anyhow, Result};
use hf_hub::{api::sync::Api, Repo, RepoType};
use log::{debug, info};
use ndarray::Array3;
use ort::execution_providers::{CoreMLExecutionProvider, ExecutionProviderDispatch};
use ort::session::Session;
use tokenizers::Tokenizer;

use crate::asr::config::WhisperConfig;

/// Decoder variant to use for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderVariant {
    /// Standard decoder without KV caching
    Standard,
    /// Merged decoder with KV caching support (faster when fully implemented)
    Merged,
}

impl DecoderVariant {
    pub fn filename(&self) -> &'static str {
        match self {
            DecoderVariant::Standard => "decoder_model.onnx",
            DecoderVariant::Merged => "decoder_model_merged.onnx",
        }
    }
}

pub fn get_onnx_model_info(model_name: &str) -> (&'static str, &'static str, &'static str) {
    match model_name {
        // Tiny models
        "tiny" => ("Xenova/whisper-tiny", "main", "onnx/encoder_model.onnx"),
        "tiny.en" => ("Xenova/whisper-tiny.en", "main", "onnx/encoder_model.onnx"),
        
        // Base models
        "base" => ("Xenova/whisper-base", "main", "onnx/encoder_model.onnx"),
        "base.en" => ("Xenova/whisper-base.en", "main", "onnx/encoder_model.onnx"),
        
        // Small models
        "small" => ("Xenova/whisper-small", "main", "onnx/encoder_model.onnx"),
        "small.en" => ("Xenova/whisper-small.en", "main", "onnx/encoder_model.onnx"),
        
        // Medium models
        "medium" => ("Xenova/whisper-medium", "main", "onnx/encoder_model.onnx"),
        "medium.en" => ("Xenova/whisper-medium.en", "main", "onnx/encoder_model.onnx"),
        
        // Large models
        "large" => ("Xenova/whisper-large", "main", "onnx/encoder_model.onnx"),
        "large-v2" => ("Xenova/whisper-large-v2", "main", "onnx/encoder_model.onnx"),
        "large-v3" => ("Xenova/whisper-large-v3", "main", "onnx/encoder_model.onnx"),
        "large-v3-turbo" => ("Xenova/whisper-large-v3-turbo", "main", "onnx/encoder_model.onnx"),
        
        // Default fallback
        _ => ("onnx-community/whisper-large-v3-turbo", "main", "encoder_model.onnx"),
    }
}

fn get_execution_provider() -> ExecutionProviderDispatch {
    ExecutionProviderDispatch::from(CoreMLExecutionProvider::default())
}

pub struct OnnxTranscriber {
    encoder_session: Session,
    decoder_session: Session,
    tokenizer: Tokenizer,
    config: WhisperConfig,
    decoder_variant: DecoderVariant,
}

impl OnnxTranscriber {
    pub fn new(
        model_name: &str,
        _language: Option<String>,
        use_kv_cache: bool,
    ) -> Result<Self> {
        info!("Loading ONNX Whisper model: {} (KV cache: {})", model_name, use_kv_cache);

        let decoder_variant = if use_kv_cache {
            DecoderVariant::Merged
        } else {
            DecoderVariant::Standard
        };

        let (repo_id, revision, encoder_file) = get_onnx_model_info(model_name);
        let decoder_file = format!("onnx/{}", decoder_variant.filename());
        info!("Repository: {} (revision: {})", repo_id, revision);

        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            repo_id.to_string(),
            RepoType::Model,
            revision.to_string(),
        ));

        info!("Downloading model files...");
        
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let encoder_path = repo.get(encoder_file)?;
        
        // Try merged decoder first, fall back to standard
        let (decoder_path, actual_variant) = match repo.get(&decoder_file) {
            Ok(path) => {
                info!("Using {} decoder", decoder_variant.filename());
                (path, decoder_variant)
            }
            Err(e) if decoder_variant == DecoderVariant::Merged => {
                info!("Merged decoder not available ({}), using standard decoder", e);
                let fallback = format!("onnx/{}", DecoderVariant::Standard.filename());
                (repo.get(&fallback)?, DecoderVariant::Standard)
            }
            Err(e) => return Err(anyhow!("Failed to download decoder: {}", e)),
        };

        // Load config
        let config_str = std::fs::read_to_string(&config_path)?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
        
        let num_mel_bins = config_json["num_mel_bins"].as_u64().unwrap_or(80) as usize;
        let vocab_size = config_json["vocab_size"].as_u64().unwrap_or(51865) as usize;
        let max_length = config_json["max_target_positions"]
            .as_u64()
            .or_else(|| config_json["max_length"].as_u64())
            .unwrap_or(448) as usize;
        let num_encoder_layers = config_json["encoder_layers"].as_u64().unwrap_or(4) as usize;
        let num_decoder_layers = config_json["decoder_layers"].as_u64().unwrap_or(4) as usize;

        let config = WhisperConfig {
            num_mel_bins,
            vocab_size,
            max_length,
            num_encoder_layers,
            num_decoder_layers,
        };

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)?;

        // Create ONNX sessions
        let encoder_session = Session::builder()?
            .with_execution_providers([get_execution_provider()])?
            .commit_from_file(&encoder_path)?;

        let decoder_session = Session::builder()?
            .with_execution_providers([get_execution_provider()])?
            .commit_from_file(&decoder_path)?;

        info!("ONNX Whisper model loaded successfully");

        Ok(Self {
            encoder_session,
            decoder_session,
            tokenizer,
            config,
            decoder_variant: actual_variant,
        })
    }

    pub fn config(&self) -> &WhisperConfig {
        &self.config
    }

    pub fn transcribe_from_mel(&mut self, mel_spectrogram: &[f32]) -> Result<String> {
        let mel_tensor = match self.preprocess_mel(mel_spectrogram) {
            Some(tensor) => tensor,
            None => return Ok(String::new()),
        };

        let audio_features = self.run_encoder(&mel_tensor)?;
        let tokens = self.generate_tokens(&audio_features)?;
        let text = self.decode_and_clean(&tokens)?;

        Ok(text)
    }

    fn preprocess_mel(&self, mel_spectrogram: &[f32]) -> Option<Array3<f32>> {
        if mel_spectrogram.is_empty() {
            return None;
        }

        let n_mels = self.config.num_mel_bins;
        let target_frames = 3000;
        let target_length = n_mels * target_frames;

        let normalized: Vec<f32> = if mel_spectrogram.len() < target_length {
            let mut padded = mel_spectrogram.to_vec();
            padded.resize(target_length, 0.0);
            padded
        } else if mel_spectrogram.len() > target_length {
            mel_spectrogram[..target_length].to_vec()
        } else {
            mel_spectrogram.to_vec()
        };

        Array3::from_shape_vec((1, n_mels, target_frames), normalized).ok()
    }

    fn run_encoder(&mut self, mel_tensor: &Array3<f32>) -> Result<Array3<f32>> {
        use ort::value::Value;

        let mel_value = Value::from_array(mel_tensor.clone())?;
        let outputs = self.encoder_session.run(ort::inputs!["input_features" => mel_value]?)?;

        let audio_features = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
        let (shape, data) = audio_features;
        
        let dims = shape.as_ref();
        if dims.len() != 3 {
            return Err(anyhow!("Expected 3D audio features, got {}D", dims.len()));
        }

        let audio_features_array = Array3::from_shape_vec(
            (dims[0] as usize, dims[1] as usize, dims[2] as usize),
            data.to_vec()
        )?;

        Ok(audio_features_array)
    }

    fn get_token_id(&self, token: &str) -> Result<u32> {
        self.tokenizer
            .token_to_id(token)
            .ok_or_else(|| anyhow!("Token '{}' not found", token))
    }

    fn initialize_tokens(&self) -> Result<Vec<u32>> {
        let sot_token = self.get_token_id("<|startoftranscript|>")?;
        let language_token = self.get_token_id("<|en|>")?;
        let transcribe_token = self.get_token_id("<|transcribe|>")?;
        let notimestamps_token = self.get_token_id("<|notimestamps|>")?;
        
        Ok(vec![sot_token, language_token, transcribe_token, notimestamps_token])
    }

    fn run_decoder(&mut self, input_tokens: &[u32], audio_features: &Array3<f32>) -> Result<Array3<f32>> {
        use ndarray::Array2;
        use ort::value::Value;

        let tokens_i64: Vec<i64> = input_tokens.iter().map(|&t| t as i64).collect();
        let seq_len = tokens_i64.len();
        
        let input_ids = Array2::from_shape_vec((1, seq_len), tokens_i64)?;
        let input_ids_value = Value::from_array(input_ids)?;
        let audio_features_value = Value::from_array(audio_features.clone())?;

        let outputs = self.decoder_session.run(ort::inputs![
            "input_ids" => input_ids_value,
            "encoder_hidden_states" => audio_features_value
        ]?)?;

        let logits = outputs["logits"].try_extract_tensor::<f32>()?;
        let (shape, data) = logits;
        
        let dims = shape.as_ref();
        if dims.len() != 3 {
            return Err(anyhow!("Expected 3D logits, got {}D", dims.len()));
        }

        let logits_array = Array3::from_shape_vec(
            (dims[0] as usize, dims[1] as usize, dims[2] as usize),
            data.to_vec()
        )?;

        Ok(logits_array)
    }

    fn sample_token(&self, logits: &Array3<f32>, tokens: &[u32]) -> Result<u32> {
        let shape = logits.shape();
        let seq_len = shape[1];
        
        let last_logits = logits.slice(ndarray::s![0, seq_len - 1, ..]);
        let mut last_logits_vec: Vec<f32> = last_logits.to_vec();
        
        // Apply repetition penalty
        Self::apply_repetition_penalty(&mut last_logits_vec, tokens, 1.1);
        
        // Greedy decoding
        let max_idx = last_logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| anyhow!("Failed to find maximum logit"))?;
        
        Ok(max_idx as u32)
    }

    fn apply_repetition_penalty(logits: &mut [f32], tokens: &[u32], penalty: f32) {
        use std::collections::HashMap;

        let mut token_counts: HashMap<u32, i32> = HashMap::new();
        for &token in tokens {
            *token_counts.entry(token).or_insert(0) += 1;
        }

        for (token_id, count) in token_counts {
            let idx = token_id as usize;
            if idx < logits.len() {
                let penalty_factor = penalty.powi(count);
                if logits[idx] > 0.0 {
                    logits[idx] /= penalty_factor;
                } else {
                    logits[idx] *= penalty_factor;
                }
            }
        }
    }

    fn generate_tokens(&mut self, audio_features: &Array3<f32>) -> Result<Vec<u32>> {
        let mut tokens = self.initialize_tokens()?;
        let eot_token = self.get_token_id("<|endoftext|>")?;
        let max_length = self.config.max_length;
        
        loop {
            if tokens.len() >= max_length {
                break;
            }
            
            let logits = self.run_decoder(&tokens, audio_features)?;
            let next_token = self.sample_token(&logits, &tokens)?;
            
            tokens.push(next_token);
            
            if next_token == eot_token {
                break;
            }
        }
        
        Ok(tokens)
    }

    fn decode_and_clean(&self, tokens: &[u32]) -> Result<String> {
        let text = self.tokenizer.decode(tokens, false)?;
        
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
