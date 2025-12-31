use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use log::{debug, info};

pub struct WhisperTransformer {
    model: m::model::Whisper,
    tokenizer: Tokenizer,
    device: Device,
    config: Config,
}

impl WhisperTransformer {
    pub fn new(model_name: &str, device: Device, _language: Option<String>) -> Result<Self> {
        info!("Loading Whisper model: {}", model_name);

        // Download model files from huggingface
        let api = Api::new()?;
        let (model_id, revision) = Self::get_model_info(model_name);
        info!("model_id: {} revision: {}", model_id, revision);
        let repo = api.repo(Repo::with_revision(model_id.to_string(), RepoType::Model, revision.to_string()));
        info!("Got repo {}", repo.url("config.json"));

        let config_filename = repo.get("config.json")?;
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let weights_filename = repo.get("model.safetensors")?;

        // Load config and tokenizer
        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

        // Load model
        let vb = unsafe { 
            VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)? 
        };

        info!("Loading normal Whisper model");
        let model = m::model::Whisper::load(&vb, config.clone())?;

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
        })
    }
    
    pub fn config(&self) -> &Config {
        &self.config
    }
    
    fn get_model_info(model_name: &str) -> (&'static str, &'static str) {
        match model_name {
            "tiny" => ("openai/whisper-tiny", "main"),
            "tiny.en" => ("openai/whisper-tiny.en", "main"),
            "base" => ("openai/whisper-base", "main"),
            "base.en" => ("openai/whisper-base.en", "main"),
            "small" => ("openai/whisper-small", "main"),
            "small.en" => ("openai/whisper-small.en", "main"),
            "medium" => ("openai/whisper-medium", "main"),
            "medium.en" => ("openai/whisper-medium.en", "main"),
            "large" => ("openai/whisper-large", "main"),
            "large-v2" => ("openai/whisper-large-v2", "main"),
            "large-v3" => ("openai/whisper-large-v3", "main"),
            "large-v3-turbo" => ("openai/whisper-large-v3-turbo", "main"),
            "distil-medium.en" => ("distil-whisper/distil-medium.en", "main"),
            "distil-large-v3" => ("distil-whisper/distil-large-v3", "main"),
            "distil-large-v3.5" => ("distil-whisper/distil-large-v3.5", "main"),
            "parakeet-tdt-0.6b-v3" => ("nvidia/parakeet-tdt-0.6b-v3", "main"),
            _ => ("openai/whisper-large-v3-turbo", "main"), // Default fallback
        }
    }

    fn sample_token(&self, logits: &[f32]) -> Result<u32> {
        use rand::Rng;

        // Convert logits to probabilities using softmax
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

        // Sample from the distribution
        let mut rng = rand::thread_rng();
        let random_val: f32 = rng.gen();
        let mut cumulative = 0.0;

        for (i, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if random_val <= cumulative {
                return Ok(i as u32);
            }
        }

        // Fallback to last token
        Ok((probs.len() - 1) as u32)
    }

    /// Transcribe directly from a mel spectrogram (bypassing mel encoding)
    /// 
    /// The mel_spectrogram should be a flat array in row-major order:
    /// [mel_bin_0_frame_0, mel_bin_0_frame_1, ..., mel_bin_0_frame_2999,
    ///  mel_bin_1_frame_0, mel_bin_1_frame_1, ..., mel_bin_1_frame_2999,
    ///  ...]
    /// 
    /// Or in other words: shape (n_mels, 3000) flattened in row-major order
    pub fn transcribe_from_mel(&mut self, mel_spectrogram: &[f32]) -> Result<String> {
        info!("Transcribing from mel spectrogram with {} elements", mel_spectrogram.len());

        if mel_spectrogram.is_empty() {
            return Ok("".to_string());
        }

        // Whisper expects exactly 3000 time steps (30 seconds at 100 fps)
        let expected_time_steps = 3000;
        let n_mels = self.config.num_mel_bins;
        let expected_total_len = n_mels * expected_time_steps;

        // Pad or truncate to exactly 3000 frames
        let mut mel_data = mel_spectrogram.to_vec();
        
        if mel_data.len() < expected_total_len {
            // Pad with zeros if we don't have enough data
            info!("Mel spectrogram too short ({} < {}), padding with zeros", 
                  mel_data.len(), expected_total_len);
            mel_data.resize(expected_total_len, 0.0);
        } else if mel_data.len() > expected_total_len {
            // Truncate if we have too much data
            info!("Mel spectrogram too long ({} > {}), truncating", 
                  mel_data.len(), expected_total_len);
            mel_data.truncate(expected_total_len);
        }

        debug!("Creating tensor with shape (1, {}, {})", n_mels, expected_time_steps);

        // Create tensor with shape (batch=1, n_mels, time_steps)
        let mel = Tensor::from_vec(
            mel_data,
            (1, n_mels, expected_time_steps),
            &self.device,
        )?;

        // Run encoder with flush=true to reset internal state
        debug!("Running encoder forward pass");
        let audio_features = self.model.encoder.forward(&mel, true)?;

        // Enhanced decoding with repetition prevention
        debug!("Decoding audio features to text");
        let sot_token = self.get_token_id(m::SOT_TOKEN)?;
        let transcribe_token = self.get_token_id(m::TRANSCRIBE_TOKEN)?;
        let eot_token = self.get_token_id(m::EOT_TOKEN)?;
        let no_timestamps_token = self.get_token_id(m::NO_TIMESTAMPS_TOKEN)?;

        // Add language token for English models
        let mut tokens = vec![sot_token];
        if self.config.vocab_size > 51864 {
            // For multilingual models, add English language token
            if let Ok(en_token) = self.get_token_id("<|en|>") {
                tokens.push(en_token);
            }
        }
        tokens.extend_from_slice(&[transcribe_token, no_timestamps_token]);
        
        let max_len = 224; // Reasonable max length for transcription
        let temperature = 0.0f32; // Use greedy decoding to reduce randomness
        let repetition_penalty = 1.1f32; // Penalty for repeating tokens
        let mut token_counts = std::collections::HashMap::new();
        
        for i in 0..max_len {
            let tokens_t = Tensor::new(tokens.as_slice(), &self.device)?;
            let tokens_t = tokens_t.unsqueeze(0)?; // Add batch dimension
            
            // Use flush=false after first iteration to maintain decoder state
            let flush = i == 0;
            let ys = self.model.decoder.forward(&tokens_t, &audio_features, flush)?;
            let (_, seq_len, _) = ys.dims3()?;
            let logits = self.model.decoder.final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?.i(0)?;
            
            // Apply repetition penalty
            let mut logits_v: Vec<f32> = logits.to_vec1()?;
            for (token_id, &count) in &token_counts {
                if *token_id < logits_v.len() {
                    let penalty = (repetition_penalty as f32).powi(count as i32);
                    if logits_v[*token_id] > 0.0 {
                        logits_v[*token_id] /= penalty;
                    } else {
                        logits_v[*token_id] *= penalty;
                    }
                }
            }
            
            // Apply temperature (if > 0)
            if temperature > 0.0 {
                for logit in &mut logits_v {
                    *logit /= temperature;
                }
            }
            
            // Get the token with highest probability
            let next_token = if temperature > 0.0 {
                // Sample from distribution
                self.sample_token(&logits_v)?
            } else {
                // Greedy selection
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            
            // Check for end conditions
            if next_token == eot_token {
                debug!("Found EOT token, stopping generation");
                break;
            }
            
            // Update token count for repetition penalty
            *token_counts.entry(next_token as usize).or_insert(0) += 1;
            
            tokens.push(next_token);
            
            // Additional safety check for very long sequences
            if tokens.len() >= max_len {
                debug!("Reached maximum length, stopping generation");
                break;
            }
        }
        
        // Decode tokens to text
        let text = self.tokenizer.decode(&tokens, true).map_err(anyhow::Error::msg)?;
        
        debug!("Decoded text: {}", text);
        debug!("Generated {} tokens", tokens.len());
        
        // Clean up the text by removing special tokens
        let text = text
            .replace("<|startoftranscript|>", "")
            .replace("<|transcribe|>", "")
            .replace("<|notimestamps|>", "")
            .replace("<|endoftext|>", "")
            .replace("<|en|>", "")
            .trim()
            .to_string();
        
        Ok(text)
    }
    
    fn get_token_id(&self, token: &str) -> Result<u32> {
        self.tokenizer
            .token_to_id(token)
            .ok_or_else(|| anyhow::anyhow!("Token '{}' not found", token))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_info() {
        let (model_id, revision) = WhisperTransformer::get_model_info("tiny.en");
        assert_eq!(model_id, "openai/whisper-tiny.en");
        assert_eq!(revision, "main");
    }
}
