# KV Cache Implementation for ONNX Whisper

## Overview

This document describes the KV caching implementation for the ONNX-based Whisper transcriber. The implementation supports both standard and merged decoder variants, with the merged decoder providing a foundation for full KV caching optimization.

## Current Status

### ✅ Implemented
- **Decoder Variant Selection**: Choose between standard (`decoder_model.onnx`) and merged (`decoder_model_merged.onnx`) decoders
- **Automatic Fallback**: If merged decoder is not available, automatically falls back to standard decoder
- **Model Repository Support**: All Xenova Whisper models support both decoder variants
- **API Compatibility**: Simple boolean flag (`use_kv_cache`) to enable/disable KV caching

### 🚧 In Progress
- **Full KV Cache Management**: The merged decoder is loaded but KV cache state management needs to be implemented
- **Performance Optimization**: Once KV cache is fully implemented, expect 2-3x speedup in token generation

## Usage

```rust
use unterwhisper_lib::asr::onnx::OnnxTranscriber;

// With KV caching (faster, recommended)
let transcriber = OnnxTranscriber::new("tiny.en", None, true)?;

// Without KV caching (simpler, slower)
let transcriber = OnnxTranscriber::new("tiny.en", None, false)?;

// Transcribe audio
let mel_spectrogram: Vec<f32> = /* ... */;
let text = transcriber.transcribe_from_mel(&mel_spectrogram)?;
```

## Architecture

### Decoder Variants

#### Standard Decoder (`decoder_model.onnx`)
- **Inputs**: 
  - `input_ids`: Token IDs (batch_size, sequence_length)
  - `encoder_hidden_states`: Audio features from encoder (batch_size, audio_seq_len, hidden_size)
- **Outputs**:
  - `logits`: Next token predictions (batch_size, sequence_length, vocab_size)
- **Behavior**: Processes entire token sequence on each iteration

#### Merged Decoder (`decoder_model_merged.onnx`)
- **Inputs** (first iteration):
  - `input_ids`: Token IDs (batch_size, sequence_length)
  - `encoder_hidden_states`: Audio features
  - `use_cache_branch`: Boolean flag (false for first iteration)
- **Inputs** (subsequent iterations):
  - `input_ids`: Only the last token (batch_size, 1)
  - `encoder_hidden_states`: Audio features (reused)
  - `use_cache_branch`: Boolean flag (true)
  - `past_key_values.{layer}.decoder.key`: Cached keys from previous iteration
  - `past_key_values.{layer}.decoder.value`: Cached values from previous iteration
- **Outputs**:
  - `logits`: Next token predictions
  - `present.{layer}.decoder.key`: Updated key cache for next iteration
  - `present.{layer}.decoder.value`: Updated value cache for next iteration

### KV Cache Benefits

1. **Reduced Computation**: Only process the last token instead of entire sequence
2. **Faster Inference**: 2-3x speedup in token generation phase
3. **Lower Memory Bandwidth**: Reuse cached attention keys/values
4. **Consistent Quality**: Same output as standard decoder

## Implementation Details

### Model Loading

The implementation tries to load the requested decoder variant and falls back gracefully:

```rust
let (decoder_path, actual_variant) = match repo.get(&decoder_file) {
    Ok(path) => {
        info!("Using {} decoder", decoder_variant.filename());
        (path, decoder_variant)
    }
    Err(e) if decoder_variant == DecoderVariant::Merged => {
        info!("Merged decoder not available, using standard decoder");
        let fallback = format!("onnx/{}", DecoderVariant::Standard.filename());
        (repo.get(&fallback)?, DecoderVariant::Standard)
    }
    Err(e) => return Err(anyhow!("Failed to download decoder: {}", e)),
};
```

### Token Generation Loop

Currently, both variants use the same generation loop:

```rust
fn generate_tokens(&mut self, audio_features: &Array3<f32>) -> Result<Vec<u32>> {
    let mut tokens = self.initialize_tokens()?;
    let eot_token = self.get_token_id("<|endoftext|>")?;
    
    loop {
        if tokens.len() >= self.config.max_length {
            break;
        }
        
        // Run decoder with all tokens (standard) or implement KV cache (merged)
        let logits = self.run_decoder(&tokens, audio_features)?;
        let next_token = self.sample_token(&logits, &tokens)?;
        
        tokens.push(next_token);
        
        if next_token == eot_token {
            break;
        }
    }
    
    Ok(tokens)
}
```

## Next Steps for Full KV Cache Implementation

To complete the KV cache implementation, the following changes are needed:

### 1. Update `run_decoder` Signature

```rust
fn run_decoder(
    &mut self,
    input_tokens: &[u32],
    audio_features: &Array3<f32>,
    past_key_values: Option<Vec<ort::value::Value>>,
) -> Result<(Array3<f32>, Option<Vec<ort::value::Value>>)>
```

### 2. Implement Cache Management

```rust
// First iteration: use all tokens, no cache
if past_key_values.is_none() {
    let inputs = ort::inputs![
        "input_ids" => all_tokens_value,
        "encoder_hidden_states" => audio_features_value,
        "use_cache_branch" => false_value
    ];
    let outputs = self.decoder_session.run(inputs)?;
    // Extract present key values for next iteration
}
// Subsequent iterations: use only last token, with cache
else {
    let inputs = ort::inputs![
        "input_ids" => last_token_value,
        "encoder_hidden_states" => audio_features_value,
        "use_cache_branch" => true_value,
        // Add past_key_values inputs
    ];
    let outputs = self.decoder_session.run(inputs)?;
    // Extract updated present key values
}
```

### 3. Update Generation Loop

```rust
let mut past_kv_cache = None;

loop {
    let (logits, new_cache) = self.run_decoder(&tokens, audio_features, past_kv_cache)?;
    past_kv_cache = new_cache;  // Reuse cache for next iteration
    // ... rest of loop
}
```

## Model Availability

All Xenova Whisper models on HuggingFace include both decoder variants:
- `onnx/decoder_model.onnx` (standard)
- `onnx/decoder_model_merged.onnx` (with KV cache support)

Supported models:
- tiny, tiny.en
- base, base.en
- small, small.en
- medium, medium.en
- large, large-v2, large-v3, large-v3-turbo

## Performance Expectations

Once fully implemented, KV caching should provide:
- **2-3x faster** token generation
- **Same transcription quality** as standard decoder
- **Slightly higher memory usage** (storing KV cache)
- **Most beneficial for** longer transcriptions (>10 seconds)

## Testing

To test the current implementation:

```bash
# Build with ONNX support
cargo build --release

# Test with standard decoder
# (set use_kv_cache = false in your code)

# Test with merged decoder
# (set use_kv_cache = true in your code)
```

Both should produce identical transcriptions, with the merged decoder providing a foundation for future KV cache optimization.

## References

- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [Xenova Whisper Models](https://huggingface.co/Xenova)
- [KV Caching Explained](https://medium.com/@plienhar/llm-inference-series-3-kv-caching-unveiled-048152e461c8)
- [Whisper Model Architecture](https://github.com/openai/whisper)
