# KV Cache Implementation Guide for ONNX Whisper with ORT

## Executive Summary

This document provides a complete guide to implementing KV (Key-Value) cache for Whisper ONNX models using the `ort` (ONNX Runtime) crate in Rust. KV caching can provide **3-5x speedup** in decoder inference by avoiding redundant computation of attention keys and values.

## Table of Contents

1. [Understanding KV Cache](#understanding-kv-cache)
2. [Whisper Decoder Variants](#whisper-decoder-variants)
3. [ORT Implementation Details](#ort-implementation-details)
4. [Code Changes Required](#code-changes-required)
5. [Performance Analysis](#performance-analysis)
6. [Troubleshooting](#troubleshooting)

---

## Understanding KV Cache

### The Problem

In autoregressive generation, the decoder processes tokens sequentially:

```
Iteration 1: Process tokens [SOT, LANG, TASK, NOTIMESTAMPS]
Iteration 2: Process tokens [SOT, LANG, TASK, NOTIMESTAMPS, token_1]
Iteration 3: Process tokens [SOT, LANG, TASK, NOTIMESTAMPS, token_1, token_2]
...
Iteration N: Process tokens [SOT, ..., token_N-1]
```

**Current behavior** (from your logs):
- Iteration 1: 4 tokens → 22ms
- Iteration 2: 5 tokens → 22ms
- Iteration 10: 13 tokens → 24ms
- Iteration 20: 23 tokens → 28ms

Each iteration recomputes attention for ALL previous tokens, even though their key/value representations don't change.

### The Solution

KV cache stores the computed attention keys and values from previous iterations:

```
Iteration 1: Process tokens [SOT, LANG, TASK, NOTIMESTAMPS]
            → Store KV cache for these 4 tokens
            
Iteration 2: Process ONLY token_1 (using cached KV for previous 4)
            → Update KV cache with token_1's KV
            
Iteration 3: Process ONLY token_2 (using cached KV for previous 5)
            → Update KV cache with token_2's KV
```

**Expected behavior with KV cache**:
- Iteration 1: 4 tokens → 22ms (same as before)
- Iteration 2: 1 token → 5ms (4-5x faster)
- Iteration 10: 1 token → 5ms (consistent)
- Iteration 20: 1 token → 5ms (consistent)

---

## Whisper Decoder Variants

### Standard Decoder (`decoder_model.onnx`)

**Inputs:**
```
input_ids: int64[batch_size, sequence_length]
encoder_hidden_states: float32[batch_size, 1500, hidden_size]
```

**Outputs:**
```
logits: float32[batch_size, sequence_length, vocab_size]
```

**Limitation:** No KV cache support - must reprocess entire sequence each time.

### Merged Decoder (`decoder_model_merged.onnx`)

The merged decoder has two execution paths controlled by `use_cache_branch`:

#### First Iteration (Prefill Phase)

**Inputs:**
```
input_ids: int64[batch_size, sequence_length]  # All initial tokens
encoder_hidden_states: float32[batch_size, 1500, hidden_size]
use_cache_branch: bool[1] = false
```

**Outputs:**
```
logits: float32[batch_size, sequence_length, vocab_size]

# KV cache outputs (for each decoder layer)
present.0.decoder.key: float32[batch_size, num_heads, sequence_length, head_dim]
present.0.decoder.value: float32[batch_size, num_heads, sequence_length, head_dim]
present.1.decoder.key: float32[batch_size, num_heads, sequence_length, head_dim]
present.1.decoder.value: float32[batch_size, num_heads, sequence_length, head_dim]
...
present.{N-1}.decoder.key: ...
present.{N-1}.decoder.value: ...

# Also cross-attention cache (encoder-decoder attention)
present.0.encoder_decoder.key: float32[batch_size, num_heads, 1500, head_dim]
present.0.encoder_decoder.value: float32[batch_size, num_heads, 1500, head_dim]
...
```

#### Subsequent Iterations (Generation Phase)

**Inputs:**
```
input_ids: int64[batch_size, 1]  # Only the last generated token
encoder_hidden_states: float32[batch_size, 1500, hidden_size]  # Reused
use_cache_branch: bool[1] = true

# KV cache inputs from previous iteration
past_key_values.0.decoder.key: float32[batch_size, num_heads, past_length, head_dim]
past_key_values.0.decoder.value: float32[batch_size, num_heads, past_length, head_dim]
past_key_values.1.decoder.key: ...
past_key_values.1.decoder.value: ...
...
past_key_values.0.encoder_decoder.key: float32[batch_size, num_heads, 1500, head_dim]
past_key_values.0.encoder_decoder.value: float32[batch_size, num_heads, 1500, head_dim]
...
```

**Outputs:**
```
logits: float32[batch_size, 1, vocab_size]  # Only for the new token

# Updated KV cache (now includes the new token)
present.0.decoder.key: float32[batch_size, num_heads, past_length+1, head_dim]
present.0.decoder.value: float32[batch_size, num_heads, past_length+1, head_dim]
...
```

### Model Configuration

For Whisper models, typical configurations:

| Model | Layers | Heads | Head Dim | Hidden Size |
|-------|--------|-------|----------|-------------|
| tiny | 4 | 6 | 64 | 384 |
| base | 6 | 8 | 64 | 512 |
| small | 12 | 12 | 64 | 768 |
| medium | 24 | 16 | 64 | 1024 |
| large | 32 | 20 | 64 | 1280 |

**Number of KV cache tensors** = `num_layers * 2 * 2`
- `* 2` for key and value
- `* 2` for decoder self-attention and encoder-decoder cross-attention

Example for tiny model: `4 layers * 2 * 2 = 16 cache tensors`

---

## ORT Implementation Details

### Working with ORT Values

The `ort` crate provides the `Value` type for ONNX tensors. Here's how to work with them:

#### Creating Input Values

```rust
use ort::value::Value;
use ndarray::{Array1, Array2, Array3};

// Boolean scalar
let use_cache = Array1::from_vec(vec![true]);
let use_cache_value = Value::from_array(use_cache)?;

// Token IDs (2D)
let input_ids = Array2::from_shape_vec((1, 1), vec![12345_i64])?;
let input_ids_value = Value::from_array(input_ids)?;

// Audio features (3D) - reused across iterations
let audio_features = Array3::from_shape_vec((1, 1500, 1024), features_vec)?;
let audio_features_value = Value::from_array(audio_features)?;
```

#### Extracting Output Values

```rust
// Run inference
let outputs = session.run(ort::inputs![
    "input_ids" => input_ids_value,
    "encoder_hidden_states" => audio_features_value,
    "use_cache_branch" => use_cache_value
])?;

// Extract logits
let logits = outputs["logits"]
    .try_extract_tensor::<f32>()?;
let (shape, data) = logits;

// Extract KV cache for next iteration
let present_key_0 = outputs["present.0.decoder.key"]
    .try_extract_tensor::<f32>()?;
```

#### Reusing Cache Values

The tricky part: you need to pass the cache tensors back as inputs in the next iteration.

**Option 1: Clone the Value** (simpler but less efficient)
```rust
// Extract and store as owned Value
let cache_values: Vec<Value> = (0..num_layers)
    .flat_map(|i| {
        vec![
            outputs[&format!("present.{}.decoder.key", i)].clone(),
            outputs[&format!("present.{}.decoder.value", i)].clone(),
            outputs[&format!("present.{}.encoder_decoder.key", i)].clone(),
            outputs[&format!("present.{}.encoder_decoder.value", i)].clone(),
        ]
    })
    .collect();
```

**Option 2: Extract and Recreate** (more control)
```rust
// Extract tensor data
let (shape, data) = outputs["present.0.decoder.key"]
    .try_extract_tensor::<f32>()?;

// Store as ndarray
let cache_array = Array4::from_shape_vec(
    (shape[0] as usize, shape[1] as usize, shape[2] as usize, shape[3] as usize),
    data.to_vec()
)?;

// Later: recreate Value for next iteration
let cache_value = Value::from_array(cache_array)?;
```

### Dynamic Input Names

For KV cache, input names change based on layer index:

```rust
use ort::inputs;

// Build inputs dynamically
let mut input_map = ort::SessionInputs::new();
input_map.insert("input_ids", input_ids_value);
input_map.insert("encoder_hidden_states", audio_features_value);
input_map.insert("use_cache_branch", use_cache_value);

// Add cache inputs
for i in 0..num_layers {
    input_map.insert(
        &format!("past_key_values.{}.decoder.key", i),
        past_cache[i * 4].clone()
    );
    input_map.insert(
        &format!("past_key_values.{}.decoder.value", i),
        past_cache[i * 4 + 1].clone()
    );
    input_map.insert(
        &format!("past_key_values.{}.encoder_decoder.key", i),
        past_cache[i * 4 + 2].clone()
    );
    input_map.insert(
        &format!("past_key_values.{}.encoder_decoder.value", i),
        past_cache[i * 4 + 3].clone()
    );
}

let outputs = session.run(input_map)?;
```

---

## Code Changes Required

### 1. Add KV Cache State to `OnnxTranscriber`

```rust
pub struct OnnxTranscriber {
    encoder_session: Session,
    decoder_session: Session,
    tokenizer: Tokenizer,
    config: WhisperConfig,
    use_kv_cache: bool,  // NEW: flag to enable KV cache
}
```

### 2. Update `run_decoder` Signature

```rust
/// Runs the decoder with optional KV cache support
///
/// # Arguments
/// * `input_tokens` - Token IDs to process (all tokens on first call, last token on subsequent)
/// * `audio_features` - Encoder output (reused across all iterations)
/// * `past_kv_cache` - Optional KV cache from previous iteration
/// * `is_first_iteration` - Whether this is the prefill phase
///
/// # Returns
/// * Logits for next token prediction
/// * Updated KV cache for next iteration (if using cache)
fn run_decoder(
    &mut self,
    input_tokens: &[u32],
    audio_features: &Array3<f32>,
    past_kv_cache: Option<&[Value]>,
    is_first_iteration: bool,
) -> Result<(Array3<f32>, Option<Vec<Value>>)>
```

### 3. Implement Cache-Aware Decoder Logic

```rust
fn run_decoder(
    &mut self,
    input_tokens: &[u32],
    audio_features: &Array3<f32>,
    past_kv_cache: Option<&[Value]>,
    is_first_iteration: bool,
) -> Result<(Array3<f32>, Option<Vec<Value>>)> {
    use ndarray::Array2;
    use ort::value::Value;

    // Determine which tokens to process
    let tokens_to_process = if self.use_kv_cache && !is_first_iteration {
        // Only process the last token
        &input_tokens[input_tokens.len() - 1..]
    } else {
        // Process all tokens (first iteration or no cache)
        input_tokens
    };

    info!("Processing {} tokens (first_iter: {}, use_cache: {})", 
          tokens_to_process.len(), is_first_iteration, self.use_kv_cache);

    // Convert tokens to i64 array
    let tokens_i64: Vec<i64> = tokens_to_process.iter().map(|&t| t as i64).collect();
    let seq_len = tokens_i64.len();
    
    let input_ids = Array2::from_shape_vec((1, seq_len), tokens_i64)?;
    let input_ids_value = Value::from_array(input_ids)?;
    let audio_features_value = Value::from_array(audio_features.clone())?;

    // Build inputs based on cache availability
    let outputs = if self.use_kv_cache {
        // Create use_cache_branch input
        let use_cache = !is_first_iteration;
        let use_cache_array = ndarray::Array1::from_vec(vec![use_cache]);
        let use_cache_value = Value::from_array(use_cache_array)?;

        if is_first_iteration {
            // First iteration: no past cache
            info!("Running decoder (prefill phase)");
            self.decoder_session.run(ort::inputs![
                "input_ids" => input_ids_value,
                "encoder_hidden_states" => audio_features_value,
                "use_cache_branch" => use_cache_value
            ])?
        } else {
            // Subsequent iterations: use past cache
            info!("Running decoder (generation phase with cache)");
            
            let past_cache = past_kv_cache.expect("Cache should be present");
            let num_layers = self.config.num_decoder_layers;
            
            // Build dynamic inputs with cache
            let mut inputs = ort::SessionInputs::new();
            inputs.insert("input_ids", input_ids_value);
            inputs.insert("encoder_hidden_states", audio_features_value);
            inputs.insert("use_cache_branch", use_cache_value);
            
            // Add past KV cache inputs
            for layer_idx in 0..num_layers {
                let base_idx = layer_idx * 4;
                inputs.insert(
                    format!("past_key_values.{}.decoder.key", layer_idx),
                    past_cache[base_idx].clone()
                );
                inputs.insert(
                    format!("past_key_values.{}.decoder.value", layer_idx),
                    past_cache[base_idx + 1].clone()
                );
                inputs.insert(
                    format!("past_key_values.{}.encoder_decoder.key", layer_idx),
                    past_cache[base_idx + 2].clone()
                );
                inputs.insert(
                    format!("past_key_values.{}.encoder_decoder.value", layer_idx),
                    past_cache[base_idx + 3].clone()
                );
            }
            
            self.decoder_session.run(inputs)?
        }
    } else {
        // Standard decoder without cache
        info!("Running decoder (standard, no cache)");
        self.decoder_session.run(ort::inputs![
            "input_ids" => input_ids_value,
            "encoder_hidden_states" => audio_features_value
        ])?
    };

    // Extract logits
    let logits = outputs["logits"].try_extract_tensor::<f32>()?;
    let (shape, data) = logits;
    let logits_array = Array3::from_shape_vec(
        (shape[0] as usize, shape[1] as usize, shape[2] as usize),
        data.to_vec()
    )?;

    // Extract KV cache if using cache
    let new_cache = if self.use_kv_cache {
        let num_layers = self.config.num_decoder_layers;
        let mut cache_values = Vec::with_capacity(num_layers * 4);
        
        for layer_idx in 0..num_layers {
            // Extract present cache for this layer
            cache_values.push(
                outputs[&format!("present.{}.decoder.key", layer_idx)].clone()
            );
            cache_values.push(
                outputs[&format!("present.{}.decoder.value", layer_idx)].clone()
            );
            cache_values.push(
                outputs[&format!("present.{}.encoder_decoder.key", layer_idx)].clone()
            );
            cache_values.push(
                outputs[&format!("present.{}.encoder_decoder.value", layer_idx)].clone()
            );
        }
        
        info!("Extracted {} cache tensors", cache_values.len());
        Some(cache_values)
    } else {
        None
    };

    Ok((logits_array, new_cache))
}
```

### 4. Update `generate_tokens` to Use Cache

```rust
fn generate_tokens(&mut self, audio_features: &Array3<f32>) -> Result<Vec<u32>> {
    info!("Initializing token sequence...");
    let mut tokens = self.initialize_tokens()?;
    info!("Initial tokens: {:?}", tokens);
    
    let eot_token = self.get_token_id("<|endoftext|>")?;
    let max_length = self.config.max_length;
    let temperature = 0.0;
    
    info!("Starting autoregressive token generation (max_length: {}, use_cache: {})...", 
          max_length, self.use_kv_cache);
    
    let mut iteration = 0;
    let mut kv_cache: Option<Vec<Value>> = None;
    
    loop {
        iteration += 1;
        
        if tokens.len() >= max_length {
            info!("Reached max length ({}) after {} iterations", max_length, iteration);
            break;
        }
        
        let is_first_iteration = iteration == 1;
        
        info!("Iteration {}: Running decoder with {} tokens (first: {})...", 
              iteration, tokens.len(), is_first_iteration);
        
        // Run decoder with cache
        let (logits, new_cache) = self.run_decoder(
            &tokens,
            audio_features,
            kv_cache.as_deref(),
            is_first_iteration
        )?;
        
        // Update cache for next iteration
        kv_cache = new_cache;
        
        // Sample next token
        info!("Iteration {}: Sampling next token...", iteration);
        let next_token = self.sample_token(&logits, &tokens, temperature)?;
        info!("Iteration {}: Sampled token {}", iteration, next_token);
        
        tokens.push(next_token);
        
        if next_token == eot_token {
            info!("End-of-text token generated after {} iterations", iteration);
            break;
        }
    }
    
    info!("Token generation completed: {} total tokens after {} iterations", 
          tokens.len(), iteration);
    Ok(tokens)
}
```

### 5. Update Constructor to Support KV Cache

```rust
impl OnnxTranscriber {
    pub fn new(
        model_name: &str,
        _language: Option<String>,
        use_kv_cache: bool,  // NEW parameter
    ) -> Result<Self> {
        info!("Loading ONNX Whisper model: {} (use_kv_cache: {})", model_name, use_kv_cache);

        let (repo_id, revision, encoder_file, decoder_file) = get_onnx_model_info(model_name);
        
        // If KV cache requested, try to use merged decoder
        let decoder_file = if use_kv_cache {
            // Replace decoder_model.onnx with decoder_model_merged.onnx
            decoder_file.replace("decoder_model.onnx", "decoder_model_merged.onnx")
        } else {
            decoder_file.to_string()
        };
        
        info!("Using decoder file: {}", decoder_file);
        
        // ... rest of initialization ...
        
        Ok(Self {
            encoder_session,
            decoder_session,
            tokenizer,
            config,
            use_kv_cache,
        })
    }
}
```

---

## Performance Analysis

### Expected Improvements

Based on your current logs showing ~25ms per decoder iteration:

**Without KV Cache (Current):**
```
Iteration 1:  4 tokens → 22ms
Iteration 5:  8 tokens → 22ms
Iteration 10: 13 tokens → 24ms
Iteration 20: 23 tokens → 28ms
Total for 24 iterations: ~600ms
```

**With KV Cache (Expected):**
```
Iteration 1:  4 tokens → 22ms (prefill)
Iteration 2:  1 token  → 5ms
Iteration 3:  1 token  → 5ms
...
Iteration 24: 1 token  → 5ms
Total for 24 iterations: ~137ms (4.4x speedup)
```

### Memory Usage

KV cache memory per layer:
```
cache_size = batch_size * num_heads * sequence_length * head_dim * 4 bytes * 4 tensors

For tiny model at sequence length 24:
= 1 * 6 * 24 * 64 * 4 * 4
= 147,456 bytes per layer
= 589 KB for 4 layers

For medium model at sequence length 24:
= 1 * 16 * 24 * 64 * 4 * 4
= 393,216 bytes per layer
= 9.4 MB for 24 layers
```

This is negligible compared to model size (tiny: ~40MB, medium: ~1.5GB).

### Bottleneck Analysis

With KV cache, your bottleneck shifts:

**Current bottleneck:** Decoder iterations (~600ms)
**After KV cache:** Encoder inference (~1400ms)

To further optimize after KV cache:
1. Use quantized encoder (int8)
2. Reduce mel spectrogram padding
3. Use distil-whisper models (smaller encoder)

---

## Troubleshooting

### Issue: "Input not found: past_key_values.0.decoder.key"

**Cause:** Using standard decoder instead of merged decoder.

**Solution:** Ensure you're downloading `decoder_model_merged.onnx`:
```rust
let decoder_file = if use_kv_cache {
    decoder_file.replace("decoder_model.onnx", "decoder_model_merged.onnx")
} else {
    decoder_file.to_string()
};
```

### Issue: "Shape mismatch in past_key_values"

**Cause:** Cache from previous iteration has wrong sequence length.

**Solution:** Verify cache is being updated correctly:
```rust
// Cache should grow by 1 each iteration
// Iteration 1: [batch, heads, 4, head_dim]
// Iteration 2: [batch, heads, 5, head_dim]
// Iteration 3: [batch, heads, 6, head_dim]
```

### Issue: "Different outputs with/without cache"

**Cause:** Likely a bug in cache management or token processing.

**Solution:** Add validation:
```rust
// Test: run same input with and without cache, compare outputs
let logits_no_cache = run_without_cache(&tokens)?;
let logits_with_cache = run_with_cache(&tokens)?;
assert_approx_eq!(logits_no_cache, logits_with_cache, epsilon=1e-4);
```

### Issue: "OOM (Out of Memory)"

**Cause:** Cache growing too large or not being released.

**Solution:** 
- Implement max sequence length limit
- Clear cache between transcriptions
- Use smaller models

### Issue: "Slower with cache than without"

**Cause:** Overhead of cache management exceeds savings (only happens with very short sequences).

**Solution:** Only enable cache for sequences > 10 tokens:
```rust
let use_cache_for_this_run = self.use_kv_cache && tokens.len() > 10;
```

---

## Testing Strategy

### 1. Unit Tests

```rust
#[test]
fn test_kv_cache_correctness() {
    // Generate tokens with and without cache
    // Verify identical outputs
}

#[test]
fn test_kv_cache_performance() {
    // Measure time with and without cache
    // Verify speedup > 2x
}
```

### 2. Integration Tests

```rust
#[test]
fn test_full_transcription_with_cache() {
    let transcriber = OnnxTranscriber::new("tiny.en", None, true)?;
    let audio = load_test_audio();
    let text = transcriber.transcribe_from_mel(&audio)?;
    assert_eq!(text, "expected transcription");
}
```

### 3. Benchmark

```rust
use std::time::Instant;

let start = Instant::now();
let text = transcriber.transcribe_from_mel(&mel)?;
let duration = start.elapsed();
println!("Transcription took: {:?}", duration);
```

---

## Summary of Changes

### Files to Modify

1. **`src-tauri/src/asr/onnx.rs`**
   - Add `use_kv_cache: bool` field to `OnnxTranscriber`
   - Update `new()` to accept `use_kv_cache` parameter
   - Modify `run_decoder()` signature and implementation
   - Update `generate_tokens()` to manage cache state

### Key Implementation Points

1. **Use merged decoder** when KV cache is enabled
2. **Track iteration state** (first vs subsequent)
3. **Manage cache lifecycle** (extract, store, pass back)
4. **Handle dynamic input names** for cache tensors
5. **Process only last token** after first iteration

### Expected Results

- **4-5x speedup** in decoder phase
- **~2x overall speedup** (encoder still dominates)
- **Identical transcription quality**
- **Minimal memory overhead** (<10MB for most models)

---

## Next Steps

1. Implement the code changes outlined above
2. Test with tiny.en model first (fewer layers = simpler debugging)
3. Verify correctness by comparing outputs with/without cache
4. Measure performance improvements
5. Extend to other model sizes
6. Consider combining with quantized models for further speedup

