# Design Document: ONNX Transcriber

## Overview

The ONNX Transcriber is an alternative implementation of the Whisper transcription interface using ONNX Runtime instead of Candle. This design provides a drop-in replacement for the existing `WhisperTransformer` that leverages ONNX's cross-platform optimizations and hardware acceleration capabilities.

The implementation will mirror the existing `WhisperTransformer` interface while using ONNX Runtime's Rust bindings (`ort` crate) for model inference. This allows the `RealtimeTranscriber` to work with either implementation without modification.

## Architecture

### Component Structure

```
OnnxTranscriber
├── Model Loading (HuggingFace Hub)
├── ONNX Runtime Session Management
│   ├── Encoder Session
│   └── Decoder Session
├── Tokenizer
├── Inference Pipeline
│   ├── Encoder Forward Pass
│   ├── Decoder Autoregressive Generation
│   └── Token Sampling/Decoding
└── Text Post-processing
```

### Integration with Existing System

The OnnxTranscriber integrates into the existing transcription pipeline:

```
RealtimeTranscriber
    ↓
Arc<Mutex<dyn Transcriber>>  ← Generic interface
    ↓
    ├── WhisperTransformer (Candle-based)
    └── OnnxTranscriber (ONNX-based)  ← New implementation
```

### Key Design Decisions

1. **Interface Compatibility**: Maintain exact same public API as `WhisperTransformer` to enable seamless swapping
2. **ONNX Model Format**: Use pre-converted ONNX models from HuggingFace (e.g., from optimum-whisper repositories)
3. **Session Management**: Create separate ONNX sessions for encoder and decoder to match Whisper's architecture
4. **Thread Safety**: Ensure thread-safe access through Rust's type system and proper synchronization
5. **Device Abstraction**: Map Candle's `Device` enum to ONNX Runtime execution providers

## Components and Interfaces

### OnnxTranscriber Struct

```rust
pub struct OnnxTranscriber {
    encoder_session: ort::Session,
    decoder_session: ort::Session,
    tokenizer: Tokenizer,
    config: WhisperConfig,
    device: Device,
}
```

**Fields:**
- `encoder_session`: ONNX Runtime session for the encoder model
- `decoder_session`: ONNX Runtime session for the decoder model  
- `tokenizer`: Tokenizers library instance for encoding/decoding text
- `config`: Whisper model configuration (num_mel_bins, vocab_size, etc.)
- `device`: Device specification (CPU, Metal, CUDA) for execution provider selection

### Public API

```rust
impl OnnxTranscriber {
    /// Create a new ONNX transcriber
    pub fn new(
        model_name: &str,
        device: Device,
        language: Option<String>
    ) -> Result<Self>;
    
    /// Get model configuration
    pub fn config(&self) -> &WhisperConfig;
    
    /// Transcribe from mel spectrogram
    pub fn transcribe_from_mel(&mut self, mel_spectrogram: &[f32]) -> Result<String>;
}
```

### WhisperConfig Struct

```rust
pub struct WhisperConfig {
    pub num_mel_bins: usize,
    pub vocab_size: usize,
    pub max_length: usize,
    pub num_encoder_layers: usize,
    pub num_decoder_layers: usize,
}
```

This configuration struct will be shared between `WhisperTransformer` and `OnnxTranscriber` to maintain compatibility.

## Data Models

### Model File Structure

ONNX models will be organized as:
```
model_repository/
├── config.json          # Whisper configuration
├── tokenizer.json       # Tokenizer vocabulary and config
├── encoder_model.onnx   # Encoder ONNX model
└── decoder_model.onnx   # Decoder ONNX model (with past key-values)
```

### Tensor Shapes

**Encoder Input:**
- `mel_spectrogram`: `[batch_size, n_mels, n_frames]` = `[1, 80 or 128, 3000]`

**Encoder Output:**
- `audio_features`: `[batch_size, sequence_length, hidden_size]` = `[1, 1500, 512/768/1024]`

**Decoder Input:**
- `input_ids`: `[batch_size, sequence_length]` = `[1, current_length]`
- `encoder_hidden_states`: `[batch_size, 1500, hidden_size]`
- `past_key_values` (optional): Cached attention states for faster generation

**Decoder Output:**
- `logits`: `[batch_size, sequence_length, vocab_size]`
- `past_key_values`: Updated cache for next iteration

### Token Generation Flow

```
1. Initialize tokens: [SOT, LANG, TRANSCRIBE, NO_TIMESTAMPS]
2. Loop until EOT or max_length:
   a. Run decoder with current tokens + audio features
   b. Extract logits for last position
   c. Apply repetition penalty
   d. Apply temperature (if > 0)
   e. Sample or select greedy token
   f. Append token to sequence
   g. Check for EOT
3. Decode token sequence to text
4. Clean special tokens
5. Return transcription
```

## Implementation Details

### Model Loading

The model loading process will:

1. **Resolve Model Path**: Map model name (e.g., "tiny.en") to HuggingFace repository
2. **Download Files**: Use `hf-hub` crate to download ONNX models, config, and tokenizer
3. **Parse Configuration**: Load `config.json` to extract model parameters
4. **Initialize Tokenizer**: Load `tokenizer.json` using the `tokenizers` crate
5. **Create ONNX Sessions**: Initialize encoder and decoder sessions with appropriate execution providers

**Model Repository Mapping:**
```rust
fn get_onnx_model_info(model_name: &str) -> (&'static str, &'static str, &'static str) {
    match model_name {
        "tiny" => ("openai/whisper-tiny", "main", "encoder_model.onnx"),
        "tiny.en" => ("openai/whisper-tiny.en", "main", "encoder_model.onnx"),
        "base" => ("openai/whisper-base", "main", "encoder_model.onnx"),
        "large-v3-turbo-onnx" => ("onnx-community/whisper-large-v3-turbo", "main", "encoder_model.onnx"),
        // ... other models
        _ => ("onnx-community/whisper-large-v3-turbo", "main", "encoder_model.onnx"),
    }
}
```

The function returns:
1. Repository ID (e.g., "openai/whisper-tiny")
2. Revision/branch (e.g., "main")
3. Model filename prefix (e.g., "encoder_model.onnx" - decoder will be "decoder_model.onnx")

Note: ONNX models may need to be sourced from optimized repositories like `onnx-community` or `optimum-whisper`, or converted using `optimum-cli`.

### Execution Provider Selection

Map Candle's `Device` to ONNX Runtime execution providers:

```rust
fn get_execution_provider(device: &Device) -> ExecutionProvider {
    match device {
        Device::Cpu => ExecutionProvider::CPU,
        Device::Cuda(_) => ExecutionProvider::CUDA,
        Device::Metal(_) => ExecutionProvider::CoreML, // or DirectML on Windows
    }
}
```

### Mel Spectrogram Processing

The mel spectrogram input processing will:

1. **Validate Input**: Check that input is not empty
2. **Pad or Truncate**: Ensure exactly 3000 time frames
   - If too short: pad with zeros
   - If too long: truncate to 3000 frames
3. **Reshape**: Convert flat array to `[1, n_mels, 3000]` tensor
4. **Create ONNX Tensor**: Wrap in `ort::Value` for inference

### Encoder Inference

```rust
fn run_encoder(&self, mel: &[f32]) -> Result<Array> {
    // Create input tensor
    let mel_tensor = Array::from_shape_vec(
        (1, self.config.num_mel_bins, 3000),
        mel.to_vec()
    )?;
    
    // Run encoder session
    let outputs = self.encoder_session.run(
        ort::inputs!["mel_spectrogram" => mel_tensor]?
    )?;
    
    // Extract audio features
    let audio_features = outputs["last_hidden_state"].extract_tensor()?;
    Ok(audio_features)
}
```

### Decoder Inference with Autoregressive Generation

The decoder runs iteratively to generate tokens one at a time:

```rust
fn generate_tokens(
    &mut self,
    audio_features: &Array,
    max_length: usize
) -> Result<Vec<u32>> {
    let mut tokens = self.initialize_tokens()?;
    let mut past_key_values = None;
    
    for _ in 0..max_length {
        // Prepare decoder inputs
        let input_ids = Array::from_vec(tokens.clone());
        let mut inputs = ort::inputs![
            "input_ids" => input_ids,
            "encoder_hidden_states" => audio_features
        ]?;
        
        // Add cached key-values if available
        if let Some(cache) = past_key_values {
            inputs.extend(cache);
        }
        
        // Run decoder
        let outputs = self.decoder_session.run(inputs)?;
        
        // Extract logits and cache
        let logits = outputs["logits"].extract_tensor()?;
        past_key_values = Some(extract_cache(&outputs));
        
        // Sample next token
        let next_token = self.sample_token(&logits, &tokens)?;
        
        // Check for end condition
        if next_token == self.eot_token {
            break;
        }
        
        tokens.push(next_token);
    }
    
    Ok(tokens)
}
```

### Repetition Penalty

Apply repetition penalty to discourage repeated tokens:

```rust
fn apply_repetition_penalty(
    logits: &mut [f32],
    tokens: &[u32],
    penalty: f32
) {
    let mut token_counts = HashMap::new();
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
```

### Token Decoding and Cleanup

After generating tokens, decode to text and clean:

```rust
fn decode_and_clean(&self, tokens: &[u32]) -> Result<String> {
    // Decode tokens to text
    let text = self.tokenizer.decode(tokens, true)
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
```

## Error Handling

The implementation will use Rust's `Result<T, E>` type with `anyhow::Error` for error propagation. Key error scenarios:

1. **Model Loading Errors**:
   - Network failures during download
   - Invalid model files
   - Unsupported model format

2. **Inference Errors**:
   - ONNX Runtime execution failures
   - Invalid tensor shapes
   - Out of memory errors

3. **Tokenizer Errors**:
   - Missing special tokens
   - Decoding failures
   - Invalid token IDs

All errors will include descriptive messages with context about the failure point.

## Testing Strategy

### Unit Tests

Unit tests will verify specific functionality:

1. **Model Info Resolution**: Test that model names map to correct repositories
2. **Tensor Shape Handling**: Test padding and truncation of mel spectrograms
3. **Token Initialization**: Test that special tokens are correctly initialized
4. **Repetition Penalty**: Test that penalty is correctly applied to logits
5. **Text Cleaning**: Test that special tokens are removed from output

### Property-Based Tests

Property-based tests will verify universal properties across many inputs using the `proptest` crate (already in dev-dependencies):


## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property 1: Model Name Recognition

*For any* supported Whisper model name (tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large, large-v2, large-v3, large-v3-turbo, distil variants), the model name should map to a valid HuggingFace repository identifier.

**Validates: Requirements 1.4**

### Property 2: Mel Spectrogram Normalization

*For any* mel spectrogram input of arbitrary length, after processing by `transcribe_from_mel`, the internal representation should have exactly 3000 time frames (either padded with zeros if too short, or truncated if too long).

**Validates: Requirements 2.5**

### Property 3: Deterministic Transcription

*For any* mel spectrogram input, when transcribed multiple times with temperature 0.0 (greedy decoding), the output text should be identical across all runs.

**Validates: Requirements 3.3**

### Property 4: Token Sequence Length Bounds

*For any* transcription, the generated token sequence length should be less than or equal to the maximum token length, and should terminate either at the end-of-text token or at the maximum length.

**Validates: Requirements 3.4, 3.5**

### Property 5: Valid Token Decoding

*For any* valid token sequence generated by the decoder, the tokenizer should successfully decode it to a text string without errors.

**Validates: Requirements 4.1**

### Property 6: Special Token Removal

*For any* transcription output, the final text should not contain any of the special token strings: `<|startoftranscript|>`, `<|transcribe|>`, `<|notimestamps|>`, `<|endoftext|>`, or `<|en|>`.

**Validates: Requirements 4.2, 4.3**

### Property 7: Whitespace Trimming

*For any* transcription output, the final text should not have leading or trailing whitespace characters.

**Validates: Requirements 4.4**

### Property 8: Thread Safety

*For any* sequence of concurrent calls to `transcribe_from_mel` from multiple threads (using Arc<Mutex<OnnxTranscriber>>), all calls should complete without panics, data races, or deadlocks, and each should produce valid transcription results.

**Validates: Requirements 8.2, 8.3**

### Example-Based Tests

In addition to properties, the following specific examples should be tested:

1. **Empty Input Handling**: Empty mel spectrogram → empty string output (Requirements 2.4)
2. **Model Initialization**: Initialize with "tiny.en" → successful loading (Requirements 1.1, 1.2, 1.5)
3. **Encoder Execution**: Valid mel input → audio features with correct shape (Requirements 3.1)
4. **Decoder Execution**: Audio features → token sequence generated (Requirements 3.2)
5. **CPU Device Selection**: Initialize with CPU device → CPU execution provider configured (Requirements 7.2)
6. **GPU Device Selection**: Initialize with GPU device → GPU execution provider configured if available (Requirements 7.3)
7. **Execution Provider Logging**: Initialization → log message contains execution provider (Requirements 7.5)

### Edge Cases

The following edge cases should be handled and tested:

1. **Model Loading Failure**: Invalid model name or network error → descriptive error (Requirements 1.3, 6.1)
2. **Inference Failure**: Invalid tensor shapes or ONNX errors → descriptive error (Requirements 6.2)
3. **Tokenizer Failure**: Missing special tokens → descriptive error (Requirements 6.3, 6.4)
4. **GPU Unavailable**: GPU requested but not available → fallback to CPU (Requirements 7.4)

### Testing Configuration

All property-based tests will:
- Use the `proptest` crate (already in dev-dependencies)
- Run a minimum of 100 iterations per property
- Be tagged with comments referencing the design property
- Tag format: `// Feature: onnx-transcriber, Property N: [property title]`

Unit tests will focus on:
- Specific examples demonstrating correct behavior
- Edge cases and error conditions
- Integration points with the existing transcription pipeline

Both unit tests and property-based tests are complementary and necessary for comprehensive coverage. Unit tests catch concrete bugs and verify specific scenarios, while property tests verify general correctness across many inputs.
