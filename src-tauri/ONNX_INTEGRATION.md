# ONNX Integration Summary

## Overview

The application now supports two backend implementations for Whisper transcription:
1. **Candle Backend** - Original implementation using the Candle ML framework
2. **ONNX Backend** - New implementation using ONNX Runtime

## How It Works

### Model Selection

The backend is automatically selected based on the model name:
- Model names **without** `-onnx` suffix → Use Candle backend
- Model names **with** `-onnx` suffix → Use ONNX backend

### Example Model Names

| Model Name | Backend | Description |
|------------|---------|-------------|
| `tiny.en` | Candle | Tiny English model with Candle |
| `tiny.en-onnx` | ONNX | Tiny English model with ONNX |
| `base` | Candle | Base multilingual with Candle |
| `base-onnx` | ONNX | Base multilingual with ONNX |
| `large-v3-turbo` | Candle | Large v3 turbo with Candle |
| `large-v3-turbo-onnx` | ONNX | Large v3 turbo with ONNX |

## Implementation Details

### New Files

1. **`src/asr/transformer.rs`** - Unified transformer interface
   - `UnifiedTransformer` enum that wraps both backends
   - Automatic backend selection based on model name
   - Common interface for both implementations

2. **`src/asr/onnx.rs`** - ONNX implementation
   - `OnnxTranscriber` struct for ONNX-based transcription
   - Full encoder-decoder pipeline
   - Hardware acceleration support (CPU, CUDA, CoreML)

3. **`src/asr/config.rs`** - Shared configuration
   - `WhisperConfig` struct used by both backends
   - Common configuration parameters

### Modified Files

1. **`src/asr/whisper.rs`**
   - Updated to return `WhisperConfig` instead of Candle's `Config`
   - Maintains backward compatibility

2. **`src/asr/transcribe.rs`**
   - Uses `UnifiedTransformer` instead of `WhisperTransformer`
   - No changes to public API

3. **`src/asr/mod.rs`**
   - Exports new `transformer` module
   - Exports `UnifiedTransformer`

## User Experience

### For End Users

Users can select the backend by choosing a model name in the settings:
- Select `tiny.en` for Candle backend
- Select `tiny.en-onnx` for ONNX backend

The settings are stored in `config.json` at:
```
~/Library/Application Support/com.unterwhisper.app/config.json
```

### For Developers

The transcriber automatically handles backend selection:

```rust
// This will use Candle backend
let config = TranscriberConfig {
    model_name: "tiny.en".to_string(),
    // ...
};

// This will use ONNX backend
let config = TranscriberConfig {
    model_name: "tiny.en-onnx".to_string(),
    // ...
};

let transcriber = RealtimeTranscriber::new(config, device)?;
// Everything else works the same!
```

## Benefits

### ONNX Backend Advantages
- Better cross-platform compatibility
- Optimized for CPU inference
- Hardware acceleration via execution providers
- Consistent behavior across platforms

### Candle Backend Advantages
- Native Rust implementation
- Excellent Metal (Apple Silicon) support
- GGUF quantized model support
- Generally faster on supported hardware

## Testing

All tests pass for both backends:
```bash
# Test transformer selection logic
cargo test --lib transformer::tests

# Test ONNX implementation
cargo test --lib onnx::tests

# Test Whisper implementation
cargo test --lib whisper::tests
```

## Future Enhancements

Potential improvements:
1. Add UI dropdown to select backend explicitly
2. Benchmark and display performance metrics
3. Auto-select optimal backend based on hardware
4. Support for additional ONNX execution providers
5. Model caching and preloading

## Documentation

See `src/asr/MODEL_SELECTION.md` for detailed model selection guide.
