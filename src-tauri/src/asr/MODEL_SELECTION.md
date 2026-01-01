# Model Selection Guide

The transcriber supports two backend implementations for Whisper models:

## Backends

### 1. Candle Backend (Default)
- Uses the `candle` framework for ML inference
- Supports both safetensors and GGUF quantized models
- Good for general use cases

### 2. ONNX Backend
- Uses ONNX Runtime for ML inference
- Supports cross-platform optimizations
- Hardware acceleration via CPU, CUDA, and CoreML execution providers
- Better compatibility across different platforms

## Model Name Convention

To select which backend to use, simply append `-onnx` to the model name:

### Candle Backend (Default)
```rust
let config = TranscriberConfig {
    model_name: "tiny.en".to_string(),
    // ... other config
};
```

### ONNX Backend
```rust
let config = TranscriberConfig {
    model_name: "tiny.en-onnx".to_string(),  // Note the -onnx suffix
    // ... other config
};
```

## Supported Models

Both backends support the same base model names:

- `tiny`, `tiny.en`
- `base`, `base.en`
- `small`, `small.en`
- `medium`, `medium.en`
- `large`, `large-v2`, `large-v3`, `large-v3-turbo`
- `distil-small.en`, `distil-medium.en`, `distil-large-v2`, `distil-large-v3`

### Examples

| Model Name | Backend | Description |
|------------|---------|-------------|
| `tiny.en` | Candle | Tiny English-only model with Candle |
| `tiny.en-onnx` | ONNX | Tiny English-only model with ONNX |
| `base` | Candle | Base multilingual model with Candle |
| `base-onnx` | ONNX | Base multilingual model with ONNX |
| `large-v3-turbo` | Candle | Large v3 turbo with Candle |
| `large-v3-turbo-onnx` | ONNX | Large v3 turbo with ONNX |

## Implementation Details

The `UnifiedTransformer` enum automatically detects the backend based on the model name:

```rust
pub enum UnifiedTransformer {
    Candle(WhisperTransformer),
    Onnx(OnnxTranscriber),
}
```

When you create a new transcriber, it will:
1. Check if the model name ends with `-onnx`
2. If yes, strip the suffix and create an `OnnxTranscriber`
3. If no, use the name as-is and create a `WhisperTransformer`

Both implementations provide the same interface:
- `config()` - Get model configuration
- `transcribe_from_mel()` - Transcribe from mel spectrogram

## Performance Considerations

- **Candle**: Generally faster on Apple Silicon (Metal) and NVIDIA GPUs (CUDA)
- **ONNX**: Better cross-platform compatibility, good CPU performance
- **GGUF models**: Only available with Candle backend (e.g., `large-v3-turbo-q41-gguf`)

## Choosing a Backend

Use **Candle** when:
- Running on Apple Silicon with Metal acceleration
- Using GGUF quantized models
- Need maximum performance on supported hardware

Use **ONNX** when:
- Need consistent behavior across platforms
- Running on CPU-only systems
- Want to leverage ONNX Runtime optimizations
- Need CoreML acceleration on macOS
