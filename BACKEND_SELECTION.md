# Backend Selection Guide

## Quick Start

UnterWhisper now supports two AI backends for speech recognition:

### 🔥 Candle Backend (Default)
Fast, native Rust implementation with excellent Apple Silicon support.

### 🎯 ONNX Backend (New!)
Cross-platform optimized runtime with broad hardware support.

## How to Choose

Simply add `-onnx` to any model name to use the ONNX backend:

| Candle (Default) | ONNX Alternative |
|------------------|------------------|
| `tiny.en` | `tiny.en-onnx` |
| `base` | `base-onnx` |
| `small` | `small-onnx` |
| `medium` | `medium-onnx` |
| `large-v3-turbo` | `large-v3-turbo-onnx` |
| `distil-large-v3` | `distil-large-v3-onnx` |

## When to Use Each Backend

### Use Candle (Default) When:
- ✅ Running on Apple Silicon (M1/M2/M3)
- ✅ You want maximum performance
- ✅ Using GGUF quantized models
- ✅ You have NVIDIA GPU with CUDA

### Use ONNX When:
- ✅ Running on older Intel Macs
- ✅ Need consistent cross-platform behavior
- ✅ CPU-only inference
- ✅ Want CoreML acceleration on macOS

## Changing the Backend

1. Open Settings in the app
2. Select a model from the dropdown
3. To use ONNX, choose a model ending in `-onnx`
4. Restart transcription for changes to take effect

## Available Models

All standard Whisper models are supported:

- **Tiny** - Fastest, least accurate
- **Base** - Good balance
- **Small** - Better accuracy
- **Medium** - High accuracy
- **Large** - Best accuracy
- **Distil** - Optimized variants

Add `.en` for English-only models (faster):
- `tiny.en`, `base.en`, `small.en`, etc.

## Performance Tips

1. **For real-time transcription**: Use `tiny.en` or `base.en`
2. **For accuracy**: Use `large-v3-turbo` or `distil-large-v3`
3. **For Apple Silicon**: Candle backend is usually faster
4. **For Intel CPUs**: Try both backends and compare

## Technical Details

The backend selection happens automatically based on the model name suffix. Both backends:
- Use the same models from HuggingFace
- Provide identical transcription quality
- Support the same features
- Have the same API

The only difference is the underlying inference engine.

## Troubleshooting

**Model download fails?**
- Check your internet connection
- Models are downloaded from HuggingFace on first use
- ONNX models come from `onnx-community` repositories

**Slow performance?**
- Try a smaller model (tiny or base)
- Ensure you're using the right backend for your hardware
- Check Activity Monitor for CPU/GPU usage

**Transcription quality issues?**
- Try a larger model
- Both backends should give identical results
- Check microphone input levels

## More Information

For developers and technical details, see:
- `src-tauri/ONNX_INTEGRATION.md` - Implementation details
- `src-tauri/src/asr/MODEL_SELECTION.md` - Technical guide
