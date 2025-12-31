# Requirements Document

## Introduction

This feature adds an ONNX-based transcriber implementation that provides an alternative to the existing Candle-based WhisperTransformer. The ONNX transcriber will use the `ort` (ONNX Runtime) library to run Whisper models, offering better cross-platform compatibility and potentially improved performance on certain hardware configurations.

## Glossary

- **ONNX**: Open Neural Network Exchange, an open format for representing machine learning models
- **ORT**: ONNX Runtime, a cross-platform inference engine for ONNX models
- **Transcriber**: A component that converts audio mel spectrograms into text transcriptions
- **WhisperTransformer**: The existing Candle-based transcriber implementation
- **OnnxTranscriber**: The new ONNX-based transcriber implementation
- **Mel Spectrogram**: A preprocessed audio representation used as input to Whisper models
- **RealtimeTranscriber**: The orchestration component that manages the transcription pipeline

## Requirements

### Requirement 1: ONNX Model Loading

**User Story:** As a developer, I want to load Whisper models in ONNX format, so that I can use ONNX Runtime for inference.

#### Acceptance Criteria

1. WHEN the OnnxTranscriber is initialized with a model name, THE OnnxTranscriber SHALL download the ONNX model files from HuggingFace
2. WHEN the model files are downloaded, THE OnnxTranscriber SHALL load the encoder and decoder ONNX models into memory
3. WHEN the model loading fails, THE OnnxTranscriber SHALL return a descriptive error message
4. THE OnnxTranscriber SHALL support the same model names as WhisperTransformer (tiny, base, small, medium, large variants)
5. THE OnnxTranscriber SHALL load the tokenizer from the model repository

### Requirement 2: Transcription Interface Compatibility

**User Story:** As a developer, I want the OnnxTranscriber to have the same interface as WhisperTransformer, so that I can swap implementations without changing other code.

#### Acceptance Criteria

1. THE OnnxTranscriber SHALL implement a `new` method that accepts model name, device, and optional language parameters
2. THE OnnxTranscriber SHALL implement a `transcribe_from_mel` method that accepts a mel spectrogram and returns transcribed text
3. THE OnnxTranscriber SHALL implement a `config` method that returns model configuration information
4. WHEN transcribe_from_mel receives an empty mel spectrogram, THE OnnxTranscriber SHALL return an empty string
5. WHEN transcribe_from_mel receives a mel spectrogram, THE OnnxTranscriber SHALL pad or truncate it to exactly 3000 time steps

### Requirement 3: ONNX Inference Execution

**User Story:** As a developer, I want the OnnxTranscriber to run inference using ONNX Runtime, so that I can leverage ONNX's optimizations and hardware support.

#### Acceptance Criteria

1. WHEN transcribe_from_mel is called, THE OnnxTranscriber SHALL run the encoder model on the mel spectrogram to produce audio features
2. WHEN audio features are generated, THE OnnxTranscriber SHALL run the decoder model iteratively to generate tokens
3. WHEN generating tokens, THE OnnxTranscriber SHALL use greedy decoding with temperature 0.0 by default
4. WHEN the end-of-text token is generated, THE OnnxTranscriber SHALL stop token generation
5. WHEN the maximum token length is reached, THE OnnxTranscriber SHALL stop token generation

### Requirement 4: Token Decoding and Text Generation

**User Story:** As a developer, I want the OnnxTranscriber to decode tokens into readable text, so that I can present transcriptions to users.

#### Acceptance Criteria

1. WHEN tokens are generated, THE OnnxTranscriber SHALL decode them using the tokenizer
2. WHEN decoding is complete, THE OnnxTranscriber SHALL remove special tokens from the output text
3. THE OnnxTranscriber SHALL remove start-of-transcript, transcribe, no-timestamps, end-of-text, and language tokens
4. WHEN the decoded text contains leading or trailing whitespace, THE OnnxTranscriber SHALL trim it
5. THE OnnxTranscriber SHALL return the cleaned transcription text

### Requirement 5: Repetition Prevention

**User Story:** As a user, I want transcriptions to avoid repetitive text, so that the output is more natural and accurate.

#### Acceptance Criteria

1. WHEN generating tokens, THE OnnxTranscriber SHALL track token occurrence counts
2. WHEN a token has been generated multiple times, THE OnnxTranscriber SHALL apply a repetition penalty to its logits
3. THE OnnxTranscriber SHALL use a default repetition penalty of 1.1
4. WHEN applying the penalty, THE OnnxTranscriber SHALL divide positive logits and multiply negative logits by the penalty factor
5. THE OnnxTranscriber SHALL apply the penalty exponentially based on token occurrence count

### Requirement 6: Error Handling

**User Story:** As a developer, I want clear error messages when transcription fails, so that I can diagnose and fix issues.

#### Acceptance Criteria

1. WHEN model loading fails, THE OnnxTranscriber SHALL return an error with the failure reason
2. WHEN inference fails, THE OnnxTranscriber SHALL return an error with the failure reason
3. WHEN tokenizer operations fail, THE OnnxTranscriber SHALL return an error with the failure reason
4. WHEN a required token is not found in the tokenizer, THE OnnxTranscriber SHALL return an error indicating the missing token
5. THE OnnxTranscriber SHALL use Rust's Result type for all fallible operations

### Requirement 7: Device Support

**User Story:** As a developer, I want to specify which device to use for inference, so that I can optimize performance for different hardware.

#### Acceptance Criteria

1. THE OnnxTranscriber SHALL accept a device parameter during initialization
2. WHEN the device is CPU, THE OnnxTranscriber SHALL configure ONNX Runtime to use CPU execution
3. WHEN the device is GPU, THE OnnxTranscriber SHALL configure ONNX Runtime to use GPU execution if available
4. WHEN GPU execution is requested but unavailable, THE OnnxTranscriber SHALL fall back to CPU execution
5. THE OnnxTranscriber SHALL log the selected execution provider

### Requirement 8: Integration with RealtimeTranscriber

**User Story:** As a developer, I want to use OnnxTranscriber with the existing RealtimeTranscriber, so that I can choose between Candle and ONNX implementations.

#### Acceptance Criteria

1. THE OnnxTranscriber SHALL be compatible with the Arc<Mutex<T>> pattern used by RealtimeTranscriber
2. THE OnnxTranscriber SHALL support concurrent access from multiple threads
3. THE OnnxTranscriber SHALL maintain thread safety for all public methods
4. WHEN used in RealtimeTranscriber, THE OnnxTranscriber SHALL process mel spectrograms at the same rate as WhisperTransformer
5. THE OnnxTranscriber SHALL provide the same configuration interface as WhisperTransformer
