# Implementation Plan: ONNX Transcriber

## Overview

This implementation plan breaks down the ONNX Transcriber feature into discrete coding tasks. Each task builds on previous work to incrementally create a fully functional ONNX-based transcriber that is compatible with the existing `WhisperTransformer` interface.

## Tasks

- [x] 1. Add ONNX Runtime dependency and create module structure
  - Add `ort` crate to Cargo.toml dependencies
  - Create `src-tauri/src/asr/onnx.rs` module file
  - Export the module in `src-tauri/src/asr/mod.rs`
  - Add basic module documentation
  - _Requirements: 1.1, 2.1_

- [x] 2. Implement WhisperConfig struct for shared configuration
  - [x] 2.1 Create WhisperConfig struct in a shared location
    - Define struct with num_mel_bins, vocab_size, max_length, num_encoder_layers, num_decoder_layers
    - Implement Default trait for WhisperConfig
    - Add serde Serialize/Deserialize derives for JSON loading
    - _Requirements: 2.3_

  - [ ]* 2.2 Write unit tests for WhisperConfig
    - Test default configuration values
    - Test JSON deserialization from config.json format
    - _Requirements: 2.3_

- [x] 3. Implement model repository mapping
  - [x] 3.1 Create get_onnx_model_info function
    - Implement function that maps model names to (repo_id, revision, model_file) tuples
    - Support all standard Whisper model variants (tiny, base, small, medium, large)
    - Include ONNX-specific repositories (e.g., onnx-community)
    - Add default fallback for unknown model names
    - _Requirements: 1.1, 1.4_

  - [ ]* 3.2 Write property test for model name recognition
    - **Property 1: Model Name Recognition**
    - **Validates: Requirements 1.4**
    - Generate all supported model names and verify each maps to valid repository info
    - _Requirements: 1.4_

- [x] 4. Implement OnnxTranscriber struct and initialization
  - [x] 4.1 Define OnnxTranscriber struct
    - Create struct with encoder_session, decoder_session, tokenizer, config, device fields
    - Add necessary derives and type annotations
    - _Requirements: 2.1_

  - [x] 4.2 Implement execution provider selection
    - Create helper function to map Device to ONNX ExecutionProvider
    - Handle CPU, CUDA, and Metal/CoreML device types
    - Add logging for selected execution provider
    - _Requirements: 7.1, 7.2, 7.3, 7.5_

  - [x] 4.3 Implement new() constructor
    - Download model files from HuggingFace using hf-hub
    - Load config.json and parse into WhisperConfig
    - Load tokenizer.json using tokenizers crate
    - Create ONNX sessions for encoder and decoder with appropriate execution provider
    - Handle errors with descriptive messages
    - _Requirements: 1.1, 1.2, 1.5, 2.1, 6.1, 7.1, 7.2, 7.3_

  - [ ]* 4.4 Write unit test for model initialization
    - Test initialization with "tiny.en" model
    - Verify sessions and tokenizer are loaded
    - _Requirements: 1.1, 1.2, 1.5_

  - [ ]* 4.5 Write unit tests for device selection
    - Test CPU device selection configures CPU execution provider
    - Test GPU device selection (if available)
    - Verify execution provider logging
    - _Requirements: 7.2, 7.3, 7.5_

- [ ] 5. Checkpoint - Ensure initialization tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement config() accessor method
  - [x] 6.1 Add config() method to OnnxTranscriber
    - Return reference to WhisperConfig
    - _Requirements: 2.3_

- [x] 7. Implement mel spectrogram preprocessing
  - [x] 7.1 Create preprocess_mel function
    - Accept mel spectrogram as &[f32]
    - Handle empty input (return early)
    - Pad with zeros if length < n_mels * 3000
    - Truncate if length > n_mels * 3000
    - Reshape to (1, n_mels, 3000) tensor format
    - Return ONNX-compatible tensor
    - _Requirements: 2.4, 2.5_

  - [ ]* 7.2 Write property test for mel normalization
    - **Property 2: Mel Spectrogram Normalization**
    - **Validates: Requirements 2.5**
    - Generate random mel spectrograms of various sizes
    - Verify all are normalized to exactly 3000 frames
    - _Requirements: 2.5_

  - [ ]* 7.3 Write edge case test for empty input
    - Test empty mel spectrogram returns empty string
    - _Requirements: 2.4_

- [x] 8. Implement encoder inference
  - [x] 8.1 Create run_encoder method
    - Accept preprocessed mel tensor
    - Run encoder ONNX session
    - Extract audio features from output
    - Handle inference errors with descriptive messages
    - _Requirements: 3.1, 6.2_

  - [ ]* 8.2 Write unit test for encoder execution
    - Test encoder produces audio features with correct shape
    - _Requirements: 3.1_

- [x] 9. Implement token initialization and special token handling
  - [x] 9.1 Create initialize_tokens method
    - Get token IDs for SOT, language, transcribe, no_timestamps
    - Handle multilingual vs English-only models
    - Return initial token sequence
    - Handle missing token errors
    - _Requirements: 3.2, 6.4_

  - [x] 9.2 Create get_token_id helper method
    - Look up token ID from tokenizer
    - Return descriptive error if token not found
    - _Requirements: 6.4_

  - [ ]* 9.3 Write edge case tests for missing tokens
    - Test error handling when special tokens are missing
    - _Requirements: 6.4_

- [-] 10. Implement repetition penalty logic
  - [x] 10.1 Create apply_repetition_penalty function
    - Track token occurrence counts
    - Apply penalty factor (1.1) exponentially based on count
    - Divide positive logits, multiply negative logits
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 11. Implement token sampling and generation
  - [x] 11.1 Create sample_token method
    - Extract logits for last position
    - Apply repetition penalty
    - Apply temperature if > 0
    - Select token greedily (temperature 0) or sample (temperature > 0)
    - _Requirements: 3.3_

  - [x] 11.2 Create generate_tokens method
    - Initialize token sequence
    - Loop until EOT or max_length
    - Run decoder with current tokens and audio features
    - Sample next token
    - Check for EOT and max_length stopping conditions
    - Return token sequence
    - _Requirements: 3.2, 3.3, 3.4, 3.5_

  - [ ]* 11.3 Write property test for deterministic transcription
    - **Property 3: Deterministic Transcription**
    - **Validates: Requirements 3.3**
    - Generate random mel spectrograms
    - Transcribe each multiple times with temperature 0
    - Verify outputs are identical
    - _Requirements: 3.3_

  - [ ]* 11.4 Write property test for token sequence bounds
    - **Property 4: Token Sequence Length Bounds**
    - **Validates: Requirements 3.4, 3.5**
    - Generate random mel spectrograms
    - Verify token sequences don't exceed max_length
    - Verify sequences end with EOT or at max_length
    - _Requirements: 3.4, 3.5_

- [x] 12. Checkpoint - Ensure token generation tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Implement decoder inference
  - [x] 13.1 Create run_decoder method
    - Accept input tokens and audio features
    - Optionally accept past key-values cache
    - Run decoder ONNX session
    - Extract logits and updated cache
    - Handle inference errors
    - _Requirements: 3.2, 6.2_

  - [ ]* 13.2 Write unit test for decoder execution
    - Test decoder produces logits with correct shape
    - _Requirements: 3.2_

- [x] 14. Implement token decoding and text cleanup
  - [x] 14.1 Create decode_and_clean method
    - Decode token sequence using tokenizer
    - Remove special tokens: <|startoftranscript|>, <|transcribe|>, <|notimestamps|>, <|endoftext|>, <|en|>
    - Trim leading and trailing whitespace
    - Handle tokenizer errors
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 6.3_

  - [ ]* 14.2 Write property test for valid token decoding
    - **Property 5: Valid Token Decoding**
    - **Validates: Requirements 4.1**
    - Generate random valid token sequences
    - Verify all decode successfully
    - _Requirements: 4.1_

  - [ ]* 14.3 Write property test for special token removal
    - **Property 6: Special Token Removal**
    - **Validates: Requirements 4.2, 4.3**
    - Generate random transcriptions
    - Verify output never contains special token strings
    - _Requirements: 4.2, 4.3_

  - [ ]* 14.4 Write property test for whitespace trimming
    - **Property 7: Whitespace Trimming**
    - **Validates: Requirements 4.4**
    - Generate random transcriptions
    - Verify output has no leading/trailing whitespace
    - _Requirements: 4.4_

- [x] 15. Implement main transcribe_from_mel method
  - [x] 15.1 Wire together the complete transcription pipeline
    - Preprocess mel spectrogram
    - Run encoder to get audio features
    - Initialize tokens
    - Generate tokens with decoder
    - Decode and clean text
    - Return final transcription
    - _Requirements: 2.2, 3.1, 3.2, 4.1, 4.2, 4.3, 4.4_

  - [ ]* 15.2 Write integration test for full transcription pipeline
    - Test end-to-end transcription with sample mel spectrogram
    - Verify output is valid text
    - _Requirements: 2.2_

- [ ] 16. Add thread safety verification
  - [ ]* 16.1 Write property test for thread safety
    - **Property 8: Thread Safety**
    - **Validates: Requirements 8.2, 8.3**
    - Create Arc<Mutex<OnnxTranscriber>>
    - Spawn multiple threads calling transcribe_from_mel concurrently
    - Verify no panics, data races, or deadlocks
    - Verify all calls produce valid results
    - _Requirements: 8.2, 8.3_

- [ ] 17. Add error handling edge case tests
  - [ ]* 17.1 Write edge case tests for error conditions
    - Test model loading failure with invalid model name
    - Test inference failure with invalid tensor shapes
    - Test tokenizer failure scenarios
    - Test GPU unavailable fallback to CPU
    - _Requirements: 1.3, 6.1, 6.2, 6.3, 7.4_

- [ ] 18. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- The implementation follows the existing WhisperTransformer interface for compatibility
