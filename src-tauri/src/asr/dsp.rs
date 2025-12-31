//! Digital Signal Processing module
//!
//! This module provides core DSP functionality for audio signal processing,
//! including Fast Fourier Transform (FFT) operations, windowing functions,
//! power spectrum computation, and mel filterbank application.
//!
//! # Examples
//!
//! ## FFT Transform
//!
//! ```
//! use whisper_transcribe::dsp;
//!
//! // Transform an audio frame to frequency domain
//! let audio_samples = vec![1.0, 0.0, -1.0, 0.0];
//! let spectrum = dsp::fft(&audio_samples).expect("FFT should succeed");
//! // Real FFT returns N/2+1 bins (positive frequencies only)
//! assert_eq!(spectrum.len(), 3); // 4/2+1 = 3
//! ```
//!
//! ## Complete Audio Processing Pipeline (Simple API)
//!
//! ```
//! use whisper_transcribe::dsp;
//!
//! // 1. Create Hann window
//! let window = dsp::hann_window(400).unwrap();
//!
//! // 2. Apply window to audio frame
//! let audio = vec![0.5; 400];
//! let windowed: Vec<f32> = audio.iter()
//!     .zip(window.iter())
//!     .map(|(s, w)| s * w)
//!     .collect();
//!
//! // 3. Compute FFT
//! let fft_output = dsp::fft(&windowed).unwrap();
//!
//! // 4. Compute mel features (combines power spectrum + mel filters)
//! let mel_features = dsp::mel(&fft_output, 80).unwrap();
//! assert_eq!(mel_features.len(), 80);
//! ```
//!
//! ## Lower-Level API (if you need more control)
//!
//! ```
//! use whisper_transcribe::dsp;
//!
//! let audio = vec![0.5; 400];
//! let window = dsp::hann_window(400).unwrap();
//! let windowed: Vec<f32> = audio.iter()
//!     .zip(window.iter())
//!     .map(|(s, w)| s * w)
//!     .collect();
//!
//! let fft_output = dsp::fft(&windowed).unwrap();
//!
//! // Compute mel features directly from FFT output
//! let mel_features = dsp::mel(&fft_output, 80).unwrap();
//! ```

use candle_transformers::models::whisper::N_FFT;
use num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use serde_json::map::Iter;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// FFT Implementation
// ============================================================================

/// Internal context for FFT operations, cached per thread and per size.
struct FftContext {
    fft: Arc<dyn RealToComplex<f32>>,
    input_buffer: Vec<f32>,
    scratch_buffer: Vec<Complex<f32>>,
}

thread_local! {
    /// Thread-local cache of FFT planners and scratch buffers, keyed by FFT size.
    static FFT_CACHE: RefCell<HashMap<usize, FftContext>> = RefCell::new(HashMap::new());
}

/// Performs a forward real FFT on the input audio frame.
///
/// This function automatically handles FFT planner creation and buffer management
/// using thread-local storage. The first call for a given input size will initialize
/// the FFT planner; subsequent calls will reuse the cached planner and buffers.
///
/// The function uses `realfft` which exploits the fact that audio input is real-valued,
/// computing only the positive frequency bins (0 to N/2). This provides ~2x speedup
/// compared to complex FFT and is ideal for Whisper's mel spectrogram computation.
///
/// # Arguments
/// * `input` - Audio samples (any non-zero length)
///
/// # Returns
/// * `Ok(Vec<Complex<f32>>)` - Frequency spectrum with length N/2+1 (positive frequencies only)
/// * `Err(String)` - Error if input size is zero
///
/// # Examples
///
/// ```
/// use whisper_transcribe::dsp::fft;
///
/// let samples = vec![1.0, 0.0, -1.0, 0.0];
/// let spectrum = fft(&samples).unwrap();
/// assert_eq!(spectrum.len(), 3); // N/2+1 = 4/2+1 = 3
/// ```
pub fn fft(input: Vec<f32>) -> Result<Vec<f32>, String> {
    let size = input.len();
    FFT_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();

        if !cache.contains_key(&size) {
            let mut planner = RealFftPlanner::new();
            let fft = planner.plan_fft_forward(size);
            let scratch_len = fft.get_scratch_len();
            let scratch_buffer = vec![Complex::new(0.0, 0.0); scratch_len];
            let input_buffer = fft.make_input_vec();

            cache.insert(size, FftContext {
                fft,
                input_buffer,
                scratch_buffer,
            });
        }

        let context = cache.get_mut(&size).unwrap();
        let mut output_buffer = context.fft.make_output_vec();
        context.input_buffer = input;

        context.fft.process_with_scratch(
            &mut context.input_buffer,
            &mut output_buffer,
            &mut context.scratch_buffer
        ).map_err(|e| format!("FFT processing failed: {:?}", e))?;

        let fft_magnitude = output_buffer.iter().map(|c| c.norm_sqr());

        Ok(fft_magnitude.collect())
    })
}

// ============================================================================
// Utility Functions
// ============================================================================

thread_local! {
    /// Thread-local cache of Hann windows, keyed by window size.
    static HANN_CACHE: RefCell<HashMap<usize, Vec<f32>>> = RefCell::new(HashMap::new());
}

/// Wrapper struct to ensure proper alignment for f32 data
#[repr(C, align(4))]
struct AlignedBytes<const N: usize>([u8; N]);

/// Mel filter bank for 80 mel bins (embedded at compile time with proper alignment)
static MEL_FILTERS_80_BYTES: &AlignedBytes<{ include_bytes!("../assets/melfilters.bytes").len() }> = 
    &AlignedBytes(*include_bytes!("../assets/melfilters.bytes"));

/// Mel filter bank for 128 mel bins (embedded at compile time with proper alignment)
static MEL_FILTERS_128_BYTES: &AlignedBytes<{ include_bytes!("../assets/melfilters128.bytes").len() }> =
    &AlignedBytes(*include_bytes!("../assets/melfilters128.bytes"));

/// Creates a Hann window of the specified size.
///
/// The Hann window is a smooth tapering function commonly used in audio signal
/// processing to reduce spectral leakage when performing FFT analysis.
///
/// # Arguments
/// * `size` - The window size (must be positive)
///
/// # Returns
/// * `Ok(Vec<f32>)` - Hann window coefficients with length equal to size
/// * `Err(String)` - Error if size is zero
///
/// # Examples
///
/// ```
/// use whisper_transcribe::dsp::hann_window;
///
/// let window = hann_window(8).unwrap();
/// assert_eq!(window.len(), 8);
/// ```
pub fn hann_window(size: usize) -> Result<Vec<f32>, String> {
    if size == 0 {
        return Err("Window size must be positive, got: 0".to_string());
    }

    HANN_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();

        if !cache.contains_key(&size) {
            let window: Vec<f32> = (0..size)
                .map(|i| {
                    0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / size as f32).cos())
                })
                .collect();
            cache.insert(size, window);
        }

        Ok(cache.get(&size).unwrap().clone())
    })
}

/// Computes mel-scale features from FFT power spectrum output.
///
/// This function applies mel filters to the power spectrum.
///
/// # Arguments
/// * `fft_output` - Power spectrum from FFT (length = n_fft/2 + 1 for real FFT)
/// * `n_mel_bins` - Number of mel bins (80 or 128)
///
/// # Returns
/// * `Ok(Vec<f32>)` - Mel features with length equal to n_mel_bins
/// * `Err(String)` - Error if unsupported number of mel bins or dimension mismatch
///
/// # Examples
///
/// ```
/// use whisper_transcribe::dsp;
///
/// let fft_output = vec![1.0; 201];
/// let mel_features = dsp::mel(&fft_output, 80).unwrap();
/// assert_eq!(mel_features.len(), 80);
/// ```
pub fn mel(fft_output: &[f32], n_mel_bins: usize) -> Result<Vec<f32>, String> {
    let mel_bytes: &[u8] = match n_mel_bins {
        80 => &MEL_FILTERS_80_BYTES.0,
        128 => &MEL_FILTERS_128_BYTES.0,
        _ => return Err(format!(
            "Unsupported number of mel bins: {}. Supported values are 80 or 128",
            n_mel_bins
        )),
    };
    let mel_filters: &[f32] = bytemuck::cast_slice(mel_bytes);
    
    let n_fft_bins = fft_output.len();
    let expected_filter_size = n_mel_bins * n_fft_bins;
    if mel_filters.len() != expected_filter_size {
        return Err(format!(
            "Mel filter size mismatch: expected {} (n_mel_bins={} * fft_bins={}), got {}. FFT output length may be incorrect.",
            expected_filter_size, n_mel_bins, n_fft_bins, mel_filters.len()
        ));
    }

    // Allocate output vector
    let mut mel_features = vec![0.0f32; n_mel_bins];

    // Optimized mel computation using transposed iteration order
    // 
    // Performance optimizations:
    // 1. Iterate over frequency bins in outer loop for better cache locality
    // 2. Inner loop writes sequentially to mel_features (cache-friendly)
    // 3. Compiler can potentially auto-vectorize the inner loop
    //
    // This is O(n_fft_bins × n_mel_bins) with good cache utilization:
    // - Better cache utilization (sequential writes vs scattered reads)
    // - Potential SIMD auto-vectorization on the inner loop
    for mel_bin in 0..n_mel_bins {
        // Accumulate this frequency's contribution to all mel bins
        // This is a column of the mel filter matrix multiplied by the power value
        for (freq_bin, &power) in fft_output.iter().enumerate() {
            let filter_idx = mel_bin * n_fft_bins + freq_bin;
            mel_features[mel_bin] += power * mel_filters[filter_idx];
        }
        mel_features[mel_bin] = mel_features[mel_bin].max(1e-10).log10();
    }

    Ok(mel_features)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    mod fft_tests {
        use super::*;
        
        #[test]
        fn test_fft_impulse() {
            let input = vec![1.0, 0.0, 0.0, 0.0];
            let result = fft(input).unwrap();
            assert_eq!(result.len(), 3);
            
            for (i, &power) in result.iter().enumerate() {
                assert!(
                    (power - 1.0).abs() < 1e-6,
                    "Bin {} power should be 1.0, got {}",
                    i, power
                );
            }
        }
        
        #[test]
        fn test_fft_sine_wave() {
            let size = 8;
            let freq = 1.0;
            let input: Vec<f32> = (0..size)
                .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / size as f32).sin())
                .collect();
            
            let result = fft(input).unwrap();
            assert_eq!(result.len(), 5);
            
            // Result is already power (magnitude squared)
            assert!(result[1] > 9.0); // Power > 9.0 means magnitude > 3.0
            assert!(result[0] < 1e-5);
        }
        
        #[test]
        fn test_fft_all_zeros() {
            let input = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            let result = fft(input).unwrap();
            assert_eq!(result.len(), 5);
            
            for (i, &power) in result.iter().enumerate() {
                assert!(
                    power.abs() < 1e-10,
                    "Bin {} power should be zero, got {}",
                    i, power
                );
            }
        }
        
        #[test]
        fn test_fft_all_ones() {
            let input = vec![1.0, 1.0, 1.0, 1.0];
            let result = fft(input).unwrap();
            assert_eq!(result.len(), 3);
            
            // DC bin should have power = 16.0 (magnitude = 4.0, power = 4.0^2)
            assert!((result[0] - 16.0).abs() < 1e-6);
            
            for i in 1..result.len() {
                assert!(result[i] < 1e-5);
            }
        }
        
        #[test]
        fn test_fft_minimum_size() {
            let input = vec![1.0, -1.0];
            let result = fft(input).unwrap();
            assert_eq!(result.len(), 2);
            
            assert!(result[0].abs() < 1e-6);
            assert!((result[1] - 4.0).abs() < 1e-6); // Power = 2.0^2 = 4.0
        }
        
        #[test]
        fn test_fft_maximum_size() {
            let size = 65536;
            let input = vec![1.0; size];
            let result = fft(input).unwrap();
            assert_eq!(result.len(), size / 2 + 1);
            let expected_power = (size as f32).powi(2);
            assert!((result[0] - expected_power).abs() < 1e3);
        }
        
        #[test]
        fn test_fft_alternating_values() {
            let input = vec![1.0, -1.0, 1.0, -1.0];
            let result = fft(input).unwrap();
            assert_eq!(result.len(), 3);
            
            assert!(result[0].abs() < 1e-6);
            
            let nyquist_idx = result.len() - 1;
            assert!(result[nyquist_idx] > 9.0); // Power > 9.0 means magnitude > 3.0
        }
        
        #[test]
        fn test_error_message_size_zero() {
            let input: Vec<f32> = vec![];
            let result = fft(input);
            assert!(result.is_err());
        }
        
        #[test]
        fn test_non_power_of_two_sizes_work() {
            let sizes = vec![3, 5, 6, 7, 9, 10, 100, 400];
            
            for size in sizes {
                let input = vec![1.0; size];
                let result = fft(input);
                assert!(result.is_ok());
                
                let spectrum = result.unwrap();
                let expected_len = size / 2 + 1;
                assert_eq!(spectrum.len(), expected_len);
            }
        }
        
        #[test]
        fn test_whisper_fft_size() {
            let input = vec![0.5; 400];
            let result = fft(input);
            assert!(result.is_ok());
            let spectrum = result.unwrap();
            assert_eq!(spectrum.len(), 201);
        }
    }


    #[cfg(test)]
    mod property_tests {
        use super::*;
        use proptest::prelude::*;
        use rand::SeedableRng;
        use realfft::RealFftPlanner;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]
            
            #[test]
            fn test_valid_size_processing_success(
                power in 1u32..=16u32,
                seed in any::<u64>()
            ) {
                let size = 2usize.pow(power);
                
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                use rand::Rng;
                let input: Vec<f32> = (0..size)
                    .map(|_| rng.gen_range(-1.0..=1.0))
                    .collect();
                
                let result = fft(input);
                prop_assert!(result.is_ok(), "FFT should succeed for power-of-2 size {}", size);
                
                let spectrum = result.unwrap();
                let expected_len = size / 2 + 1;
                prop_assert_eq!(spectrum.len(), expected_len, 
                    "Output length should be N/2+1 for size {}", size);
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]
            
            #[test]
            fn test_fft_power_spectrum_properties(
                power in 1u32..=12u32,
                seed in any::<u64>()
            ) {
                let size = 2usize.pow(power);
                
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                use rand::Rng;
                let input: Vec<f32> = (0..size)
                    .map(|_| rng.gen_range(-1.0..=1.0))
                    .collect();
                
                let result = fft(input);
                prop_assert!(result.is_ok(), "FFT should succeed");
                let power_spectrum = result.unwrap();
                
                // All power values should be non-negative
                for (i, &power) in power_spectrum.iter().enumerate() {
                    prop_assert!(
                        power >= 0.0,
                        "Power at bin {} should be non-negative, got {}",
                        i, power
                    );
                }
                
                // Power spectrum should have correct length
                prop_assert_eq!(power_spectrum.len(), size / 2 + 1);
            }
        }
    }


    mod hann_window_tests {
        use super::*;

        #[test]
        fn test_hann_window_basic() {
            let window = hann_window(8).unwrap();
            assert_eq!(window.len(), 8);
        }

        #[test]
        fn test_hann_window_starts_near_zero() {
            let window = hann_window(400).unwrap();
            assert!(window[0].abs() < 1e-6, "Window should start near zero");
        }

        #[test]
        fn test_hann_window_peaks_at_center() {
            let window = hann_window(400).unwrap();
            assert!(
                (window[200] - 1.0).abs() < 0.01,
                "Window should peak near 1.0 at center"
            );
        }

        #[test]
        fn test_hann_window_symmetry() {
            let window = hann_window(400).unwrap();
            for i in 0..200 {
                assert!(
                    (window[i] - window[399 - i]).abs() < 0.01,
                    "Window should be approximately symmetric"
                );
            }
        }

        #[test]
        fn test_hann_window_zero_size() {
            let result = hann_window(0);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("positive"));
        }

        #[test]
        fn test_hann_window_caching() {
            let window1 = hann_window(400).unwrap();
            let window2 = hann_window(400).unwrap();
            assert_eq!(window1, window2);
        }

        #[test]
        fn test_hann_window_whisper_size() {
            let window = hann_window(400).unwrap();
            assert_eq!(window.len(), 400);
        }
    }

    mod mel_tests {
        use super::*;

        #[test]
        fn test_mel_basic_80_bins() {
            let fft_output = vec![1.25; 201]; // Power = 1.0^2 + 0.5^2 = 1.25
            let mel_features = mel(&fft_output, 80).unwrap();
            assert_eq!(mel_features.len(), 80);
        }

        #[test]
        fn test_mel_basic_128_bins() {
            let fft_output = vec![1.0; 201];
            let mel_features = mel(&fft_output, 128).unwrap();
            assert_eq!(mel_features.len(), 128);
        }

        #[test]
        fn test_mel_unsupported_bins() {
            let fft_output = vec![1.0; 201];
            let result = mel(&fft_output, 64);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("Unsupported"));
        }

        #[test]
        fn test_mel_consistent_results() {
            let fft_output = vec![1.0; 201];
            
            let mel1 = mel(&fft_output, 80).unwrap();
            let mel2 = mel(&fft_output, 80).unwrap();
            
            assert_eq!(mel1, mel2);
        }

        #[test]
        fn test_mel_non_zero_output() {
            let fft_output = vec![5.0; 201]; // Power = 2.0^2 + 1.0^2 = 5.0
            let mel_features = mel(&fft_output, 80).unwrap();
            
            let total_energy: f32 = mel_features.iter().sum();
            assert!(total_energy > 0.0);
        }

        #[test]
        fn test_mel_zero_input() {
            let fft_output = vec![0.0; 201];
            let mel_features = mel(&fft_output, 80).unwrap();
            
            for &f in &mel_features {
                assert_eq!(f, 0.0);
            }
        }
    }
}
