#!/usr/bin/env python3
"""
Real-time microphone transcription using MLX and Parakeet.
Captures audio from the microphone and transcribes it in real-time.

Requires: pip install mlx-audio sounddevice numpy dacite
"""

import numpy as np
import sounddevice as sd
import queue
from collections import deque
import mlx.core as mx
from mlx_audio.stt.utils import load_model

# Configuration
SAMPLE_RATE = 16000  # Parakeet expects 16kHz audio
CHUNK_DURATION = 10.0  # Process audio in 10-second chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
OVERLAP_DURATION = 9.9  # Overlap between chunks for continuity
OVERLAP_SIZE = int(SAMPLE_RATE * OVERLAP_DURATION)

# Model configuration
MODEL_NAME = "mlx-community/parakeet-tdt-0.6b-v2"

# Global queue for audio data
audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    """Callback function for audio input stream."""
    if status:
        print(f"Audio callback status: {status}")
    # Convert to mono if stereo and add to queue
    audio_data = indata.copy()
    if audio_data.shape[1] > 1:
        audio_data = audio_data.mean(axis=1)
    else:
        audio_data = audio_data.flatten()
    audio_queue.put(audio_data)


def transcription_worker(model):
    """Worker that processes audio chunks and transcribes them."""
    # Buffer to accumulate audio samples
    audio_buffer = deque(maxlen=CHUNK_SIZE + OVERLAP_SIZE)
    
    print("\n=== Starting transcription ===")
    print("Speak into your microphone. Press Ctrl+C to stop.\n")
    
    while True:
        try:
            # Get audio data from queue
            chunk = audio_queue.get(timeout=0.1)
            audio_buffer.extend(chunk)
            
            # Process when we have enough audio
            if len(audio_buffer) >= CHUNK_SIZE:
                # Convert to numpy first, then to mlx array
                audio_np = np.array(list(audio_buffer)[:CHUNK_SIZE], dtype=np.float32)
                audio_array = mx.array(audio_np)
                
                # Transcribe using the model's decode_chunk method (takes audio directly)
                result = model.decode_chunk(audio_array, verbose=False)
                
                text = result.text.strip() if hasattr(result, 'text') else str(result).strip()
                
                # Only print if there's actual transcribed text
                if text and text not in ["", ".", "...", "Thank you."]:
                    print(f"Transcription: {text}")
                
                # Keep overlap for next chunk
                if len(audio_buffer) > OVERLAP_SIZE:
                    # Remove old samples, keeping overlap
                    for _ in range(CHUNK_SIZE - OVERLAP_SIZE):
                        if audio_buffer:
                            audio_buffer.popleft()
        
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            print("\n\nStopping transcription...")
            break
        except Exception as e:
            print(f"Error during transcription: {e}")
            import traceback
            traceback.print_exc()
            continue


def main():
    """Main function to set up audio stream and start transcription."""
    print("Real-time Microphone Transcription with MLX Audio (Parakeet)")
    print("=" * 60)
    
    # List available audio devices
    print("\nAvailable audio input devices:")
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  [{idx}] {device['name']}")
    
    # Use default input device
    default_device = sd.default.device[0]
    print(f"\nUsing input device: {devices[default_device]['name']}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Chunk duration: {CHUNK_DURATION} seconds\n")
    
    # Load model
    print(f"Loading model: {MODEL_NAME}")
    print("This may take a moment on first run...")
    model = load_model(MODEL_NAME)
    print("Model loaded!")
    
    # Start audio input stream and process in main thread
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=audio_callback,
            blocksize=int(SAMPLE_RATE * 0.1)  # 100ms blocks
        ):
            transcription_worker(model)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
