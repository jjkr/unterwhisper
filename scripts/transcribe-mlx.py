#!/usr/bin/env python3
"""
Real-time microphone transcription using MLX and distil-whisper.
Captures audio from the microphone and transcribes it in real-time.
"""

import numpy as np
import sounddevice as sd
import queue
import threading
from collections import deque
import mlx.core as mx
from mlx_whisper import load_models, transcribe

# Configuration
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
CHUNK_DURATION = 10.0  # Process audio in 3-second chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
OVERLAP_DURATION = 9.8  # Overlap between chunks for continuity
OVERLAP_SIZE = int(SAMPLE_RATE * OVERLAP_DURATION)

# Model configuration
# Use MLX-converted model from mlx-community
MODEL_NAME = "mlx-community/whisper-tiny.en-mlx-q4"

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


def transcription_worker(model_path):
    """Worker thread that processes audio chunks and transcribes them."""
    print(f"Loading model: {MODEL_NAME}")
    print("This may take a moment on first run...")
    
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
                # Convert to numpy array and normalize
                audio_array = np.array(list(audio_buffer)[:CHUNK_SIZE], dtype=np.float32)
                
                # Transcribe using MLX Whisper
                result = transcribe(
                    audio_array,
                    path_or_hf_repo=MODEL_NAME,
                    language="en",
                    word_timestamps=False,
                    verbose=False
                )
                
                text = result.get("text", "").strip()
                
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
            continue


def main():
    """Main function to set up audio stream and start transcription."""
    print("Real-time Microphone Transcription with MLX Whisper")
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
    
    # Start transcription worker thread
    transcription_thread = threading.Thread(
        target=transcription_worker,
        args=(MODEL_NAME,),
        daemon=True
    )
    transcription_thread.start()
    
    # Start audio input stream
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=audio_callback,
            blocksize=int(SAMPLE_RATE * 0.1)  # 100ms blocks
        ):
            # Keep main thread alive
            transcription_thread.join()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()