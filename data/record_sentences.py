#!/usr/bin/env python3
"""
Audio Recording Script for TTS Dataset Creation.

This module provides an interactive recording interface for creating high-quality
TTS training datasets. It records audio samples from a list of sentences and
generates metadata in LJSpeech format.

Features:
    - Interactive recording with countdown timer
    - Real-time playback for quality verification
    - Automatic silence trimming with padding
    - Resume capability from last recorded index
    - Support for multiple languages and sample rates

Usage:
    $ uv run record_sentences.py --sentences sentences_en.txt
    $ uv run record_sentences.py --sentences sentences_es.txt --duration 8.0
    $ uv run record_sentences.py --sentences custom.txt --device "USB Microphone"

Requirements:
    - sounddevice: Audio I/O
    - soundfile: WAV file handling
    - numpy: Audio processing
    - Linux: libportaudio2 (apt-get install libportaudio2)

Author: TTS Training Pipeline
License: MIT
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_sentences(path: Path) -> List[str]:
    """
    Read sentences from a text file.
    
    Args:
        path: Path to the text file containing sentences (one per line)
        
    Returns:
        List of non-empty sentences
        
    Raises:
        FileNotFoundError: If the sentences file doesn't exist
        PermissionError: If the file cannot be read
    """
    if not path.exists():
        raise FileNotFoundError(f"Sentences file not found: {path}")
        
    sentences = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    sentences.append(line)
                    
        logger.info(f"Loaded {len(sentences)} sentences from {path}")
        return sentences
        
    except PermissionError as e:
        logger.error(f"Permission denied reading {path}: {e}")
        raise
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error in {path}: {e}")
        raise


def next_index_from_metadata(meta_path: Path, start_index: int) -> int:
    """
    Determine the next available index for recording based on existing metadata.
    
    This function scans the metadata CSV to find the highest index used so far,
    allowing recordings to resume from where they left off.
    
    Args:
        meta_path: Path to the metadata CSV file
        start_index: Default starting index if metadata doesn't exist
        
    Returns:
        Next available index number for recordings
    """
    if not meta_path.exists():
        logger.info(f"Metadata file doesn't exist, starting at index {start_index}")
        return start_index
        
    max_idx = 0
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            for row in reader:
                if not row or len(row) < 1:
                    continue
                    
                fname = row[0].strip()
                try:
                    stem = Path(fname).stem
                    idx = int(stem)
                    max_idx = max(max_idx, idx)
                except (ValueError, AttributeError):
                    # Skip non-numeric filenames
                    continue
                    
    except Exception as e:
        logger.warning(f"Error reading metadata: {e}, starting fresh")
        return start_index
        
    next_idx = max(max_idx + 1, start_index)
    logger.info(f"Resuming from index {next_idx}")
    return next_idx


def record_fixed_duration(
    seconds: float, 
    samplerate: int, 
    channels: int
) -> np.ndarray:
    """
    Record audio for a fixed duration.
    
    Args:
        seconds: Duration to record in seconds
        samplerate: Sample rate in Hz (e.g., 24000)
        channels: Number of channels (1 for mono, 2 for stereo)
        
    Returns:
        Numpy array containing the recorded audio
        
    Raises:
        sd.PortAudioError: If audio device is unavailable
    """
    print(f"🔴 Recording for {seconds:.1f}s @ {samplerate} Hz, {channels} ch ...")
    
    try:
        sd.default.samplerate = samplerate
        sd.default.channels = channels
        
        # Record audio
        audio = sd.rec(int(seconds * samplerate), dtype="float32")
        sd.wait()  # Wait for recording to complete
        
        print("⏹️  Finished recording.")
        return audio.squeeze()  # Remove extra dimensions for mono
        
    except sd.PortAudioError as e:
        logger.error(f"Audio recording failed: {e}")
        raise


def play_audio(audio: np.ndarray, samplerate: int) -> None:
    """
    Play back recorded audio for review.
    
    Args:
        audio: Audio data as numpy array
        samplerate: Sample rate in Hz
        
    Raises:
        sd.PortAudioError: If audio playback fails
    """
    print("🔁 Playing back...")
    
    try:
        sd.play(audio, samplerate=samplerate)
        sd.wait()  # Wait for playback to complete
    except sd.PortAudioError as e:
        logger.error(f"Audio playback failed: {e}")
        print("Warning: Could not play audio. Continuing...")


def trim_silence(
    audio: np.ndarray, 
    threshold: float = 1e-3, 
    frame_size: int = 2048
) -> np.ndarray:
    """
    Remove silence from the beginning and end of audio.
    
    Uses energy-based detection to find the start and end of speech,
    removing silent portions while preserving the actual audio content.
    
    Args:
        audio: Input audio array
        threshold: Energy threshold for silence detection (default: 1e-3)
        frame_size: Size of analysis frames in samples (default: 2048)
        
    Returns:
        Trimmed audio array
    """
    if audio.ndim > 1:
        audio = audio.squeeze()
        
    n_samples = audio.shape[0]
    
    # Find first non-silent frame
    start = 0
    for i in range(0, n_samples, frame_size):
        frame = audio[i:min(i+frame_size, n_samples)]
        if np.max(np.abs(frame)) > threshold:
            start = max(0, i - frame_size // 2)  # Back up half a frame
            break
            
    # Find last non-silent frame
    end = n_samples
    for i in range(n_samples - frame_size, -1, -frame_size):
        frame = audio[max(0, i):min(i+frame_size, n_samples)]
        if np.max(np.abs(frame)) > threshold:
            end = min(n_samples, i + frame_size + frame_size // 2)
            break
            
    # Ensure we have valid audio
    if end <= start:
        logger.warning("No speech detected, returning original audio")
        return audio
        
    return audio[start:end]


def ensure_dirs(wavs_dir: Path, meta_path: Path) -> None:
    """
    Create necessary directories if they don't exist.
    
    Args:
        wavs_dir: Directory for WAV files
        meta_path: Path to metadata CSV file
    """
    wavs_dir.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not meta_path.exists():
        # Create empty metadata file with header
        meta_path.touch()
        logger.info(f"Created metadata file: {meta_path}")


def validate_audio_device(device: Optional[str]) -> bool:
    """
    Validate that the specified audio device exists.
    
    Args:
        device: Device name or ID to validate
        
    Returns:
        True if device is valid or None, False otherwise
    """
    if device is None:
        return True
        
    try:
        devices = sd.query_devices()
        if isinstance(device, str):
            # Check if device name exists
            for d in devices:
                if device.lower() in d['name'].lower():
                    return True
        else:
            # Check if device ID is valid
            device_id = int(device)
            if 0 <= device_id < len(devices):
                return True
    except Exception as e:
        logger.error(f"Error querying audio devices: {e}")
        
    return False


def save_recording(
    audio: np.ndarray,
    output_path: Path,
    samplerate: int
) -> bool:
    """
    Save audio to WAV file with error handling.
    
    Args:
        audio: Audio data to save
        output_path: Path where to save the WAV file
        samplerate: Sample rate of the audio
        
    Returns:
        True if save was successful, False otherwise
    """
    try:
        sf.write(output_path, audio, samplerate, subtype="PCM_16")
        return True
    except Exception as e:
        logger.error(f"Failed to save audio to {output_path}: {e}")
        return False


def main():
    """Main entry point for the recording application."""
    parser = argparse.ArgumentParser(
        description="Interactive TTS Dataset Recorder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --sentences sentences_en.txt
  %(prog)s --sentences sentences_es.txt --duration 8.0
  %(prog)s --sentences custom.txt --device "USB Microphone" --no-trim
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--sentences", 
        type=str, 
        required=True,
        help="Path to text file containing sentences (one per line)"
    )
    
    # Output configuration
    parser.add_argument(
        "--out-dir", 
        type=str, 
        default="wavs",
        help="Directory to save WAV files (default: wavs)"
    )
    parser.add_argument(
        "--metadata", 
        type=str, 
        default="metadata.csv",
        help="Path to metadata CSV file (default: metadata.csv)"
    )
    
    # Audio configuration
    parser.add_argument(
        "--samplerate", 
        type=int, 
        default=24000,
        help="Sample rate in Hz (default: 24000 for SNAC)"
    )
    parser.add_argument(
        "--channels", 
        type=int, 
        default=1,
        choices=[1, 2],
        help="Number of channels: 1=mono, 2=stereo (default: 1)"
    )
    parser.add_argument(
        "--duration", 
        type=float, 
        default=6.0,
        help="Default recording duration in seconds (default: 6.0)"
    )
    
    # File naming
    parser.add_argument(
        "--start-index", 
        type=int, 
        default=1,
        help="Starting index for filenames (default: 1)"
    )
    parser.add_argument(
        "--zero-pad", 
        type=int, 
        default=3,
        help="Zero-padding width for filenames (default: 3 -> 001.wav)"
    )
    
    # Audio processing
    parser.add_argument(
        "--no-trim", 
        action="store_true",
        help="Disable automatic silence trimming"
    )
    parser.add_argument(
        "--trim-threshold",
        type=float,
        default=1e-3,
        help="Energy threshold for silence detection (default: 0.001)"
    )
    parser.add_argument(
        "--padding-ms",
        type=int,
        default=100,
        help="Padding in milliseconds to add after trimming (default: 100)"
    )
    
    # Device configuration
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Audio input device name or ID (default: system default)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate paths
    sentences_path = Path(args.sentences)
    if not sentences_path.exists():
        logger.error(f"Sentences file not found: {sentences_path}")
        sys.exit(1)
        
    wavs_dir = Path(args.out_dir)
    meta_path = Path(args.metadata)
    
    # Create directories
    ensure_dirs(wavs_dir, meta_path)
    
    # Configure audio device
    if args.device is not None:
        if not validate_audio_device(args.device):
            logger.error(f"Invalid audio device: {args.device}")
            print("\nAvailable devices:")
            print(sd.query_devices())
            sys.exit(1)
        sd.default.device = (args.device, args.device)
        logger.info(f"Using audio device: {args.device}")
    
    # Load sentences
    try:
        sentences = read_sentences(sentences_path)
    except Exception as e:
        logger.error(f"Failed to load sentences: {e}")
        sys.exit(1)
        
    if not sentences:
        logger.error("No sentences found in file")
        sys.exit(1)
        
    print(f"\n📚 Loaded {len(sentences)} sentences from {sentences_path}")
    
    # Determine starting index
    idx = next_index_from_metadata(meta_path, args.start_index)
    print(f"📝 Starting at index: {idx:0{args.zero_pad}d}")
    print(f"🎤 Audio settings: {args.samplerate}Hz, {'mono' if args.channels == 1 else 'stereo'}")
    print(f"💾 Saving to: {wavs_dir}")
    print("\n" + "="*80)
    
    # Open metadata file for appending
    try:
        with open(meta_path, "a", encoding="utf-8", newline="") as mf:
            writer = csv.writer(mf, delimiter="|", quoting=csv.QUOTE_MINIMAL)
            
            # Process each sentence
            for si, sentence in enumerate(sentences, start=1):
                recording_complete = False
                
                while not recording_complete:
                    print("\n" + "="*80)
                    print(f"📖 Sentence {si}/{len(sentences)} | File: {idx:0{args.zero_pad}d}.wav")
                    print(f"📝 Text: {sentence}")
                    print("-"*80)
                    
                    # Get recording duration
                    try:
                        duration_input = input(
                            f"⏱️  Press ENTER for {args.duration:.1f}s recording "
                            f"(or type duration, e.g., 5.5): "
                        ).strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\n👋 Exiting gracefully...")
                        return
                        
                    rec_duration = args.duration
                    if duration_input:
                        try:
                            rec_duration = float(duration_input)
                            if rec_duration <= 0 or rec_duration > 60:
                                print("⚠️  Duration must be between 0 and 60 seconds")
                                continue
                        except ValueError:
                            print(f"⚠️  Invalid duration, using default: {args.duration}s")
                    
                    # Countdown
                    print("\n🎬 Get ready...")
                    for t in [3, 2, 1]:
                        print(f"   {t}...")
                        time.sleep(1)
                    
                    # Record audio
                    try:
                        audio = record_fixed_duration(
                            rec_duration, 
                            args.samplerate, 
                            args.channels
                        )
                    except Exception as e:
                        print(f"❌ Recording failed: {e}")
                        continue
                    
                    # Process audio (trimming and padding)
                    if not args.no_trim:
                        trimmed = trim_silence(
                            audio, 
                            threshold=args.trim_threshold, 
                            frame_size=2048
                        )
                        # Add padding to avoid hard cuts
                        pad_samples = int(args.padding_ms * args.samplerate / 1000)
                        audio_out = np.pad(trimmed, (pad_samples, pad_samples), mode="constant")
                    else:
                        audio_out = audio
                    
                    # Review loop
                    while True:
                        choice = input(
                            "\n✨ [A]ccept  🔄 [R]edo  🔊 [P]lay  ⏭️  [S]kip  🚪 [Q]uit: "
                        ).strip().lower()
                        
                        if choice == "p":
                            play_audio(audio_out, samplerate=args.samplerate)
                        elif choice == "r":
                            print("🔄 Redoing this sentence...")
                            break
                        elif choice == "s":
                            print("⏭️  Skipping this sentence")
                            recording_complete = True
                            break
                        elif choice == "q":
                            print("\n👋 Quitting. Progress saved.")
                            return
                        elif choice in ["a", ""]:
                            # Save the recording
                            fname = f"{idx:0{args.zero_pad}d}.wav"
                            out_path = wavs_dir / fname
                            
                            if save_recording(audio_out, out_path, args.samplerate):
                                # Update metadata
                                writer.writerow([fname, sentence])
                                mf.flush()
                                print(f"✅ Saved: {out_path}")
                                idx += 1
                                recording_complete = True
                            else:
                                print("❌ Failed to save audio, please retry")
                            break
                        else:
                            print("❓ Invalid option. Choose A/R/P/S/Q")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("🎉 Recording session complete!")
    print(f"📊 Next file index: {idx:0{args.zero_pad}d}")
    print(f"💾 Audio files saved to: {wavs_dir}")
    print(f"📋 Metadata saved to: {meta_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Recording interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)