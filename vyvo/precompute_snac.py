#!/usr/bin/env python3
"""
SNAC Audio Tokenization Script.

This module precomputes SNAC (Scalable Neural Audio Codec) token sequences from
audio files for VyvoTTS training. It converts 24kHz WAV files into hierarchical
token representations suitable for language model training.

The SNAC codec provides:
    - Hierarchical audio representation with 3 quality layers
    - Efficient compression while preserving speech quality
    - Token sequences compatible with language model training

Features:
    - Batch processing of audio files from metadata
    - 7-way token packing for efficient LM training
    - Progress tracking and error handling
    - GPU acceleration support

Usage:
    $ uv run precompute_snac.py
    $ uv run precompute_snac.py --metadata ../data/metadata.csv --output ../data/train_snac.jsonl
    $ uv run precompute_snac.py --batch-size 16 --device cuda

Requirements:
    - snac: SNAC audio codec
    - torch: PyTorch backend
    - soundfile: Audio file I/O
    - CUDA (optional): GPU acceleration

Author: TTS Training Pipeline
License: MIT
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import soundfile as sf

try:
    from snac import SNAC
except ImportError:
    print(
        "Error: SNAC not installed. Please run: uv add git+https://github.com/hubertsiuzdak/snac.git"
    )
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Token constants matching VyvoTTS model
TOKENISER_LEN = 151669
AUDIO_BASE = TOKENISER_LEN + 10

# SNAC codec configuration
EXPECTED_SAMPLE_RATE = 24000  # SNAC encoder expects 24 kHz
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"


def validate_audio_file(
    audio_path: str, expected_sr: int = EXPECTED_SAMPLE_RATE
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Load and validate an audio file for SNAC encoding.

    Args:
        audio_path: Path to the audio file
        expected_sr: Expected sample rate (default: 24000Hz)

    Returns:
        Tuple of (audio_data, sample_rate) or (None, None) if invalid
    """
    try:
        audio, sr = sf.read(audio_path)

        # Validate sample rate
        if sr != expected_sr:
            logger.warning(
                f"Sample rate mismatch in {audio_path}: "
                f"got {sr}Hz, expected {expected_sr}Hz"
            )
            # You could resample here if needed
            return None, None

        # Ensure mono audio
        if audio.ndim > 1:
            logger.warning(f"Converting stereo to mono for {audio_path}")
            audio = audio.mean(axis=1)

        # Check for valid audio
        if len(audio) == 0:
            logger.error(f"Empty audio file: {audio_path}")
            return None, None

        # Check for NaN or Inf values
        if not np.isfinite(audio).all():
            logger.error(f"Invalid audio values (NaN/Inf) in {audio_path}")
            return None, None

        return audio, sr

    except Exception as e:
        logger.error(f"Failed to load audio {audio_path}: {e}")
        return None, None


def pack_codes(
    snac_codes: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> List[int]:
    """
    Pack SNAC multi-layer codes into a single token stream for LM training.

    SNAC.encode returns 3 tensors representing different quality layers:
        - Layer 1: Base layer (75Hz frame rate, 4096 codes)
        - Layer 2: Enhancement layer (150Hz frame rate, 4096 codes)
        - Layer 3: Fine detail layer (300Hz frame rate, 4096 codes)

    The packing scheme interleaves codes from all layers into a 7-token
    pattern per base frame, then offsets by AUDIO_BASE.

    Args:
        snac_codes: Tuple of 3 tensors from SNAC encoder

    Returns:
        List of packed token IDs for language model
    """
    l1, l2, l3 = snac_codes  # shapes: [B, T], [B, T*2], [B, T*4]

    if l1.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got {l1.shape[0]}")

    packed = []
    T = l1.shape[1]  # Number of base frames

    for i in range(T):
        # Pack 7 codes per base frame following VyvoTTS convention
        # This specific ordering is crucial for compatibility
        try:
            frame_codes = [
                int(l1[0, i]),  # Base layer
                int(l2[0, 2 * i + 0]) + 4096,  # Layer 2, first sub-frame
                int(l3[0, 4 * i + 0]) + 2 * 4096,  # Layer 3, sub-frames
                int(l3[0, 4 * i + 1]) + 3 * 4096,
                int(l2[0, 2 * i + 1]) + 4 * 4096,  # Layer 2, second sub-frame
                int(l3[0, 4 * i + 2]) + 5 * 4096,  # Layer 3, remaining sub-frames
                int(l3[0, 4 * i + 3]) + 6 * 4096,
            ]
            packed.extend(frame_codes)
        except IndexError as e:
            logger.error(f"Index error during packing at frame {i}: {e}")
            break

    # Add AUDIO_BASE offset for language model vocabulary
    return [AUDIO_BASE + code for code in packed]


def process_audio_batch(
    audio_paths: List[str],
    texts: List[str],
    snac_model: SNAC,
    device: torch.device,
    batch_size: int = 1,
) -> List[Dict]:
    """
    Process a batch of audio files through SNAC encoder.

    Args:
        audio_paths: List of audio file paths
        texts: Corresponding transcript texts
        snac_model: SNAC encoder model
        device: Torch device (cuda/cpu)
        batch_size: Processing batch size

    Returns:
        List of dictionaries with text and codes
    """
    results = []

    for audio_path, text in zip(audio_paths, texts):
        # Load and validate audio
        audio, sr = validate_audio_file(audio_path)
        if audio is None:
            logger.warning(f"Skipping invalid audio: {audio_path}")
            continue

        # Convert to tensor and add batch/channel dimensions
        audio_tensor = (
            torch.tensor(audio, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # [1, 1, T]

        # Encode with SNAC
        try:
            with torch.no_grad():
                codes = snac_model.encode(audio_tensor)

            # Pack codes for LM training
            packed_codes = pack_codes(codes)

            results.append(
                {
                    "text": text.strip(),
                    "codes": packed_codes,
                    "audio_path": audio_path,  # Keep for debugging
                    "duration": len(audio) / sr,
                }
            )

        except Exception as e:
            logger.error(f"Failed to encode {audio_path}: {e}")
            continue

    return results


def main():
    """Main entry point for SNAC preprocessing."""
    parser = argparse.ArgumentParser(
        description="Precompute SNAC codes for VyvoTTS training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process default metadata and create training file
  %(prog)s
  
  # Custom paths
  %(prog)s --metadata custom.csv --output custom_snac.jsonl
  
  # GPU processing with larger batch
  %(prog)s --device cuda --batch-size 8
  
  # Limit processing for testing
  %(prog)s --max-samples 100
        """,
    )

    # Input/Output paths
    parser.add_argument(
        "--wav-root",
        default="../data/wavs",
        help="Root directory for WAV files (default: ../data/wavs)",
    )
    parser.add_argument(
        "--metadata",
        default="../data/metadata.csv",
        help="Input metadata CSV file (default: ../data/metadata.csv)",
    )
    parser.add_argument(
        "--output",
        default="../data/train_snac.jsonl",
        help="Output JSONL file with SNAC codes (default: ../data/train_snac.jsonl)",
    )

    # Processing options
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for SNAC encoding (default: auto-detect)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save checkpoint every N samples (default: 100)",
    )

    args = parser.parse_args()

    # Validate paths
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        sys.exit(1)

    wav_root = Path(args.wav_root)
    if not wav_root.exists():
        logger.error(f"WAV directory not found: {wav_root}")
        sys.exit(1)

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Load SNAC model
    logger.info(f"Loading SNAC model: {SNAC_MODEL_ID}")
    try:
        snac = SNAC.from_pretrained(SNAC_MODEL_ID).to(device).eval()
    except Exception as e:
        logger.error(f"Failed to load SNAC model: {e}")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process metadata
    audio_paths = []
    texts = []

    logger.info(f"Reading metadata from {metadata_path}")
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            for row_num, row in enumerate(reader, 1):
                if args.max_samples and row_num > args.max_samples:
                    break

                if not row or len(row) < 2:
                    logger.warning(f"Skipping invalid row {row_num}")
                    continue

                filename = row[0].strip()
                text = row[1].strip()

                if not text:
                    logger.warning(f"Skipping row {row_num}: empty text")
                    continue

                audio_path = wav_root / filename
                if not audio_path.exists():
                    logger.warning(f"Audio file not found: {audio_path}")
                    continue

                audio_paths.append(str(audio_path))
                texts.append(text)

    except Exception as e:
        logger.error(f"Error reading metadata: {e}")
        sys.exit(1)

    if not audio_paths:
        logger.error("No valid audio files found")
        sys.exit(1)

    logger.info(f"Found {len(audio_paths)} valid audio files")

    # Process audio files
    logger.info("Starting SNAC encoding...")
    all_results = []
    failed_count = 0

    # Process in batches (though SNAC typically processes one at a time)
    for i in range(0, len(audio_paths), args.batch_size):
        batch_paths = audio_paths[i : i + args.batch_size]
        batch_texts = texts[i : i + args.batch_size]

        results = process_audio_batch(
            batch_paths, batch_texts, snac, device, args.batch_size
        )

        all_results.extend(results)
        failed_count += len(batch_paths) - len(results)

        # Progress update
        if (i + args.batch_size) % args.checkpoint_every == 0:
            logger.info(
                f"Processed {min(i + args.batch_size, len(audio_paths))}/{len(audio_paths)} files. "
                f"Failed: {failed_count}"
            )

            # Save checkpoint
            checkpoint_path = output_path.with_suffix(".checkpoint.jsonl")
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                for result in all_results:
                    # Don't save audio_path and duration in final output
                    output_dict = {"text": result["text"], "codes": result["codes"]}
                    f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
            logger.info(f"Checkpoint saved to {checkpoint_path}")

    # Write final output
    logger.info(f"Writing final output to {output_path}")
    total_duration = 0.0

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for result in all_results:
                # Final output only needs text and codes
                output_dict = {"text": result["text"], "codes": result["codes"]}
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
                total_duration += result.get("duration", 0)

    except Exception as e:
        logger.error(f"Failed to write output: {e}")
        sys.exit(1)

    # Print summary
    logger.info("=" * 60)
    logger.info(f"✅ SNAC preprocessing complete!")
    logger.info(f"   Successful: {len(all_results)} files")
    logger.info(f"   Failed: {failed_count} files")
    logger.info(
        f"   Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)"
    )
    logger.info(f"   Output saved to: {output_path}")
    logger.info("=" * 60)

    # Cleanup checkpoint if everything succeeded
    checkpoint_path = output_path.with_suffix(".checkpoint.jsonl")
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Cleaned up checkpoint file")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
