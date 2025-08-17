#!/usr/bin/env python3
"""
NVIDIA NeMo TTS Training Script.

This module provides training functionality for single-speaker Text-to-Speech models
using NVIDIA NeMo framework. It supports training both FastPitch (acoustic model)
and HiFi-GAN (vocoder) models.

The training pipeline consists of:
    1. Manifest preparation from LJSpeech-format metadata
    2. FastPitch training for text-to-mel-spectrogram conversion
    3. HiFi-GAN training for mel-spectrogram-to-audio conversion

Features:
    - Single-speaker TTS training optimized for 8GB GPU
    - Automatic manifest generation from metadata.csv
    - Support for 24kHz audio (SNAC-compatible)
    - Checkpoint saving and resumption

Usage:
    $ uv run nemo_train.py --prepare-manifest
    $ uv run nemo_train.py --train-fastpitch
    $ uv run nemo_train.py --train-hifigan
    $ uv run nemo_train.py --prepare-manifest --train-fastpitch  # Combined

Requirements:
    - nemo-toolkit: NVIDIA NeMo framework
    - pytorch-lightning: Training framework
    - omegaconf: Configuration management
    - soundfile: Audio file I/O
    - torch, torchaudio: Deep learning backend

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

try:
    import soundfile as sf
except ImportError:
    print("Error: soundfile not installed. Please run: uv add soundfile")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_audio_file(wav_path: str) -> Optional[Dict]:
    """
    Validate an audio file and extract metadata.

    Args:
        wav_path: Path to the WAV file

    Returns:
        Dictionary with audio metadata or None if invalid
    """
    try:
        info = sf.info(wav_path)

        # Validate audio properties
        if info.samplerate != 24000:
            logger.warning(
                f"Non-standard sample rate {info.samplerate}Hz in {wav_path} "
                f"(expected 24000Hz)"
            )

        if info.channels != 1:
            logger.warning(f"Non-mono audio ({info.channels} channels) in {wav_path}")

        return {
            "duration": float(info.duration),
            "samplerate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
        }

    except Exception as e:
        logger.error(f"Failed to read audio file {wav_path}: {e}")
        return None


def build_manifest(
    metadata_csv: str, out_jsonl: str, wav_root: str, validate: bool = True
) -> Tuple[int, int]:
    """
    Build NeMo training manifest from LJSpeech-format metadata.

    The manifest is a JSONL file where each line contains:
        - audio_filepath: Path to the audio file
        - text: Transcript text
        - duration: Audio duration in seconds

    Args:
        metadata_csv: Path to metadata CSV file (filename|transcript format)
        out_jsonl: Output path for the JSONL manifest
        wav_root: Root directory containing WAV files
        validate: Whether to validate audio files

    Returns:
        Tuple of (successful_count, failed_count)

    Raises:
        FileNotFoundError: If metadata CSV doesn't exist
    """
    metadata_path = Path(metadata_csv)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")

    wav_root_path = Path(wav_root)
    if not wav_root_path.exists():
        raise FileNotFoundError(f"WAV directory not found: {wav_root}")

    # Create output directory
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0
    total_duration = 0.0

    logger.info(f"Building manifest from {metadata_csv}")
    logger.info(f"WAV root directory: {wav_root}")

    try:
        with (
            open(metadata_csv, "r", newline="", encoding="utf-8") as f_in,
            open(out_jsonl, "w", encoding="utf-8") as f_out,
        ):

            reader = csv.reader(f_in, delimiter="|")

            for row_num, row in enumerate(reader, 1):
                if not row or len(row) < 2:
                    logger.warning(f"Skipping invalid row {row_num}: {row}")
                    failed += 1
                    continue

                filename = row[0].strip()
                text = row[1].strip()

                if not text:
                    logger.warning(f"Skipping row {row_num}: empty transcript")
                    failed += 1
                    continue

                # Build audio path
                audio_path = os.path.join(wav_root, filename)

                if not os.path.exists(audio_path):
                    logger.error(f"Audio file not found: {audio_path}")
                    failed += 1
                    continue

                # Validate audio if requested
                if validate:
                    audio_info = validate_audio_file(audio_path)
                    if audio_info is None:
                        failed += 1
                        continue
                    duration = audio_info["duration"]
                else:
                    # Quick duration calculation without validation
                    try:
                        info = sf.info(audio_path)
                        duration = float(info.duration)
                    except Exception as e:
                        logger.error(f"Failed to get duration for {audio_path}: {e}")
                        failed += 1
                        continue

                # Create manifest entry
                entry = {
                    "audio_filepath": audio_path,
                    "text": text,
                    "duration": duration,
                }

                # Write to manifest
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                successful += 1
                total_duration += duration

                if successful % 100 == 0:
                    logger.info(f"Processed {successful} entries...")

    except Exception as e:
        logger.error(f"Error building manifest: {e}")
        raise

    # Print summary
    logger.info("=" * 60)
    logger.info(f"✅ Manifest created: {out_jsonl}")
    logger.info(f"   Successful entries: {successful}")
    logger.info(f"   Failed entries: {failed}")
    logger.info(
        f"   Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)"
    )
    logger.info(
        f"   Average duration: {total_duration/successful:.2f} seconds"
        if successful > 0
        else ""
    )
    logger.info("=" * 60)

    return successful, failed


def check_gpu_memory() -> Optional[float]:
    """
    Check available GPU memory.

    Returns:
        Available GPU memory in GB, or None if no GPU
    """
    try:
        import torch

        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            total_memory = device_props.total_memory / (1024**3)  # Convert to GB
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            available = total_memory - allocated

            logger.info(f"GPU: {device_props.name}")
            logger.info(f"Total memory: {total_memory:.2f} GB")
            logger.info(f"Available memory: {available:.2f} GB")

            return available
        else:
            logger.warning("No GPU detected, training will be slow")
            return None
    except ImportError:
        logger.warning("PyTorch not installed, cannot check GPU")
        return None


def validate_config_file(config_path: str) -> bool:
    """
    Validate that a configuration file exists and is readable.

    Args:
        config_path: Path to the configuration file

    Returns:
        True if config is valid, False otherwise
    """
    config_file = Path(config_path)

    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return False

    if not config_file.suffix in [".yaml", ".yml"]:
        logger.warning(f"Unexpected config file extension: {config_file.suffix}")

    try:
        # Try to load the config to validate syntax
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(config_path)
        logger.info(f"Configuration loaded successfully: {config_path}")
        return True
    except ImportError:
        logger.warning("OmegaConf not installed, skipping config validation")
        return True  # Assume valid if we can't check
    except Exception as e:
        logger.error(f"Invalid configuration file: {e}")
        return False


def train_fastpitch(cfg_path: str, resume_from: Optional[str] = None) -> None:
    """
    Train FastPitch acoustic model.

    FastPitch is a fully-parallel text-to-mel-spectrogram model based on
    Transformer architecture. It generates mel-spectrograms from text with
    explicit duration prediction.

    Args:
        cfg_path: Path to the training configuration file
        resume_from: Optional checkpoint path to resume training

    Raises:
        ImportError: If NeMo is not installed
        FileNotFoundError: If config file doesn't exist
    """
    if not validate_config_file(cfg_path):
        raise FileNotFoundError(f"Invalid configuration: {cfg_path}")

    # Check GPU memory
    gpu_memory = check_gpu_memory()
    if gpu_memory and gpu_memory < 6.0:
        logger.warning(
            f"Low GPU memory ({gpu_memory:.2f} GB). "
            f"Consider reducing batch size in config."
        )

    try:
        from omegaconf import OmegaConf
        from nemo.collections.tts.models import FastPitchModel
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        from pytorch_lightning.loggers import TensorBoardLogger
    except ImportError as e:
        logger.error(f"Missing required dependencies: {e}")
        logger.error("Please install: uv add nemo-toolkit pytorch-lightning")
        raise

    logger.info("=" * 60)
    logger.info("Starting FastPitch training")
    logger.info(f"Configuration: {cfg_path}")

    # Load configuration
    cfg = OmegaConf.load(cfg_path)

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints/fastpitch")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Setup callbacks
    callbacks = []

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="fastpitch-{epoch:02d}-{val_loss:.3f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback (optional)
    if cfg.get("early_stopping", False):
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.001, patience=10, verbose=True, mode="min"
        )
        callbacks.append(early_stop_callback)

    # TensorBoard logger
    tb_logger = TensorBoardLogger(save_dir="logs", name="fastpitch", version=None)

    # Create trainer with callbacks
    trainer_config = cfg.trainer.copy()
    trainer_config["callbacks"] = callbacks
    trainer_config["logger"] = tb_logger

    trainer = Trainer(**trainer_config)

    # Create or restore model
    if resume_from and Path(resume_from).exists():
        logger.info(f"Resuming training from checkpoint: {resume_from}")
        model = FastPitchModel.restore_from(resume_from, trainer=trainer)
    else:
        logger.info("Creating new FastPitch model")
        model = FastPitchModel(cfg.model, trainer=trainer)

    # Start training
    logger.info("Starting training...")
    trainer.fit(model)

    # Save final model
    output_path = checkpoint_dir / "fastpitch_final.nemo"
    model.save_to(str(output_path))

    logger.info("=" * 60)
    logger.info(f"✅ Training complete!")
    logger.info(f"   Final model saved to: {output_path}")
    logger.info(f"   Checkpoints saved to: {checkpoint_dir}")
    logger.info(f"   TensorBoard logs: tensorboard --logdir logs/fastpitch")
    logger.info("=" * 60)


def train_hifigan(cfg_path: str, resume_from: Optional[str] = None) -> None:
    """
    Train HiFi-GAN vocoder model.

    HiFi-GAN is a GAN-based vocoder that converts mel-spectrograms to
    high-fidelity audio waveforms. It uses multi-period and multi-scale
    discriminators for improved audio quality.

    Args:
        cfg_path: Path to the training configuration file
        resume_from: Optional checkpoint path to resume training

    Raises:
        ImportError: If NeMo is not installed
        FileNotFoundError: If config file doesn't exist
    """
    if not validate_config_file(cfg_path):
        raise FileNotFoundError(f"Invalid configuration: {cfg_path}")

    # Check GPU memory
    gpu_memory = check_gpu_memory()
    if gpu_memory and gpu_memory < 6.0:
        logger.warning(
            f"Low GPU memory ({gpu_memory:.2f} GB). "
            f"Consider using the small model variant."
        )

    try:
        from omegaconf import OmegaConf
        from nemo.collections.tts.models import HifiGanModel
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint
        from pytorch_lightning.loggers import TensorBoardLogger
    except ImportError as e:
        logger.error(f"Missing required dependencies: {e}")
        logger.error("Please install: uv add nemo-toolkit pytorch-lightning")
        raise

    logger.info("=" * 60)
    logger.info("Starting HiFi-GAN training")
    logger.info(f"Configuration: {cfg_path}")

    # Load configuration
    cfg = OmegaConf.load(cfg_path)

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints/hifigan")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Setup callbacks
    callbacks = []

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="hifigan-{epoch:02d}-{val_loss:.3f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
        every_n_epochs=5,  # Save less frequently for vocoder
    )
    callbacks.append(checkpoint_callback)

    # TensorBoard logger
    tb_logger = TensorBoardLogger(save_dir="logs", name="hifigan", version=None)

    # Create trainer
    trainer_config = cfg.trainer.copy()
    trainer_config["callbacks"] = callbacks
    trainer_config["logger"] = tb_logger

    trainer = Trainer(**trainer_config)

    # Create or restore model
    if resume_from and Path(resume_from).exists():
        logger.info(f"Resuming training from checkpoint: {resume_from}")
        model = HifiGanModel.restore_from(resume_from, trainer=trainer)
    else:
        logger.info("Creating new HiFi-GAN model")
        model = HifiGanModel(cfg.model, trainer=trainer)

    # Start training
    logger.info("Starting training...")
    logger.info("Note: Vocoder training typically takes longer than acoustic model")
    trainer.fit(model)

    # Save final model
    output_path = checkpoint_dir / "hifigan_final.nemo"
    model.save_to(str(output_path))

    logger.info("=" * 60)
    logger.info(f"✅ Training complete!")
    logger.info(f"   Final model saved to: {output_path}")
    logger.info(f"   Checkpoints saved to: {checkpoint_dir}")
    logger.info(f"   TensorBoard logs: tensorboard --logdir logs/hifigan")
    logger.info("=" * 60)


def main():
    """Main entry point for the NeMo training script."""
    parser = argparse.ArgumentParser(
        description="NVIDIA NeMo TTS Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare manifest from metadata
  %(prog)s --prepare-manifest
  
  # Train FastPitch acoustic model
  %(prog)s --train-fastpitch
  
  # Train HiFi-GAN vocoder
  %(prog)s --train-hifigan
  
  # Full pipeline
  %(prog)s --prepare-manifest --train-fastpitch --train-hifigan
  
  # Resume training from checkpoint
  %(prog)s --train-fastpitch --resume checkpoints/fastpitch/last.ckpt
        """,
    )

    # Actions
    parser.add_argument(
        "--prepare-manifest",
        action="store_true",
        help="Prepare training manifest from metadata CSV",
    )
    parser.add_argument(
        "--train-fastpitch", action="store_true", help="Train FastPitch acoustic model"
    )
    parser.add_argument(
        "--train-hifigan", action="store_true", help="Train HiFi-GAN vocoder"
    )

    # Data paths
    parser.add_argument(
        "--metadata",
        default="../data/metadata.csv",
        help="Path to metadata CSV file (default: ../data/metadata.csv)",
    )
    parser.add_argument(
        "--wav-root",
        default="../data/wavs",
        help="Root directory for WAV files (default: ../data/wavs)",
    )
    parser.add_argument(
        "--manifest",
        default="../data/manifest_train.json",
        help="Path to output manifest JSONL (default: ../data/manifest_train.json)",
    )

    # Configuration files
    parser.add_argument(
        "--fastpitch-cfg",
        default="../configs/fastpitch_single_speaker_8gb.yaml",
        help="FastPitch config file (default: ../configs/fastpitch_single_speaker_8gb.yaml)",
    )
    parser.add_argument(
        "--hifigan-cfg",
        default="../configs/hifigan_small_8gb.yaml",
        help="HiFi-GAN config file (default: ../configs/hifigan_small_8gb.yaml)",
    )

    # Training options
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--validate-audio",
        action="store_true",
        help="Validate audio files when building manifest",
    )

    args = parser.parse_args()

    # Check if any action was specified
    if not any([args.prepare_manifest, args.train_fastpitch, args.train_hifigan]):
        parser.print_help()
        print(
            "\n❌ Error: No action specified. Use --prepare-manifest, --train-fastpitch, or --train-hifigan"
        )
        sys.exit(1)

    try:
        # Prepare manifest if requested
        if args.prepare_manifest:
            logger.info("Preparing training manifest...")
            successful, failed = build_manifest(
                args.metadata,
                args.manifest,
                args.wav_root,
                validate=args.validate_audio,
            )

            if successful == 0:
                logger.error("No valid training data found!")
                sys.exit(1)

            if failed > successful * 0.1:  # More than 10% failed
                logger.warning(
                    f"High failure rate ({failed}/{successful + failed}). "
                    f"Please check your data."
                )

        # Train FastPitch if requested
        if args.train_fastpitch:
            logger.info("Starting FastPitch training...")

            # Check if manifest exists
            if not Path(args.manifest).exists() and not args.prepare_manifest:
                logger.error(
                    f"Manifest not found: {args.manifest}\n"
                    f"Run with --prepare-manifest first"
                )
                sys.exit(1)

            train_fastpitch(args.fastpitch_cfg, args.resume)

        # Train HiFi-GAN if requested
        if args.train_hifigan:
            logger.info("Starting HiFi-GAN training...")

            # Check if manifest exists
            if not Path(args.manifest).exists() and not args.prepare_manifest:
                logger.error(
                    f"Manifest not found: {args.manifest}\n"
                    f"Run with --prepare-manifest first"
                )
                sys.exit(1)

            train_hifigan(args.hifigan_cfg, args.resume)

        logger.info("\n✅ All requested operations completed successfully!")

    except KeyboardInterrupt:
        logger.info("\n⚠️ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
