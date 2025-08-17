#!/usr/bin/env python3
"""
Audio Device Listing Utility.

This module provides a simple utility to list all available audio input and output
devices on the system. Useful for configuring the recording setup and troubleshooting
audio device issues.

Features:
    - Lists all available audio devices with detailed information
    - Shows default input/output devices
    - Displays device capabilities (channels, sample rates)
    - Suggests suitable devices for recording

Usage:
    $ uv run list_audio_devices.py
    $ python list_audio_devices.py

Requirements:
    - sounddevice: Audio device querying
    - Linux: libportaudio2 (apt-get install libportaudio2)

Author: TTS Training Pipeline
License: MIT
"""

import sys
from typing import Dict, List, Optional

try:
    import sounddevice as sd
except ImportError:
    print("Error: sounddevice not installed. Please run: uv add sounddevice")
    sys.exit(1)


def get_device_info() -> Dict:
    """
    Retrieve information about all audio devices.

    Returns:
        Dictionary containing device information and defaults
    """
    try:
        devices = sd.query_devices()
        defaults = sd.default.device
        hostapis = sd.query_hostapis()

        return {"devices": devices, "defaults": defaults, "hostapis": hostapis}
    except Exception as e:
        print(f"Error querying audio devices: {e}")
        return None


def format_device_entry(device_id: int, device: Dict) -> str:
    """
    Format a single device entry for display.

    Args:
        device_id: Numeric ID of the device
        device: Device information dictionary

    Returns:
        Formatted string representation of the device
    """
    channels_in = device.get("max_input_channels", 0)
    channels_out = device.get("max_output_channels", 0)
    sample_rate = int(device.get("default_samplerate", 0))
    hostapi_id = device.get("hostapi", -1)

    # Determine device type
    device_type = []
    if channels_in > 0:
        device_type.append(f"IN:{channels_in}ch")
    if channels_out > 0:
        device_type.append(f"OUT:{channels_out}ch")
    type_str = " | ".join(device_type) if device_type else "NONE"

    return (
        f"  [{device_id:3d}] {device['name']:<40} "
        f"[{type_str}] "
        f"@ {sample_rate}Hz "
        f"(API:{hostapi_id})"
    )


def print_device_list(devices: List[Dict], title: str, filter_func=None) -> None:
    """
    Print a filtered list of devices.

    Args:
        devices: List of all devices
        title: Section title to display
        filter_func: Optional function to filter devices
    """
    print(f"\n{title}")
    print("=" * 80)

    found_any = False
    for i, device in enumerate(devices):
        if filter_func and not filter_func(device):
            continue
        print(format_device_entry(i, device))
        found_any = True

    if not found_any:
        print("  No devices found in this category")


def suggest_recording_device(devices: List[Dict]) -> Optional[int]:
    """
    Suggest the best device for recording based on heuristics.

    Args:
        devices: List of all devices

    Returns:
        Device ID of suggested recording device, or None
    """
    candidates = []

    for i, device in enumerate(devices):
        if device.get("max_input_channels", 0) == 0:
            continue

        # Score based on various factors
        score = 0
        name_lower = device["name"].lower()

        # Prefer USB devices
        if "usb" in name_lower:
            score += 10

        # Prefer devices with "microphone" in name
        if "microphone" in name_lower or "mic" in name_lower:
            score += 5

        # Avoid built-in devices for better quality
        if "built-in" not in name_lower and "internal" not in name_lower:
            score += 3

        # Prefer mono or stereo
        channels = device.get("max_input_channels", 0)
        if channels in [1, 2]:
            score += 2

        # Check sample rate support
        if device.get("default_samplerate", 0) >= 24000:
            score += 1

        candidates.append((i, score, device))

    if not candidates:
        return None

    # Sort by score (highest first)
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def main():
    """Main entry point for the audio device listing utility."""
    print("\n" + "=" * 80)
    print(" " * 25 + "AUDIO DEVICE INFORMATION")
    print("=" * 80)

    # Get device information
    info = get_device_info()
    if not info:
        print("\nFailed to query audio devices. Please check your audio system.")
        sys.exit(1)

    devices = info["devices"]
    defaults = info["defaults"]
    hostapis = info["hostapis"]

    # Print default devices
    print("\n📌 DEFAULT DEVICES")
    print("=" * 80)
    print(
        f"  Input:  {defaults[0] if defaults[0] is not None else 'None (System Default)'}"
    )
    print(
        f"  Output: {defaults[1] if defaults[1] is not None else 'None (System Default)'}"
    )

    # Print host APIs
    print("\n🔧 HOST APIS")
    print("=" * 80)
    for api in hostapis:
        print(f"  [{api['index']}] {api['name']} - {api['device_count']} devices")

    # Print all devices
    print_device_list(devices, "📋 ALL DEVICES", filter_func=None)

    # Print input devices
    print_device_list(
        devices,
        "🎤 INPUT DEVICES (for recording)",
        filter_func=lambda d: d.get("max_input_channels", 0) > 0,
    )

    # Print output devices
    print_device_list(
        devices,
        "🔊 OUTPUT DEVICES (for playback)",
        filter_func=lambda d: d.get("max_output_channels", 0) > 0,
    )

    # Suggest best recording device
    suggested = suggest_recording_device(devices)
    if suggested is not None:
        print("\n💡 SUGGESTED RECORDING DEVICE")
        print("=" * 80)
        print(format_device_entry(suggested, devices[suggested]))
        print(f"\n  To use this device, add to your command:")
        print(f"  --device \"{devices[suggested]['name']}\"")
        print(f"  or")
        print(f"  --device {suggested}")

    # Print usage tips
    print("\n📝 USAGE TIPS")
    print("=" * 80)
    print("  1. Use device ID or partial name match with --device parameter")
    print("  2. For best quality, use an external USB microphone")
    print("  3. Ensure your device supports 24kHz sample rate for SNAC compatibility")
    print("  4. Test recording with different devices to find the best quality")

    # Check for common issues
    input_devices = [d for d in devices if d.get("max_input_channels", 0) > 0]
    if not input_devices:
        print("\n⚠️  WARNING: No input devices found!")
        print("  - Check microphone connection")
        print("  - Install audio drivers if needed")
        print("  - On Linux: sudo apt-get install libportaudio2")

    print("\n" + "=" * 80)
    print("✅ Device scan complete\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
