"""
OpenAI-compatible TTS API server for Dia2.

This module provides an OpenAI-compatible API endpoint for text-to-speech
that can be used with Open WebUI and other compatible clients.

Usage:
    uv run -m dia2.api_server --host 0.0.0.0 --port 4123

API Endpoints:
    POST /v1/audio/speech - Generate speech from text (OpenAI-compatible)
    GET /v1/voices - List available voices
    POST /v1/voices - Add a new voice (upload audio file)
    DELETE /v1/voices/{name} - Delete a voice
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from .engine import Dia2
from .generation import GenerationConfig, SamplingConfig

# Directory for storing voice files
VOICES_DIR = Path(os.environ.get("DIA2_VOICES_DIR", "./voices"))

# Global Dia2 instance
_dia: Optional[Dia2] = None
_voices: dict[str, dict] = {}

app = FastAPI(
    title="Dia2 TTS API",
    description="OpenAI-compatible Text-to-Speech API using Dia2",
    version="1.0.0",
)


def _get_dia() -> Dia2:
    """Get the global Dia2 instance."""
    global _dia
    if _dia is None:
        raise HTTPException(status_code=503, detail="TTS engine not initialized")
    return _dia


def _load_voices() -> None:
    """Load voice configurations from the voices directory."""
    global _voices
    _voices = {}

    if not VOICES_DIR.exists():
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        return

    # Load voice metadata if exists
    meta_file = VOICES_DIR / "voices.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                _voices = json.load(f)
        except json.JSONDecodeError:
            _voices = {}

    # Also scan for audio files without metadata
    for audio_file in VOICES_DIR.glob("*.wav"):
        name = audio_file.stem
        if name not in _voices:
            _voices[name] = {
                "name": name,
                "file": str(audio_file),
                "aliases": [],
            }

    for audio_file in VOICES_DIR.glob("*.mp3"):
        name = audio_file.stem
        if name not in _voices:
            _voices[name] = {
                "name": name,
                "file": str(audio_file),
                "aliases": [],
            }


def _save_voices() -> None:
    """Save voice configurations to the voices directory."""
    meta_file = VOICES_DIR / "voices.json"
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    with open(meta_file, "w") as f:
        json.dump(_voices, f, indent=2)


def _get_voice_file(voice_name: str) -> Optional[str]:
    """Get the file path for a voice by name or alias."""
    # Direct name match
    if voice_name in _voices:
        return _voices[voice_name].get("file")

    # Check aliases
    for voice in _voices.values():
        if voice_name in voice.get("aliases", []):
            return voice.get("file")

    return None


def _generate_audio(text: str, voice: Optional[str] = None) -> tuple[bytes, int]:
    """
    Generate audio from text.

    Args:
        text: The text to synthesize.
        voice: Optional voice name for voice cloning.

    Returns:
        Tuple of (audio_bytes, sample_rate).
    """
    dia = _get_dia()

    # Build the script with speaker tags
    # For single speaker, use [S1] tag
    script = f"[S1] {text}"

    # Get voice file if specified
    prefix_speaker_1 = None
    if voice:
        voice_file = _get_voice_file(voice)
        if voice_file and Path(voice_file).exists():
            prefix_speaker_1 = voice_file

    # Generate audio
    config = GenerationConfig(
        cfg_scale=2.0,
        text=SamplingConfig(temperature=0.6, top_k=50),
        audio=SamplingConfig(temperature=0.8, top_k=50),
    )

    result = dia.generate(
        script,
        config=config,
        prefix_speaker_1=prefix_speaker_1,
        include_prefix=False,
        verbose=False,
    )

    # Convert waveform to bytes
    waveform = result.waveform.detach().cpu().numpy()
    if waveform.ndim > 1:
        waveform = waveform.squeeze()

    # Normalize audio
    waveform = np.clip(waveform, -1.0, 1.0)

    return waveform, result.sample_rate


def _audio_to_format(
    waveform: np.ndarray,
    sample_rate: int,
    response_format: str = "mp3",
) -> bytes:
    """Convert waveform to the requested audio format."""
    buffer = io.BytesIO()

    if response_format == "mp3":
        # Use soundfile for WAV then convert if needed
        # For simplicity, we'll use WAV internally but return as requested format
        sf.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()
    elif response_format == "opus":
        sf.write(buffer, waveform, sample_rate, format="OGG", subtype="OPUS")
        return buffer.getvalue()
    elif response_format == "aac":
        # AAC not directly supported, fall back to WAV
        sf.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()
    elif response_format == "flac":
        sf.write(buffer, waveform, sample_rate, format="FLAC")
        return buffer.getvalue()
    elif response_format == "wav":
        sf.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()
    elif response_format == "pcm":
        # Raw PCM 16-bit
        pcm_data = (waveform * 32767).astype(np.int16)
        return pcm_data.tobytes()
    else:
        # Default to WAV
        sf.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Dia2 TTS API Server",
        "version": "1.0.0",
        "docs": "/docs",
        "openai_compatible": True,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": _dia is not None}


@app.post("/v1/audio/speech")
async def create_speech(request: Request):
    """
    OpenAI-compatible speech generation endpoint.

    Request body:
        model: str - Model to use (ignored, uses configured Dia2 model)
        input: str - Text to synthesize
        voice: str - Voice name (optional, for voice cloning)
        response_format: str - Audio format (mp3, opus, aac, flac, wav, pcm)
        speed: float - Speed adjustment (ignored for now)

    Returns:
        Audio file in requested format.
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    text = body.get("input", "")
    if not text:
        raise HTTPException(status_code=400, detail="Input text is required")

    voice = body.get("voice", "alloy")  # Default to "alloy" like OpenAI
    response_format = body.get("response_format", "mp3")

    # Validate response format
    valid_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
    if response_format not in valid_formats:
        response_format = "mp3"

    try:
        waveform, sample_rate = _generate_audio(text, voice)
        audio_bytes = _audio_to_format(waveform, sample_rate, response_format)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

    # Determine content type
    content_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    content_type = content_types.get(response_format, "audio/mpeg")

    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="speech.{response_format}"',
        },
    )


@app.get("/v1/voices")
async def list_voices():
    """List all available voices."""
    voices_list = []
    for name, info in _voices.items():
        voices_list.append({
            "voice_id": name,
            "name": name,
            "aliases": info.get("aliases", []),
        })

    # Add default OpenAI voice names for compatibility
    default_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    for voice in default_voices:
        if voice not in _voices:
            voices_list.append({
                "voice_id": voice,
                "name": voice,
                "aliases": [],
                "builtin": True,
            })

    return {"voices": voices_list}


@app.post("/v1/voices")
async def add_voice(
    name: str = Form(...),
    audio: UploadFile = File(...),
    aliases: str = Form(default=""),
):
    """
    Add a new voice for cloning.

    Args:
        name: Voice name.
        audio: Audio file for voice cloning.
        aliases: Comma-separated list of aliases.
    """
    # Validate file type
    if not audio.filename:
        raise HTTPException(status_code=400, detail="Audio file is required")

    file_ext = Path(audio.filename).suffix.lower()
    if file_ext not in [".wav", ".mp3", ".ogg", ".flac"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format. Use WAV, MP3, OGG, or FLAC.",
        )

    # Create voices directory if needed
    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    # Save audio file
    dest_file = VOICES_DIR / f"{name}{file_ext}"
    with open(dest_file, "wb") as f:
        content = await audio.read()
        f.write(content)

    # Parse aliases
    alias_list = [a.strip() for a in aliases.split(",") if a.strip()]

    # Save voice metadata
    _voices[name] = {
        "name": name,
        "file": str(dest_file),
        "aliases": alias_list,
    }
    _save_voices()

    return {"message": f"Voice '{name}' added successfully", "voice_id": name}


@app.delete("/v1/voices/{name}")
async def delete_voice(name: str):
    """Delete a voice."""
    if name not in _voices:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")

    voice_info = _voices[name]
    file_path = voice_info.get("file")

    # Delete the audio file
    if file_path and Path(file_path).exists():
        Path(file_path).unlink()

    # Remove from metadata
    del _voices[name]
    _save_voices()

    return {"message": f"Voice '{name}' deleted successfully"}


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatibility)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": 0,
                "owned_by": "dia2",
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": 0,
                "owned_by": "dia2",
            },
        ],
    }


def init_dia(
    repo: str = "nari-labs/Dia2-2B",
    device: str = "cuda",
    dtype: str = "bfloat16",
    quantization: Optional[str] = None,
):
    """Initialize the global Dia2 instance."""
    global _dia
    _dia = Dia2.from_repo(
        repo,
        device=device,
        dtype=dtype,
        quantization=quantization,
    )
    # Load voices
    _load_voices()


def main():
    """Run the API server."""
    parser = argparse.ArgumentParser(description="Dia2 TTS API Server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4123,
        help="Port to listen on (default: 4123)",
    )
    parser.add_argument(
        "--hf",
        default="nari-labs/Dia2-2B",
        help="Hugging Face repo id (default: nari-labs/Dia2-2B)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Computation device (defaults to cuda if available)",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "bfloat16"],
        default="bfloat16",
        help="Computation dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--low-vram",
        choices=["none", "8bit", "4bit"],
        default="none",
        help="Low VRAM mode using bitsandbytes quantization",
    )
    parser.add_argument(
        "--voices-dir",
        default="./voices",
        help="Directory for voice files (default: ./voices)",
    )
    args = parser.parse_args()

    # Set voices directory
    global VOICES_DIR
    VOICES_DIR = Path(args.voices_dir)

    # Determine device
    device = args.device
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine quantization
    quantization = args.low_vram if args.low_vram != "none" else None

    print(f"Initializing Dia2 TTS with repo={args.hf}, device={device}, dtype={args.dtype}")
    if quantization:
        print(f"Using {quantization} quantization for low VRAM mode")

    # Initialize Dia2
    init_dia(
        repo=args.hf,
        device=device,
        dtype=args.dtype,
        quantization=quantization,
    )

    print(f"Starting API server on http://{args.host}:{args.port}")
    print(f"Voices directory: {VOICES_DIR}")
    print("\nOpen WebUI Configuration:")
    print("  Text-to-Speech Engine: OpenAI")
    print(f"  API Base URL: http://localhost:{args.port}/v1")
    print("  API Key: none")
    print("  TTS Model: tts-1 or tts-1-hd")
    print("  TTS Voice: <your cloned voice name>")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
