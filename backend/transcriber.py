#!/usr/bin/env python3
"""
Local Transcription using Faster-Whisper
All audio stays on your server - no third-party APIs.
"""

import asyncio
import io
import tempfile
import wave
from pathlib import Path
from typing import Callable, Optional
import numpy as np

# Faster-Whisper for local transcription
from faster_whisper import WhisperModel

class LocalTranscriber:
    """Real-time transcription using Faster-Whisper."""

    def __init__(
        self,
        model_size: str = "base",  # tiny, base, small, medium, large-v3
        device: str = "auto",       # auto, cpu, cuda
        compute_type: str = "auto"  # auto, int8, float16, float32
    ):
        """
        Initialize the transcriber.

        Model sizes and approximate requirements:
        - tiny:    ~1GB RAM, fastest, lower accuracy
        - base:    ~1GB RAM, good balance for real-time
        - small:   ~2GB RAM, better accuracy
        - medium:  ~5GB RAM, high accuracy
        - large-v3: ~10GB RAM, best accuracy (needs GPU)
        """
        print(f"Loading Whisper model '{model_size}'...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        print(f"Model loaded on {self.model.device}")

        # Audio buffer for streaming
        self.audio_buffer = bytearray()
        self.sample_rate = 16000
        self.chunk_duration = 3  # Process every 3 seconds of audio
        self.min_chunk_size = self.sample_rate * 2 * self.chunk_duration  # 16-bit = 2 bytes

    async def process_audio_chunk(self, audio_data: bytes) -> Optional[str]:
        """
        Add audio data to buffer and transcribe if enough accumulated.
        Returns transcription or None if still buffering.
        """
        self.audio_buffer.extend(audio_data)

        # Check if we have enough audio to process
        if len(self.audio_buffer) < self.min_chunk_size:
            return None

        # Extract chunk for processing
        chunk = bytes(self.audio_buffer[:self.min_chunk_size])
        self.audio_buffer = self.audio_buffer[self.min_chunk_size:]

        # Transcribe
        return await self._transcribe_bytes(chunk)

    async def _transcribe_bytes(self, audio_bytes: bytes) -> str:
        """Transcribe raw audio bytes."""
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Run transcription in thread pool to not block async
        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None,
            lambda: self.model.transcribe(
                audio_array,
                language="en",
                vad_filter=True,  # Filter out silence
                vad_parameters=dict(min_silence_duration_ms=500)
            )
        )

        # Combine segments
        text = " ".join([segment.text.strip() for segment in segments])
        return text.strip()

    async def transcribe_file(self, file_path: str) -> str:
        """Transcribe an audio file."""
        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None,
            lambda: self.model.transcribe(file_path, language="en")
        )
        return " ".join([segment.text.strip() for segment in segments])

    def flush_buffer(self) -> bytes:
        """Get remaining audio in buffer and clear it."""
        remaining = bytes(self.audio_buffer)
        self.audio_buffer = bytearray()
        return remaining

    async def finalize(self) -> Optional[str]:
        """Transcribe any remaining audio in buffer."""
        remaining = self.flush_buffer()
        if len(remaining) > self.sample_rate:  # At least 0.5 second
            return await self._transcribe_bytes(remaining)
        return None


class AudioConverter:
    """Convert various audio formats to raw PCM for Whisper."""

    @staticmethod
    async def webm_to_pcm(webm_data: bytes) -> bytes:
        """
        Convert WebM/Opus audio to raw PCM.
        Uses ffmpeg for conversion.
        """
        import subprocess

        # Write WebM to temp file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
            f.write(webm_data)
            webm_path = f.name

        pcm_path = webm_path.replace('.webm', '.pcm')

        try:
            # Convert using ffmpeg
            process = await asyncio.create_subprocess_exec(
                'ffmpeg', '-y', '-i', webm_path,
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',      # Mono
                '-f', 's16le',   # Raw PCM 16-bit little-endian
                pcm_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.wait()

            # Read PCM data
            with open(pcm_path, 'rb') as f:
                pcm_data = f.read()

            return pcm_data

        finally:
            # Cleanup
            Path(webm_path).unlink(missing_ok=True)
            Path(pcm_path).unlink(missing_ok=True)

    @staticmethod
    def is_webm(data: bytes) -> bool:
        """Check if data is WebM format."""
        return data[:4] == b'\x1a\x45\xdf\xa3'


# Singleton instance
_transcriber: Optional[LocalTranscriber] = None

def get_transcriber(model_size: str = "base") -> LocalTranscriber:
    """Get or create the transcriber instance."""
    global _transcriber
    if _transcriber is None:
        _transcriber = LocalTranscriber(model_size=model_size)
    return _transcriber
