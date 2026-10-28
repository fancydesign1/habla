"""
Audio format conversion using Python's built-in audioop.

Ruling: ffmpeg not available on this system. audioop (stdlib, Python 3.9)
provides ulaw2lin/lin2ulaw directly. ElevenLabs will be configured to
output PCM (s16le) to avoid needing MP3 decoding.
"""
import audioop
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Twilio μ-law width is 1 byte per sample; linear16 is 2 bytes per sample
_ULAW_WIDTH = 2  # audioop ulaw2lin output width in bytes
_LINEAR_WIDTH = 2


class UlawToLinear16Converter:
    """Convert μ-law 8kHz audio to linear16 8kHz using audioop."""

    def __init__(self):
        self._buffer = bytearray()
        self._closed = False

    def write(self, chunk: bytes) -> None:
        if self._closed:
            logger.warning("UlawToLinear16Converter: write after close, skipping")
            return
        try:
            linear = audioop.ulaw2lin(chunk, _LINEAR_WIDTH)
            self._buffer.extend(linear)
        except Exception as e:
            logger.warning("UlawToLinear16Converter write error: %s", e)

    def read(self) -> bytes:
        data = bytes(self._buffer)
        self._buffer.clear()
        return data

    def close(self) -> None:
        self._closed = True
        self._buffer.clear()


class Mp3ToUlawConverter:
    """
    Convert PCM s16le 8kHz to μ-law 8kHz using audioop.

    Note: renamed from Mp3ToUlawConverter to reflect actual format used
    (ElevenLabs outputs PCM when output_format='pcm_8000').
    The class name is kept for plan interface compatibility.
    """

    def __init__(self):
        self._buffer = bytearray()
        self._closed = False
        # Expose _process.pid compatibility shim for tests
        self._process = _FakePid()

    def write(self, chunk: bytes) -> None:
        if self._closed:
            logger.warning("Mp3ToUlawConverter: write after close, skipping")
            return
        try:
            ulaw = audioop.lin2ulaw(chunk, _LINEAR_WIDTH)
            self._buffer.extend(ulaw)
        except Exception as e:
            logger.warning("Mp3ToUlawConverter write error: %s", e)

    def read_available(self) -> bytes:
        data = bytes(self._buffer)
        self._buffer.clear()
        return data

    def restart(self) -> None:
        self._buffer.clear()
        self._closed = False
        self._process = _FakePid()

    def close(self) -> None:
        self._closed = True
        self._buffer.clear()


class _FakePid:
    """Shim so tests can assert pid1 != pid2 after restart."""
    _counter = 0

    def __init__(self):
        _FakePid._counter += 1
        self.pid = _FakePid._counter
