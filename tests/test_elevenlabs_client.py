import pytest
from unittest.mock import MagicMock, patch
from src.elevenlabs_client import ElevenLabsClient, ElevenLabsError


@pytest.mark.asyncio
async def test_synthesize_yields_bytes():
    with patch("src.elevenlabs_client.AsyncElevenLabs") as MockEL:
        async def fake_stream(*a, **kw):
            yield b"mp3chunk1"
            yield b"mp3chunk2"
        MockEL.return_value.text_to_speech.convert_as_stream = MagicMock(
            return_value=fake_stream()
        )
        client = ElevenLabsClient("key", "voice_id")
        chunks = [chunk async for chunk in client.synthesize("Hola")]
        assert len(chunks) == 2
        assert chunks[0] == b"mp3chunk1"


@pytest.mark.asyncio
async def test_abort_stops_synthesis():
    with patch("src.elevenlabs_client.AsyncElevenLabs") as MockEL:
        collected = []

        async def fake_stream(*a, **kw):
            yield b"chunk1"
            yield b"chunk2"
            yield b"chunk3"

        MockEL.return_value.text_to_speech.convert_as_stream = MagicMock(
            return_value=fake_stream()
        )
        client = ElevenLabsClient("key", "voice_id")
        async for chunk in client.synthesize("Hola"):
            collected.append(chunk)
            if len(collected) == 1:
                client.abort()
        assert len(collected) == 1


@pytest.mark.asyncio
async def test_synthesize_raises_elevenlabs_error_on_exception():
    with patch("src.elevenlabs_client.AsyncElevenLabs") as MockEL:
        async def bad_stream(*a, **kw):
            raise RuntimeError("stream failed")
            yield  # make it a generator

        MockEL.return_value.text_to_speech.convert_as_stream = MagicMock(
            return_value=bad_stream()
        )
        client = ElevenLabsClient("key", "voice_id")
        with pytest.raises(ElevenLabsError):
            async for _ in client.synthesize("Hola"):
                pass
