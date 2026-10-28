import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.deepgram_client import DeepgramSTTClient


@pytest.mark.asyncio
async def test_connect_registers_handlers():
    on_final = AsyncMock()
    on_interim = AsyncMock()
    mock_live = MagicMock()
    mock_live.register_handler = MagicMock()

    with patch("src.deepgram_client.Deepgram") as MockDG:
        mock_transcription = AsyncMock(return_value=mock_live)
        MockDG.return_value.transcription.live = mock_transcription
        client = DeepgramSTTClient("key", on_final, on_interim)
        await client.connect()
        assert mock_live.register_handler.call_count >= 1


@pytest.mark.asyncio
async def test_reconnect_returns_false_after_three_attempts():
    on_final = AsyncMock()
    on_interim = AsyncMock()
    with patch("src.deepgram_client.Deepgram") as MockDG:
        MockDG.return_value.transcription.live = AsyncMock(
            side_effect=Exception("connection refused")
        )
        client = DeepgramSTTClient("key", on_final, on_interim)
        result = await client.reconnect_with_backoff()
        assert result is False
