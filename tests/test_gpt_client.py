import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.gpt_client import GPTClient, GPTResponse, GPTClientError

SYSTEM_PROMPT = "You are a Spanish assistant. Return JSON {reply_es, translation_uk}."


@pytest.mark.asyncio
async def test_get_response_returns_gpt_response():
    with patch("src.gpt_client.AsyncOpenAI") as MockOAI:
        mock_msg = MagicMock()
        mock_msg.content = '{"reply_es": "Hola", "translation_uk": "Привіт"}'
        MockOAI.return_value.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=mock_msg)])
        )
        client = GPTClient("key", SYSTEM_PROMPT)
        result = await client.get_response([{"role": "user", "content": "Buenos días"}])
        assert isinstance(result, GPTResponse)
        assert result.reply_es == "Hola"
        assert result.translation_uk == "Привіт"


@pytest.mark.asyncio
async def test_get_response_retries_on_invalid_json():
    with patch("src.gpt_client.AsyncOpenAI") as MockOAI:
        bad = MagicMock()
        bad.content = '{"reply_es": "Hola'  # truncated
        good = MagicMock()
        good.content = '{"reply_es": "Hola", "translation_uk": "Привіт"}'
        MockOAI.return_value.chat.completions.create = AsyncMock(side_effect=[
            MagicMock(choices=[MagicMock(message=bad)]),
            MagicMock(choices=[MagicMock(message=good)]),
        ])
        client = GPTClient("key", SYSTEM_PROMPT)
        result = await client.get_response([{"role": "user", "content": "Hola"}])
        assert result.reply_es == "Hola"


@pytest.mark.asyncio
async def test_get_response_raises_after_two_invalid_json():
    with patch("src.gpt_client.AsyncOpenAI") as MockOAI:
        bad = MagicMock()
        bad.content = "not json"
        MockOAI.return_value.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=bad)])
        )
        client = GPTClient("key", SYSTEM_PROMPT)
        with pytest.raises(GPTClientError):
            await client.get_response([{"role": "user", "content": "Hola"}])


@pytest.mark.asyncio
async def test_get_response_times_out_after_3s():
    with patch("src.gpt_client.AsyncOpenAI") as MockOAI:
        async def slow(*a, **kw):
            await asyncio.sleep(5)
        MockOAI.return_value.chat.completions.create = slow
        client = GPTClient("key", SYSTEM_PROMPT)
        with pytest.raises(asyncio.TimeoutError):
            await client.get_response([{"role": "user", "content": "Hola"}])
