import asyncio
import os
import wave
import pytest
from src.deepgram_client import DeepgramSTTClient

pytestmark = pytest.mark.skipif(
    not os.getenv("DEEPGRAM_API_KEY"),
    reason="DEEPGRAM_API_KEY not set"
)


@pytest.mark.asyncio
async def test_spanish_greeting_transcribed():
    received = []

    async def on_final(text):
        received.append(text)

    async def on_interim(text):
        pass

    client = DeepgramSTTClient(os.environ["DEEPGRAM_API_KEY"], on_final, on_interim)
    await client.connect()

    with wave.open("tests/fixtures/spanish_greeting.wav", "rb") as wf:
        while True:
            chunk = wf.readframes(160)
            if not chunk:
                break
            client.send_audio(chunk)
            await asyncio.sleep(0.02)

    await asyncio.sleep(2)
    await client.disconnect()

    full = " ".join(received).lower()
    assert "movistar" in full or "buenos" in full
