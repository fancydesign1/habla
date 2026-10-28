import logging
from typing import AsyncIterator

from elevenlabs import AsyncElevenLabs

logger = logging.getLogger(__name__)

_MODEL_ID = "eleven_turbo_v2_5"
# ulaw_8000 is natively supported by the SDK and is Twilio-compatible directly
_OUTPUT_FORMAT = "ulaw_8000"


class ElevenLabsError(Exception):
    pass


class ElevenLabsClient:
    """ElevenLabs TTS client streaming μ-law 8kHz audio chunks."""

    def __init__(self, api_key: str, voice_id: str):
        self._client = AsyncElevenLabs(api_key=api_key)
        self._voice_id = voice_id
        self._aborted = False

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self._aborted = False
        try:
            stream = self._client.text_to_speech.convert_as_stream(
                voice_id=self._voice_id,
                text=text,
                model_id=_MODEL_ID,
                output_format=_OUTPUT_FORMAT,
            )
            async for chunk in stream:
                if self._aborted:
                    return
                yield chunk
        except Exception as e:
            logger.error("ElevenLabs synthesis error: %s", e)
            raise ElevenLabsError(str(e)) from e

    def abort(self) -> None:
        self._aborted = True
