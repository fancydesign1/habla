import asyncio
import logging
from typing import Callable, Awaitable

from deepgram import Deepgram
from deepgram.transcription import LiveTranscriptionEvent

logger = logging.getLogger(__name__)

_BACKOFF_DELAYS = [0.5, 1.0, 2.0]


class DeepgramSTTClient:
    """Streaming Spanish STT client using Deepgram SDK v2."""

    def __init__(
        self,
        api_key: str,
        on_final: Callable[[str], Awaitable[None]],
        on_interim: Callable[[str], Awaitable[None]],
    ):
        self._api_key = api_key
        self._on_final = on_final
        self._on_interim = on_interim
        self._live = None

    async def connect(self) -> None:
        dg = Deepgram(self._api_key)
        self._live = await dg.transcription.live({
            "model": "nova-2",
            "language": "es",
            "encoding": "linear16",
            "sample_rate": 8000,
            "interim_results": True,
        })
        self._live.register_handler(
            LiveTranscriptionEvent.TRANSCRIPT_RECEIVED,
            self._on_transcript,
        )
        self._live.register_handler(
            LiveTranscriptionEvent.ERROR,
            self._on_error,
        )

    def _on_transcript(self, data) -> None:
        try:
            channel = data.get("channel", {})
            alts = channel.get("alternatives", [])
            if not alts:
                return
            text = alts[0].get("transcript", "").strip()
            if not text:
                return
            is_final = data.get("is_final", False)
            if is_final:
                asyncio.ensure_future(self._on_final(text))
            else:
                asyncio.ensure_future(self._on_interim(text))
        except Exception as e:
            logger.error("Deepgram transcript handler error: %s", e)

    def _on_error(self, error) -> None:
        logger.error("Deepgram error: %s", error)

    def send_audio(self, chunk: bytes) -> None:
        if self._live is None:
            return
        try:
            self._live.send(chunk)
        except Exception as e:
            logger.warning("Deepgram send error: %s", e)

    async def disconnect(self) -> None:
        if self._live is not None:
            try:
                await self._live.finish()
            except Exception as e:
                logger.warning("Deepgram disconnect error: %s", e)
            self._live = None

    async def reconnect_with_backoff(self) -> bool:
        for i, delay in enumerate(_BACKOFF_DELAYS):
            try:
                await self.connect()
                logger.info("Deepgram reconnected on attempt %d", i + 1)
                return True
            except Exception as e:
                logger.warning("Deepgram reconnect attempt %d failed: %s", i + 1, e)
                if i < len(_BACKOFF_DELAYS) - 1:
                    await asyncio.sleep(delay)
        return False
