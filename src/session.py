import asyncio
import base64
import logging
from typing import Optional

from src.audio import UlawToLinear16Converter, Mp3ToUlawConverter
from src.deepgram_client import DeepgramSTTClient
from src.gpt_client import GPTClient, GPTClientError
from src.elevenlabs_client import ElevenLabsClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a Spanish telephone assistant handling a call on behalf of the user.\n"
    "- Respond naturally and politely in Spanish only.\n"
    "- Ask for clarification when the caller's intent is unclear.\n"
    "- Never accept contracts, authorize payments, or share personal information.\n"
    "- If asked to make a commitment, say you need to check with the account holder first.\n"
    '- Always return valid JSON: {"reply_es": "<Spanish response>", '
    '"translation_uk": "<Ukrainian translation of what the caller just said>"}'
)

_MAX_HISTORY = 20
_CANNED_RESPONSE_ES = "Disculpe, ¿puede repetir?"
_CANNED_RESPONSE_UK = "(Не вдалося обробити відповідь)"


class CallSession:
    """Orchestrates STT → GPT → TTS pipeline for a single Twilio call."""

    def __init__(self, session_id: str, twilio_ws, settings):
        self.session_id = session_id
        self.twilio_ws = twilio_ws
        self.stream_sid: Optional[str] = None
        self.is_ai_speaking = False
        self.conversation_history: list[dict] = []
        self.ui_clients: set = set()

        self._conv_in = UlawToLinear16Converter()
        self._conv_out = Mp3ToUlawConverter()

        self._stt_client = DeepgramSTTClient(
            api_key=settings.deepgram_api_key,
            on_final=self._on_final_transcript,
            on_interim=self._on_interim_transcript,
        )
        self._gpt_client = GPTClient(
            api_key=settings.openai_api_key,
            system_prompt=SYSTEM_PROMPT,
        )
        self._tts_client = ElevenLabsClient(
            api_key=settings.elevenlabs_api_key,
            voice_id=settings.elevenlabs_voice_id,
        )

    def add_ui_client(self, ws) -> None:
        self.ui_clients.add(ws)

    def remove_ui_client(self, ws) -> None:
        self.ui_clients.discard(ws)

    async def broadcast_ui_event(self, event: dict) -> None:
        dead = set()
        for ws in list(self.ui_clients):
            try:
                await ws.send_json(event)
            except Exception as e:
                logger.warning("UI client disconnected: %s", e)
                dead.add(ws)
        self.ui_clients -= dead

    def _trim_history(self) -> None:
        if len(self.conversation_history) > _MAX_HISTORY:
            self.conversation_history = self.conversation_history[-_MAX_HISTORY:]

    async def _handle_interruption(self) -> None:
        self._tts_client.abort()
        await self.twilio_ws.send_json({"event": "clear", "streamSid": self.stream_sid})
        self.is_ai_speaking = False

    async def _on_interim_transcript(self, text: str) -> None:
        if self.is_ai_speaking:
            await self._handle_interruption()
        await self.broadcast_ui_event({"type": "interim", "text": text, "lang": "es"})

    async def _on_final_transcript(self, text: str) -> None:
        if self.is_ai_speaking:
            await self._handle_interruption()
        await self.broadcast_ui_event({"type": "caller", "text": text, "lang": "es"})
        self.conversation_history.append({"role": "user", "content": text})
        self._trim_history()
        await self._generate_and_speak(text)

    async def _generate_and_speak(self, user_text: str) -> None:
        try:
            response = await self._gpt_client.get_response(self.conversation_history)
        except (GPTClientError, asyncio.TimeoutError) as e:
            logger.error("GPT error: %s", e)
            await self.broadcast_ui_event({
                "type": "ai",
                "text": _CANNED_RESPONSE_ES,
                "translation": _CANNED_RESPONSE_UK,
            })
            return

        self.conversation_history.append({"role": "assistant", "content": response.reply_es})
        await self.broadcast_ui_event({
            "type": "ai",
            "text": response.reply_es,
            "translation": response.translation_uk,
        })

        self.is_ai_speaking = True
        try:
            async for chunk in self._tts_client.synthesize(response.reply_es):
                if not self.is_ai_speaking:
                    break
                payload = base64.b64encode(chunk).decode()
                await self.twilio_ws.send_json({
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": payload},
                })
        except Exception as e:
            logger.error("TTS error: %s", e)
        finally:
            self.is_ai_speaking = False

    async def handle_audio_chunk(self, chunk: bytes) -> None:
        if self.is_ai_speaking:
            await self._handle_interruption()
        self._conv_in.write(chunk)
        linear_data = self._conv_in.read()
        if linear_data:
            self._stt_client.send_audio(linear_data)

    async def run(self) -> None:
        await self._stt_client.connect()
        try:
            async for message in self.twilio_ws.iter_json():
                event = message.get("event")
                if event == "start":
                    self.stream_sid = message.get("start", {}).get("streamSid")
                elif event == "media":
                    payload = message.get("media", {}).get("payload", "")
                    chunk = base64.b64decode(payload)
                    await self.handle_audio_chunk(chunk)
                elif event == "stop":
                    break
        except Exception as e:
            logger.error("Twilio WS error: %s", e)
        finally:
            await self.cleanup()

    async def cleanup(self) -> None:
        self._tts_client.abort()
        try:
            await self._stt_client.disconnect()
        except Exception as e:
            logger.warning("STT disconnect error: %s", e)
        self._conv_in.close()
        self._conv_out.close()
