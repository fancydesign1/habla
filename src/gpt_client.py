import asyncio
import json
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 3.0
_JSON_RETRY_MSG = "Your response was not valid JSON. Return only a JSON object with keys reply_es and translation_uk."


class GPTClientError(Exception):
    pass


@dataclass
class GPTResponse:
    reply_es: str
    translation_uk: str


class GPTClient:
    """GPT-4o conversation client returning bilingual JSON responses."""

    def __init__(self, api_key: str, system_prompt: str):
        self._client = AsyncOpenAI(api_key=api_key)
        self._system_prompt = system_prompt

    async def get_response(self, history: list[dict]) -> GPTResponse:
        messages = [{"role": "system", "content": self._system_prompt}] + history
        for attempt in range(2):
            raw = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model="gpt-4o",
                    response_format={"type": "json_object"},
                    messages=messages,
                ),
                timeout=_TIMEOUT_SECONDS,
            )
            content = raw.choices[0].message.content
            try:
                data = json.loads(content)
                return GPTResponse(
                    reply_es=data["reply_es"],
                    translation_uk=data["translation_uk"],
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("GPT JSON parse error (attempt %d): %s", attempt + 1, e)
                if attempt == 0:
                    messages = messages + [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": _JSON_RETRY_MSG},
                    ]
        raise GPTClientError("GPT returned invalid JSON after 2 attempts")
