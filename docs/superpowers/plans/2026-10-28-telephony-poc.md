# Telephony PoC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI backend that bridges a real Spanish phone call through Twilio, transcribes it with Deepgram, generates a Spanish response via GPT-4o, speaks it back via ElevenLabs, and streams a live bilingual (Spanish + Ukrainian) transcript to a browser UI over WebSocket.

**Architecture:** Twilio Media Streams sends μ-law 8kHz audio over WebSocket to the backend. The backend converts audio with long-running piped ffmpeg processes, feeds it to Deepgram for streaming STT, sends transcripts to GPT-4o which returns `{reply_es, translation_uk}` JSON, then streams ElevenLabs MP3 back through ffmpeg to Twilio. A second WebSocket endpoint pushes JSON events to any connected browser UI tabs.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, Deepgram Python SDK (nova-2/es), OpenAI Python SDK (gpt-4o), ElevenLabs Python SDK (eleven_turbo_v2_5), ffmpeg, Twilio Media Streams, pytest, python-dotenv

**Spec:** `docs/superpowers/specs/2026-10-28-telephony-poc-design.md`

## Global Constraints

- Python 3.12 — no earlier versions
- GPT-4o model ID: `gpt-4o` (not gpt-4o-mini)
- Deepgram model: `nova-2`, language: `es`
- ElevenLabs model: `eleven_turbo_v2_5`
- ffmpeg processes must be long-running with stdin/stdout pipes — no per-chunk subprocess spawning
- One ffmpeg process per call per direction (ulaw→linear16, mp3→ulaw); mp3→ulaw process restarts on stream abort
- `conversation_history` trimmed to last 20 messages (10 caller + 10 AI turns)
- Deepgram reconnect: exponential backoff, max 3 attempts, delays 0.5s / 1s / 2s
- GPT-4o timeout: 3s; canned fallback: `"Disculpe, ¿puede repetir?"`
- GPT-4o invalid JSON: retry once with JSON reminder appended; second failure uses canned response
- ElevenLabs stream error: fall back to Twilio `<Say>` for that turn only
- All errors log to stdout with `session_id`, timestamp, error type

## Review Focus

1. **Caller speaks while AI is mid-response** — `is_ai_speaking=True` when new Deepgram final transcript arrives; orchestrator must send Twilio `clear`, abort ElevenLabs stream, and process new transcript without dropping it. Covered in Task 6, `test_interruption_aborts_tts`.
2. **GPT-4o returns truncated JSON** — e.g. `{"reply_es": "Hola` with no closing brace; `json.JSONDecodeError` must trigger the retry path, not a crash. Covered in Task 4, `test_get_response_retries_on_invalid_json`.
3. **Multiple browser UI tabs open simultaneously** — all `ui_clients` in the session must receive every event; a disconnected client must be removed without breaking delivery to others. Covered in Task 6, `test_broadcast_to_multiple_ui_clients`.
4. **ffmpeg process dies mid-call** — `IOError` on the pipe write must log and skip the chunk, not raise to the Twilio WebSocket handler. Covered in Task 2, `test_converter_write_after_close_skips_gracefully`.
5. **Twilio sends the `stop` event before all ElevenLabs audio is flushed** — session cleanup must abort the TTS stream and close both ffmpeg processes without hanging. Covered in Task 6, `test_session_cleanup_on_twilio_stop`.

---

## File Structure

```
src/
  audio.py          — UlawToLinear16Converter, Mp3ToUlawConverter (piped ffmpeg)
  deepgram_client.py — DeepgramSTTClient (streaming Spanish STT)
  gpt_client.py     — GPTClient, GPTResponse (conversation + translation)
  elevenlabs_client.py — ElevenLabsClient (streaming TTS, abort)
  session.py        — CallSession dataclass + orchestration logic
  twilio_handler.py — FastAPI router: POST /twiml, WS /ws/call, WS /ws/ui
  main.py           — FastAPI app, mounts router, serves static/index.html
static/
  index.html        — browser transcript UI (plain HTML + JS, no framework)
tests/
  test_audio.py
  test_gpt_client.py
  test_deepgram_client.py
  test_elevenlabs_client.py
  test_session.py
  test_twilio_handler.py
  test_integration_stt.py   — WAV replay smoke test (requires DEEPGRAM_API_KEY)
  fixtures/
    spanish_greeting.wav    — pre-recorded "Buenos días, llamo de Movistar"
requirements.txt
.env.example
```

---

### Task 1: Project scaffold

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `src/main.py` (app skeleton only)

**Interfaces:**
- Produces: `app` FastAPI instance importable as `from src.main import app`

- [ ] **Step 1: Create `requirements.txt`**

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
python-dotenv==1.0.1
deepgram-sdk==3.7.7
openai==1.51.0
elevenlabs==1.9.0
twilio==9.3.4
pytest==8.3.3
pytest-asyncio==0.24.0
httpx==0.27.2
```

- [ ] **Step 2: Create `.env.example`**

```
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=
DEEPGRAM_API_KEY=
OPENAI_API_KEY=
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=
PUBLIC_URL=https://your-ngrok-url.ngrok.io
```

- [ ] **Step 3: Create `src/settings.py`**

```python
from dataclasses import dataclass
import os

@dataclass
class Settings:
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_phone_number: str
    deepgram_api_key: str
    openai_api_key: str
    elevenlabs_api_key: str
    elevenlabs_voice_id: str
    public_url: str  # e.g. "abc123.ngrok.io" — no scheme, no trailing slash

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(**{f.name: os.environ[f.name.upper()] for f in fields(cls)})
```

Use `from dataclasses import fields` for `from_env`. Add `src/settings.py` to the file structure above.

- [ ] **Step 4: Create `src/main.py`**

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
```

- [ ] **Step 5: Write the scaffold test**

```python
# tests/test_main.py
from fastapi.testclient import TestClient
from src.main import app

def test_app_creates():
    client = TestClient(app)
    assert client is not None
```

- [ ] **Step 6: Run test**

```bash
pytest tests/test_main.py -v
```
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .env.example src/ tests/ static/
git commit -m "feat: project scaffold"
```

---

### Task 2: Audio conversion

**Files:**
- Create: `src/audio.py`
- Create: `tests/test_audio.py`

**Interfaces:**
- Produces:
  - `UlawToLinear16Converter` — call `write(chunk: bytes)` to feed μ-law, `read() -> bytes` to get linear16; `close()` shuts down ffmpeg
  - `Mp3ToUlawConverter` — call `write(chunk: bytes)` to feed MP3, `read_available() -> bytes` to drain ulaw output; `close()` shuts down ffmpeg; `restart()` kills and relaunches the process

- [ ] **Step 1: Write failing tests**

```python
# tests/test_audio.py
import pytest
from src.audio import UlawToLinear16Converter, Mp3ToUlawConverter

def test_ulaw_converter_produces_bytes():
    c = UlawToLinear16Converter()
    silence_ulaw = b'\x7f' * 160  # 20ms of silence at 8kHz
    c.write(silence_ulaw)
    out = c.read()
    c.close()
    assert isinstance(out, bytes)
    assert len(out) > 0

def test_ulaw_converter_write_after_close_skips_gracefully():
    c = UlawToLinear16Converter()
    c.close()
    c.write(b'\x7f' * 160)  # must not raise

def test_mp3_converter_restart_replaces_process():
    c = Mp3ToUlawConverter()
    pid1 = c._process.pid
    c.restart()
    pid2 = c._process.pid
    c.close()
    assert pid1 != pid2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_audio.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/audio.py`**

`UlawToLinear16Converter`: launch `ffmpeg -f mulaw -ar 8000 -ac 1 -i pipe:0 -f s16le -ar 8000 -ac 1 pipe:1` as a `subprocess.Popen` with `stdin=PIPE, stdout=PIPE`. `write` feeds stdin; `read` drains stdout non-blocking. `close` terminates the process; `write` after close logs a warning and returns.

`Mp3ToUlawConverter`: launch `ffmpeg -f mp3 -i pipe:0 -f mulaw -ar 8000 -ac 1 pipe:1`. Same interface. `restart` terminates the old process and spawns a new one. `read_available` returns all bytes currently buffered in stdout without blocking.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_audio.py -v
```
Expected: PASS (requires ffmpeg installed)

- [ ] **Step 5: Commit**

```bash
git add src/audio.py tests/test_audio.py
git commit -m "feat: piped ffmpeg audio converters"
```

---

### Task 3: Deepgram STT client

**Files:**
- Create: `src/deepgram_client.py`
- Create: `tests/test_deepgram_client.py`

**Interfaces:**
- Consumes: linear16 8kHz bytes from `UlawToLinear16Converter.read()`
- Produces:
  - `DeepgramSTTClient(api_key: str, on_final: Callable[[str], Awaitable[None]], on_interim: Callable[[str], Awaitable[None]])`
  - `async connect() -> None`
  - `send_audio(chunk: bytes) -> None` — sends linear16 bytes to Deepgram live stream
  - `async disconnect() -> None`
  - `async reconnect_with_backoff() -> bool` — returns `True` if reconnected, `False` after 3 failed attempts

- [ ] **Step 1: Write failing tests**

```python
# tests/test_deepgram_client.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.deepgram_client import DeepgramSTTClient

@pytest.mark.asyncio
async def test_connect_subscribes_to_events():
    on_final = AsyncMock()
    on_interim = AsyncMock()
    mock_live = MagicMock()
    mock_live.on = MagicMock()
    mock_live.start = AsyncMock()

    with patch("src.deepgram_client.DeepgramClient") as MockDG:
        MockDG.return_value.listen.asynclive.v.return_value = mock_live
        client = DeepgramSTTClient("key", on_final, on_interim)
        await client.connect()
        assert mock_live.start.called

@pytest.mark.asyncio
async def test_reconnect_returns_false_after_three_attempts():
    on_final = AsyncMock()
    on_interim = AsyncMock()
    with patch("src.deepgram_client.DeepgramClient") as MockDG:
        MockDG.return_value.listen.asynclive.v.return_value.start = AsyncMock(
            side_effect=Exception("connection refused")
        )
        client = DeepgramSTTClient("key", on_final, on_interim)
        result = await client.reconnect_with_backoff()
        assert result is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_deepgram_client.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement `src/deepgram_client.py`**

`connect()`: initialise `DeepgramClient`, call `listen.asynclive.v("1")` with options `model="nova-2", language="es", encoding="linear16", sample_rate=8000, interim_results=True`. Register `on` handlers for `LiveTranscriptionEvents.Transcript` (fires `on_final` on `is_final=True`, `on_interim` on `is_final=False`) and `LiveTranscriptionEvents.Error`.

`reconnect_with_backoff()`: try `connect()` up to 3 times with `asyncio.sleep(0.5)`, `asyncio.sleep(1)`, `asyncio.sleep(2)` between attempts. Log each failure with attempt number. Return `True` on success, `False` after all attempts exhausted.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_deepgram_client.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/deepgram_client.py tests/test_deepgram_client.py
git commit -m "feat: Deepgram streaming STT client"
```

---

### Task 4: GPT-4o conversation client

**Files:**
- Create: `src/gpt_client.py`
- Create: `tests/test_gpt_client.py`

**Interfaces:**
- Produces:
  - `GPTResponse(reply_es: str, translation_uk: str)` dataclass
  - `GPTClient(api_key: str, system_prompt: str)`
  - `async get_response(history: list[dict]) -> GPTResponse` — raises `asyncio.TimeoutError` after 3s; raises `GPTClientError` on second consecutive invalid JSON

- [ ] **Step 1: Write failing tests**

```python
# tests/test_gpt_client.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_gpt_client.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement `src/gpt_client.py`**

`get_response(history)`: call `AsyncOpenAI.chat.completions.create(model="gpt-4o", messages=[system]+history, response_format={"type":"json_object"})` wrapped in `asyncio.wait_for(..., timeout=3.0)`. Parse response with `json.loads`. On `JSONDecodeError`: append `"Return valid JSON only: {\"reply_es\":...,\"translation_uk\":...}"` to the last user message and retry once. On second failure raise `GPTClientError`.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_gpt_client.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/gpt_client.py tests/test_gpt_client.py
git commit -m "feat: GPT-4o conversation client with JSON retry"
```

---

### Task 5: ElevenLabs TTS client

**Files:**
- Create: `src/elevenlabs_client.py`
- Create: `tests/test_elevenlabs_client.py`

**Interfaces:**
- Produces:
  - `ElevenLabsClient(api_key: str, voice_id: str)`
  - `async synthesize(text: str) -> AsyncIterator[bytes]` — yields MP3 chunks; raises `ElevenLabsError` on stream failure
  - `abort() -> None` — sets internal flag; `synthesize` stops yielding on next iteration

- [ ] **Step 1: Write failing tests**

```python
# tests/test_elevenlabs_client.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.elevenlabs_client import ElevenLabsClient, ElevenLabsError

@pytest.mark.asyncio
async def test_synthesize_yields_bytes():
    with patch("src.elevenlabs_client.AsyncElevenLabs") as MockEL:
        async def fake_stream(*a, **kw):
            yield b"mp3chunk1"
            yield b"mp3chunk2"
        MockEL.return_value.text_to_speech.stream = AsyncMock(
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
        MockEL.return_value.text_to_speech.stream = AsyncMock(
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
        MockEL.return_value.text_to_speech.stream = AsyncMock(
            return_value=bad_stream()
        )
        client = ElevenLabsClient("key", "voice_id")
        with pytest.raises(ElevenLabsError):
            async for _ in client.synthesize("Hola"):
                pass
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_elevenlabs_client.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement `src/elevenlabs_client.py`**

`synthesize(text)`: call `AsyncElevenLabs.text_to_speech.stream(voice_id=..., model_id="eleven_turbo_v2_5", text=text)`. Set `_aborted = False` at start. On each yielded chunk, check `_aborted`; if set, return. Wrap in `try/except Exception` → raise `ElevenLabsError`. `abort()` sets `self._aborted = True`.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_elevenlabs_client.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elevenlabs_client.py tests/test_elevenlabs_client.py
git commit -m "feat: ElevenLabs TTS client with abort"
```

---

### Task 6: CallSession orchestrator

**Files:**
- Create: `src/session.py`
- Create: `tests/test_session.py`

**Interfaces:**
- Consumes: `DeepgramSTTClient`, `GPTClient`, `ElevenLabsClient`, `UlawToLinear16Converter`, `Mp3ToUlawConverter` from earlier tasks
- Produces:
  - `CallSession(session_id: str, twilio_ws: WebSocket, settings: Settings)`
  - `async run() -> None` — main loop; returns when Twilio WS closes
  - `async handle_audio_chunk(chunk: bytes) -> None`
  - `async broadcast_ui_event(event: dict) -> None`
  - `add_ui_client(ws: WebSocket) -> None`
  - `remove_ui_client(ws: WebSocket) -> None`
  - `SYSTEM_PROMPT: str` module-level constant (exact text from spec section 4.4)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_session.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.session import CallSession, SYSTEM_PROMPT

def make_session(twilio_ws=None):
    settings = MagicMock()
    settings.deepgram_api_key = "dg"
    settings.openai_api_key = "oai"
    settings.elevenlabs_api_key = "el"
    settings.elevenlabs_voice_id = "v"
    ws = twilio_ws or AsyncMock()
    return CallSession(session_id="test-123", twilio_ws=ws, settings=settings)

@pytest.mark.asyncio
async def test_history_trimmed_to_20_messages():
    session = make_session()
    for i in range(25):
        role = "user" if i % 2 == 0 else "assistant"
        session.conversation_history.append({"role": role, "content": f"msg {i}"})
    session._trim_history()
    assert len(session.conversation_history) == 20

@pytest.mark.asyncio
async def test_interruption_aborts_tts():
    session = make_session()
    session.is_ai_speaking = True
    mock_el = MagicMock()
    mock_el.abort = MagicMock()
    session._tts_client = mock_el
    twilio_ws = AsyncMock()
    session.twilio_ws = twilio_ws

    await session._handle_interruption()

    mock_el.abort.assert_called_once()
    sent = twilio_ws.send_json.call_args[0][0]
    assert sent["event"] == "clear"
    assert session.is_ai_speaking is False

@pytest.mark.asyncio
async def test_broadcast_to_multiple_ui_clients():
    session = make_session()
    ws1 = AsyncMock()
    ws2 = AsyncMock()
    session.add_ui_client(ws1)
    session.add_ui_client(ws2)
    await session.broadcast_ui_event({"type": "caller", "text": "Hola", "lang": "es"})
    ws1.send_json.assert_called_once_with({"type": "caller", "text": "Hola", "lang": "es"})
    ws2.send_json.assert_called_once_with({"type": "caller", "text": "Hola", "lang": "es"})

@pytest.mark.asyncio
async def test_broadcast_removes_disconnected_client():
    session = make_session()
    ws_dead = AsyncMock()
    ws_dead.send_json = AsyncMock(side_effect=Exception("disconnected"))
    ws_live = AsyncMock()
    session.add_ui_client(ws_dead)
    session.add_ui_client(ws_live)
    await session.broadcast_ui_event({"type": "status", "text": "test"})
    assert ws_dead not in session.ui_clients
    ws_live.send_json.assert_called_once()

@pytest.mark.asyncio
async def test_session_cleanup_on_twilio_stop():
    session = make_session()
    mock_el = MagicMock()
    mock_el.abort = MagicMock()
    session._tts_client = mock_el
    mock_dg = AsyncMock()
    session._stt_client = mock_dg
    mock_conv_in = MagicMock()
    mock_conv_in.close = MagicMock()
    mock_conv_out = MagicMock()
    mock_conv_out.close = MagicMock()
    session._conv_in = mock_conv_in
    session._conv_out = mock_conv_out

    await session.cleanup()

    mock_el.abort.assert_called_once()
    mock_dg.disconnect.assert_called_once()
    mock_conv_in.close.assert_called_once()
    mock_conv_out.close.assert_called_once()

def test_system_prompt_contains_required_rules():
    assert "never accept contracts" in SYSTEM_PROMPT.lower() or "never accept" in SYSTEM_PROMPT.lower()
    assert "reply_es" in SYSTEM_PROMPT
    assert "translation_uk" in SYSTEM_PROMPT
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_session.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement `src/session.py`**

`SYSTEM_PROMPT`: exact text from spec section 4.4.

`CallSession.__init__`: initialise all clients (`DeepgramSTTClient`, `GPTClient`, `ElevenLabsClient`, two converters), set `is_ai_speaking=False`, `conversation_history=[]`, `ui_clients=set()`.

`_trim_history()`: if `len(conversation_history) > 20`, slice to last 20.

`_handle_interruption()`: call `_tts_client.abort()`, send `{"event": "clear", "streamSid": self.stream_sid}` to `twilio_ws`, set `is_ai_speaking=False`.

`broadcast_ui_event(event)`: iterate `ui_clients` copy; on exception remove client from set.

`cleanup()`: call `_tts_client.abort()`, `await _stt_client.disconnect()`, `_conv_in.close()`, `_conv_out.close()`.

`run()`: connect STT, loop on Twilio WS messages; on `media` event call `handle_audio_chunk`; on `stop` event call `cleanup()` and return.

`handle_audio_chunk(chunk)`: if `is_ai_speaking`, call `_handle_interruption()`. Decode base64 chunk, write to `_conv_in`, read linear16 output, send to `_stt_client`. On Deepgram final transcript: `_trim_history()`, call `GPTClient.get_response()`, broadcast events to UI, stream ElevenLabs → `_conv_out` → Twilio. On `GPTClientError` or `asyncio.TimeoutError`: send canned response `"Disculpe, ¿puede repetir?"` via Twilio `<Say>`. On `ElevenLabsError`: fall back to Twilio `<Say>`.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_session.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/session.py tests/test_session.py
git commit -m "feat: CallSession orchestrator"
```

---

### Task 7: Twilio handler + full FastAPI wiring

**Files:**
- Create: `src/twilio_handler.py`
- Modify: `src/main.py` (add router, settings, UI WebSocket endpoint, `/` route)
- Create: `tests/test_twilio_handler.py`

**Interfaces:**
- Consumes: `CallSession` from Task 6
- Produces:
  - `POST /twiml` → TwiML response connecting call to `/ws/call`
  - `WS /ws/call` → creates `CallSession`, runs it, cleans up on close
  - `WS /ws/ui` → registers browser client with active session; sends `{"type":"status","text":"no_active_call"}` if none

- [ ] **Step 1: Write failing tests**

```python
# tests/test_twilio_handler.py
import pytest
from fastapi.testclient import TestClient
from src.main import app

def test_twiml_endpoint_returns_stream_connect():
    client = TestClient(app)
    response = client.post("/twiml", data={"CallSid": "CA123"})
    assert response.status_code == 200
    assert "Connect" in response.text
    assert "Stream" in response.text
    assert "/ws/call" in response.text

def test_twiml_uses_wss_scheme():
    client = TestClient(app)
    response = client.post("/twiml", data={"CallSid": "CA123"})
    assert "wss://" in response.text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_twilio_handler.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement `src/twilio_handler.py`**

`POST /twiml`: read `PUBLIC_URL` from settings; return TwiML `<Response><Connect><Stream url="wss://{PUBLIC_URL}/ws/call"/></Connect></Response>` with `content_type="text/xml"`.

`WS /ws/call`: accept WS, create `CallSession`, call `await session.run()`, call `await session.cleanup()` in `finally`.

`WS /ws/ui`: accept WS, look up current session from `active_session` module-level variable (a single `CallSession | None`); if `None`, send `{"type":"status","text":"no_active_call"}` and close; else call `session.add_ui_client(ws)` and keep alive until disconnect, then call `session.remove_ui_client(ws)`.

`WS /ws/call`: on connect, create `CallSession`, assign to `active_session`; in `finally`, set `active_session = None`.

`src/main.py`: load `.env` with `python-dotenv`, instantiate `Settings.from_env()`, include `twilio_handler.router`, serve `static/index.html` at `GET /`.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_twilio_handler.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/twilio_handler.py src/main.py tests/test_twilio_handler.py
git commit -m "feat: Twilio handler and FastAPI wiring"
```

---

### Task 8: Web transcript UI

**Files:**
- Create: `static/index.html`

**Interfaces:**
- Consumes: `WS /ws/ui` JSON events (from Task 7)
- No automated tests — verify manually after Task 9

- [ ] **Step 1: Implement `static/index.html`**

Plain HTML + vanilla JS. On load connect to `ws://<host>/ws/ui`. Handle event types:
- `interim`: update a "typing..." indicator in the Spanish column (replace on each interim)
- `caller`: clear interim indicator; append a caller row to Spanish column
- `translation`: append the Ukrainian text to Ukrainian column aligned with the last caller row
- `ai`: append AI row to Spanish column; update "AI active" indicator based on `is_ai_speaking` state
- `status` with `text="call_ended"`: show "Call ended" banner

Layout: two-column scrolling transcript. Green dot labelled "AI active" in the header. Auto-scroll to bottom on new events. No framework, no build step.

- [ ] **Step 2: Manual smoke check**

```bash
uvicorn src.main:app --reload
```

Open `http://localhost:8000` in a browser. Verify the page loads with two columns and a header. Open the browser console — no JS errors.

- [ ] **Step 3: Commit**

```bash
git add static/index.html
git commit -m "feat: web transcript UI"
```

---

### Task 9: Integration smoke test + ngrok setup

**Files:**
- Create: `tests/test_integration_stt.py`
- Create: `tests/fixtures/spanish_greeting.wav` (generated in step 1)
- Create: `README.md`

**Interfaces:**
- Consumes: `DeepgramSTTClient` from Task 3
- Marked `pytest.mark.integration` — skipped unless `DEEPGRAM_API_KEY` is set in environment

- [ ] **Step 1: Generate test WAV fixture**

```bash
# requires ffmpeg and espeak-ng or a pre-recorded file
# simplest: use a short silence + spoken phrase via macOS say
say -v Juan "Buenos días, llamo de Movistar para ofrecerle una nueva tarifa" -o /tmp/greeting.aiff
ffmpeg -i /tmp/greeting.aiff -ar 8000 -ac 1 -f wav tests/fixtures/spanish_greeting.wav
```

Commit the WAV file (it will be a few KB).

- [ ] **Step 2: Write integration test**

```python
# tests/test_integration_stt.py
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
    async def on_final(text): received.append(text)
    async def on_interim(text): pass

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
```

- [ ] **Step 3: Run integration test**

```bash
DEEPGRAM_API_KEY=your_key pytest tests/test_integration_stt.py -v -m integration
```
Expected: PASS with transcript containing "movistar" or "buenos"

- [ ] **Step 4: Write `README.md`**

Cover: prerequisites (Python 3.12, ffmpeg, ngrok), API key setup (copy `.env.example` → `.env`, fill in keys), how to provision a Twilio number and set the webhook to `https://<ngrok>/twiml`, how to run locally (`uvicorn src.main:app --reload`), how to expose with ngrok (`ngrok http 8000`), and the end-to-end manual test steps from the spec.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v --ignore=tests/test_integration_stt.py
```
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_integration_stt.py tests/fixtures/spanish_greeting.wav README.md
git commit -m "feat: integration smoke test and README"
```

---

## Summary

| Task | Deliverable | Tests |
|---|---|---|
| 1 | Scaffold, deps, app skeleton | test_main.py |
| 2 | piped ffmpeg audio converters | test_audio.py |
| 3 | Deepgram streaming STT | test_deepgram_client.py |
| 4 | GPT-4o client + JSON retry | test_gpt_client.py |
| 5 | ElevenLabs TTS + abort | test_elevenlabs_client.py |
| 6 | CallSession orchestrator | test_session.py |
| 7 | Twilio handler + FastAPI wiring | test_twilio_handler.py |
| 8 | Web transcript UI | manual |
| 9 | Integration smoke test + README | test_integration_stt.py |
