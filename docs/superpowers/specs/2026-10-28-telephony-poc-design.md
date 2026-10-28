# Telephony PoC Design — Real-Time Spanish AI Call Assistant

**Date:** 2026-10-28  
**Status:** Approved  
**Scope:** Sub-project 1 of 5 — audio pipeline validation, no mobile app

---

## 1. Goal

Validate that an AI assistant can participate in a real Spanish telephone conversation with acceptable latency. A caller speaks Spanish; the AI listens, understands, and responds in natural Spanish; the user watching a web UI sees a live bilingual transcript in Spanish and Ukrainian.

This PoC proves the hard part — two-way audio through a real phone call — before any mobile app is built.

---

## 2. Decisions

| Question | Decision | Rationale |
|---|---|---|
| Telephony approach | Twilio bridge | Reliable two-way audio without OS restrictions; works on any phone |
| PoC scope | Audio pipeline only, no mobile app | Isolates the core unknown; fastest to ship |
| Translation model | Autonomous AI + live transcript | Lower latency; translation is a display layer, not a call blocker |
| User language | Ukrainian | Single target language for PoC |
| AI stack | Deepgram + GPT-4o + ElevenLabs | ~2× cheaper than OpenAI Realtime at ~$0.21/2-min call; acceptable ~1.1s latency |

---

## 3. Architecture

```
Spanish caller
    ↓
Twilio number (provisioned Spanish +34 number)
    ↓  μ-law 8kHz WebSocket (Media Stream)
FastAPI backend
    ├── Deepgram STT  ← streaming Spanish → final transcript
    ├── GPT-4o        ← transcript + history → {reply_es, translation_uk}
    ├── ElevenLabs    ← reply_es → streaming MP3
    │   ↓  μ-law 8kHz back to Twilio
    └── WebSocket → Web transcript UI (browser)
                       live bilingual display (ES + UK)
```

No database. No mobile app. No auth. Session state lives in memory.

---

## 4. Components

### 4.1 Twilio Media Stream handler

- Receives a TwiML `<Connect><Stream>` webhook on call connect
- Opens a WebSocket connection; Twilio sends audio as base64-encoded μ-law 8kHz chunks
- Sends `clear` media command to stop AI playback on interruption
- Writes outgoing μ-law audio chunks back to Twilio as `media` messages

### 4.2 Deepgram STT client

- Language: `es` (Spanish), model: `nova-2`
- Streaming mode: sends audio continuously, receives interim and final transcripts
- Input: linear16 8kHz (converted from μ-law by ffmpeg)
- Output: fires `on_final_transcript(text: str)` event to orchestrator
- Also fires `on_interim_transcript(text: str)` for web UI display

### 4.3 Orchestrator

Per-call state:
```python
session_id: str
conversation_history: list[{"role": str, "content": str}]
is_ai_speaking: bool
stream_sid: str
ui_clients: set[WebSocket]
```

Responsibilities:
- Routes audio chunks from Twilio → Deepgram
- On final transcript: calls GPT-4o, sends response to ElevenLabs + web UI
- On new speech while `is_ai_speaking=True`: sends Twilio `clear`, aborts ElevenLabs stream, processes new input
- Trims `conversation_history` to last 20 messages (10 caller + 10 AI turns) to stay within GPT-4o context

### 4.4 GPT-4o conversation engine

Model: `gpt-4o`  
Response format: JSON `{"reply_es": "...", "translation_uk": "..."}`

System prompt:
```
You are a Spanish telephone assistant handling a call on behalf of the user.
- Respond naturally and politely in Spanish only.
- Ask for clarification when the caller's intent is unclear.
- Never accept contracts, authorize payments, or share personal information.
- If asked to make a commitment, say you need to check with the account holder first.
- Always return valid JSON: {"reply_es": "<Spanish response>", "translation_uk": "<Ukrainian translation of what the caller just said>"}
```

### 4.5 ElevenLabs TTS client

- Model: `eleven_turbo_v2_5`
- Voice: a consistent Spanish-accent voice (configured at startup)
- Input: `reply_es` text
- Output: streaming MP3 chunks → converted to μ-law 8kHz by ffmpeg → sent to Twilio
- Supports stream abort: cancel pending chunks on interruption

### 4.6 Audio format conversion

All conversions via long-running ffmpeg processes with stdin/stdout pipes (not per-chunk subprocess spawning — too slow):
- Twilio → Deepgram: μ-law 8kHz → linear16 8kHz (one persistent ffmpeg process per call)
- ElevenLabs → Twilio: MP3 → μ-law 8kHz mono (one persistent ffmpeg process per call, restarted on stream abort)

### 4.7 Web transcript UI

Single HTML page served by FastAPI at `/`.  
Connects to `/ws/ui` WebSocket on load.  
Receives JSON events from backend:

```json
{"type": "interim", "text": "Buenos día...", "lang": "es"}
{"type": "caller", "text": "Buenos días, llamo de Movistar...", "lang": "es"}
{"type": "translation", "text": "Доброго ранку, телефоную з Movistar...", "lang": "uk"}
{"type": "ai", "text": "¿Me puede decir qué incluye la oferta?", "lang": "es"}
{"type": "status", "text": "call_ended"}
```

Displays a scrolling two-column transcript: Spanish on the left, Ukrainian on the right. Shows a green "AI active" indicator while `is_ai_speaking=True`.

---

## 5. Data Flow — One Full Turn

1. Caller speaks → Twilio streams μ-law audio chunks to backend WS
2. Orchestrator converts to linear16 → streams to Deepgram
3. Deepgram fires interim transcripts → forwarded to web UI (greyed)
4. Deepgram fires final transcript → orchestrator adds to history
5. Orchestrator sends history + system prompt to GPT-4o
6. GPT-4o returns `{reply_es, translation_uk}`
7. Orchestrator sends `translation_uk` + caller Spanish text to web UI
8. Orchestrator sends `reply_es` to ElevenLabs
9. ElevenLabs streams MP3 → ffmpeg converts → μ-law chunks → Twilio
10. Twilio plays audio; caller hears AI speak Spanish
11. Orchestrator sends `reply_es` to web UI; sets `is_ai_speaking=False`

**Interruption path (step 1 while step 9–11 in progress):**  
→ Orchestrator sends Twilio `clear` command  
→ Aborts ElevenLabs stream  
→ Sets `is_ai_speaking=False`  
→ Continues from step 2 with new audio

---

## 6. Error Handling

| Failure | Behaviour |
|---|---|
| Deepgram disconnect | Exponential backoff reconnect (max 3 attempts, 0.5s / 1s / 2s); on failure send Twilio `<Say>` *"Un momento, por favor"* |
| GPT-4o timeout > 3s | Send canned response *"Disculpe, ¿puede repetir?"*; retry on next utterance |
| GPT-4o returns invalid JSON | Retry once with explicit JSON reminder appended to prompt; on second failure use canned response |
| ElevenLabs stream error | Fall back to Twilio `<Say>` TTS for that turn; log degradation to web UI |
| Twilio WebSocket drop | Session ends; web UI shows "Call ended" |
| ffmpeg error | Log and skip chunk; Twilio plays silence rather than crashing |

All errors log to stdout with `session_id`, timestamp, and error type.

---

## 7. Technology Stack

| Component | Technology |
|---|---|
| Backend | Python 3.12, FastAPI, uvicorn |
| WebSockets | FastAPI WebSocket (Starlette) |
| STT | Deepgram Python SDK, nova-2 model, `es` language |
| LLM | OpenAI Python SDK, `gpt-4o` |
| TTS | ElevenLabs Python SDK, `eleven_turbo_v2_5` |
| Audio conversion | ffmpeg (subprocess) |
| Telephony | Twilio Media Streams, TwiML |
| Web UI | Plain HTML + JS (no framework), served by FastAPI |
| Configuration | `.env` file with API keys |

---

## 8. Testing

### Unit tests
- Orchestrator state transitions: interruption logic, history trimming, JSON parsing fallback
- Mock all service clients; no real API calls
- Framework: pytest

### Integration smoke test
- Replay a pre-recorded WAV (Spanish test phrases) directly into the Deepgram client
- Assert transcript contains expected keywords
- Validates Spanish STT quality without Twilio

### End-to-end manual test
1. Start backend locally
2. Expose via ngrok (Twilio needs a public URL for webhooks)
3. Call the Twilio number from a real phone
4. Say *"Buenos días, llamo de Movistar para ofrecerle una nueva tarifa"*
5. Verify: AI responds in Spanish within ~2s; web UI shows Spanish + Ukrainian transcript

### Acceptance criteria

- Median end-to-end response latency ≤ 1.5s
- Deepgram returns correct Spanish transcript for 5 standard test phrases
- GPT-4o returns valid `{reply_es, translation_uk}` JSON for all inputs
- Ukrainian translation appears in web UI within 200ms of GPT-4o response
- Interruption stops AI audio playback within one Twilio media chunk (~20ms)
- ElevenLabs fallback works when primary stream errors

---

## 9. Out of Scope

- Mobile app or call-merging UX (Sub-project 4)
- Database or persistent transcript storage
- Authentication
- Multiple languages
- Autonomous contract/payment handling
- Production deployment, monitoring, GDPR compliance

---

## 10. Estimated Cost Per Call

| Component | 2-min call |
|---|---|
| Deepgram STT | ~$0.01 |
| GPT-4o (5 exchanges) | ~$0.04 |
| ElevenLabs Turbo v2.5 | ~$0.09 |
| GPT-4o Ukrainian translation | ~$0.002 |
| Twilio (2 legs × 2 min) | ~$0.07 |
| **Total** | **~$0.21** |

*Prices as of 2026-10-28 — verify current rates before scaling.*
