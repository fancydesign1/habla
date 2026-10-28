# Habla — Real-Time AI Spanish Phone Call Assistant (Telephony PoC)

An AI assistant that answers Spanish phone calls autonomously via Twilio, transcribes them with Deepgram, generates bilingual replies with GPT-4o, and speaks back with ElevenLabs. A live bilingual (Spanish + Ukrainian) transcript streams to your browser in real time.

## Prerequisites

- Python 3.9+
- [ngrok](https://ngrok.com/) (free tier is fine)
- A [Twilio](https://twilio.com/) account with a phone number
- API keys: Deepgram, OpenAI, ElevenLabs

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in your keys in .env
```

**.env.example** (create this file):
```
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_PHONE_NUMBER=+1xxxxxxxxxx
DEEPGRAM_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ELEVENLABS_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ELEVENLABS_VOICE_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
PUBLIC_URL=your-ngrok-subdomain.ngrok.io
```

To find an ElevenLabs voice ID: visit [elevenlabs.io/voices](https://elevenlabs.io/voices), pick a Spanish-accent voice, and copy its ID.

## Running locally

```bash
# Terminal 1 — start the app
uvicorn src.main:app --reload --port 8000

# Terminal 2 — expose to internet
ngrok http 8000
```

Copy the ngrok HTTPS URL (e.g. `https://abc123.ngrok.io`) and update `PUBLIC_URL` in your `.env` (without `https://`).

## Twilio webhook

1. Go to Twilio Console → Phone Numbers → your number → Voice configuration
2. Set **A call comes in** → Webhook → `https://<your-ngrok>/twiml` (HTTP POST)
3. Save

## Live transcript

Open `http://localhost:8000` in your browser. The two-column transcript updates in real time as calls happen.

## End-to-end test

1. Start the app and ngrok
2. Update Twilio webhook to your ngrok URL
3. Call your Twilio number from a phone
4. Speak Spanish — the AI replies in Spanish
5. Watch the live transcript in your browser (Spanish left, Ukrainian right)

## Running unit tests

```bash
pytest tests/ --ignore=tests/test_integration_stt.py
```

## Running integration test (requires live Deepgram key)

```bash
DEEPGRAM_API_KEY=your_key pytest tests/test_integration_stt.py -v
```
