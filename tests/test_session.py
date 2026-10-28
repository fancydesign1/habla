import pytest
from unittest.mock import AsyncMock, MagicMock
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
    assert "never accept" in SYSTEM_PROMPT.lower()
    assert "reply_es" in SYSTEM_PROMPT
    assert "translation_uk" in SYSTEM_PROMPT
