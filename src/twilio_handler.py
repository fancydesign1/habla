import logging
import os
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from src.session import CallSession

logger = logging.getLogger(__name__)

router = APIRouter()

active_session: Optional[CallSession] = None


@router.post("/twiml")
async def twiml():
    public_url = os.getenv("PUBLIC_URL", "localhost")
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response><Connect>"
        f'<Stream url="wss://{public_url}/ws/call"/>'
        "</Connect></Response>"
    )
    return Response(content=xml, media_type="text/xml")


@router.websocket("/ws/call")
async def ws_call(websocket: WebSocket):
    global active_session
    await websocket.accept()
    settings = _get_settings()
    if settings is None:
        await websocket.close(code=1011)
        return
    session = CallSession(
        session_id=str(id(websocket)),
        twilio_ws=websocket,
        settings=settings,
    )
    active_session = session
    try:
        await session.run()
    except WebSocketDisconnect:
        pass
    finally:
        await session.cleanup()
        active_session = None


@router.websocket("/ws/ui")
async def ws_ui(websocket: WebSocket):
    await websocket.accept()
    session = active_session
    if session is None:
        await websocket.send_json({"type": "status", "text": "no_active_call"})
        await websocket.close()
        return
    session.add_ui_client(websocket)
    try:
        async for _ in websocket.iter_text():
            pass
    except WebSocketDisconnect:
        pass
    finally:
        session.remove_ui_client(websocket)


def _get_settings():
    try:
        from src.settings import Settings
        return Settings.from_env()
    except Exception as e:
        logger.warning("Settings not available: %s", e)
        return None
