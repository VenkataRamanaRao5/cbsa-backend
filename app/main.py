import json
import logging
import time
import asyncio
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings, configure_logging
from app.engine.ingestion import validate_and_extract
from app.engine.preprocessing import process_event
from app.engine.prototype_engine import compute_prototype_metrics
from app.storage.sqlite_store import SQLiteStore, DB_PATH
from app.websocket_manager import ConnectionManager

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
EVENT_MAP_PATH = BASE_DIR.parent / "EVENT_FLOW_MAP.json"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

behaviour_manager = ConnectionManager()
monitor_manager = ConnectionManager()


@app.on_event("startup")
async def startup_event() -> None:
    app.state.sqlite_store = SQLiteStore(DB_PATH)


def load_event_flow_map() -> dict:
    try:
        with EVENT_MAP_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error("EVENT_FLOW_MAP.json not found at %s", EVENT_MAP_PATH)
        return {"eventFlowMap": {}}


EVENT_FLOW_MAP = load_event_flow_map()


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return JSONResponse(
        content={
            "status": "healthy",
            "active_connections": behaviour_manager.get_connection_count(),
            "monitor_connections": monitor_manager.get_connection_count(),
        }
    )


@app.get("/event-flow-map")
async def event_flow_map():
    return JSONResponse(content=EVENT_FLOW_MAP)


@app.websocket(settings.WEBSOCKET_ENDPOINT)
async def websocket_behaviour_endpoint(websocket: WebSocket):
    await behaviour_manager.connect(websocket)
    client_id = id(websocket)
    logger.info(f"Client {client_id} connected to behaviour stream")

    try:
        while True:
            try:
                data = await websocket.receive_json()
                try:
                    event = validate_and_extract(data)
                    preprocessed = process_event(event)

                    try:
                        await asyncio.to_thread(
                            app.state.sqlite_store.insert_behaviour_log,
                            event.username,
                            event.session_id,
                            event.timestamp,
                            event.event_type,
                            preprocessed.window_vector,
                            preprocessed.short_drift,
                            preprocessed.long_drift,
                            preprocessed.stability_score,
                        )
                    except Exception as log_error:
                        logger.error("Failed to persist behaviour log: %s", log_error, exc_info=True)

                    warmup_state = await asyncio.to_thread(
                        app.state.sqlite_store.collect_warmup_window,
                        event.username,
                        preprocessed.window_vector,
                    )

                    logger.info(
                        "Processed event username=%s session=%s type=%s short_drift=%.4f long_drift=%.4f warmup=%s",
                        event.username,
                        event.session_id,
                        event.event_type,
                        preprocessed.short_drift,
                        preprocessed.long_drift,
                        warmup_state.get("warmup", False),
                    )

                    monitor_message = {
                        "receivedAt": time.time(),
                        "eventType": event.event_type,
                        "payload": data.get("event_data") if isinstance(data, dict) else None,
                        "deviceInfo": (
                            data.get("event_data", {}).get("deviceInfo")
                            if isinstance(data, dict) and isinstance(data.get("event_data"), dict)
                            else None
                        ),
                        "signature": data.get("signature") if isinstance(data, dict) else None,
                        "raw": data,
                    }

                    try:
                        await monitor_manager.broadcast(monitor_message)
                    except Exception as e:
                        logger.error("Failed to broadcast to monitor clients: %s", e, exc_info=True)

                    if bool(warmup_state.get("warmup", False)):
                        response = {
                            "status": "WARMUP",
                            "collected_windows": int(warmup_state.get("collected_windows", 0)),
                        }
                    else:
                        metrics = await asyncio.to_thread(
                            compute_prototype_metrics,
                            app.state.sqlite_store,
                            event.username,
                            preprocessed,
                        )
                        response = {
                            "similarity_score": metrics.similarity_score,
                            "short_drift": metrics.short_drift,
                            "long_drift": metrics.long_drift,
                            "stability_score": metrics.stability_score,
                            "matched_prototype_id": metrics.matched_prototype_id,
                        }

                    await behaviour_manager.send_personal_message(response, websocket)
                except ValueError as e:
                    logger.warning("Validation error from client %s: %s", client_id, e)
                    await behaviour_manager.send_personal_message({"error": str(e)}, websocket)
                    
            except ValueError as e:
                logger.error(f"JSON decode error from client {client_id}: {e}")
                error_response = {
                    "status": "error",
                    "message": "Malformed JSON",
                }
                await behaviour_manager.send_personal_message(error_response, websocket)
                
    except WebSocketDisconnect:
        behaviour_manager.disconnect(websocket)
        logger.info(f"Client {client_id} disconnected normally")
        
    except Exception as e:
        logger.error(f"Unexpected error with client {client_id}: {e}", exc_info=True)
        behaviour_manager.disconnect(websocket)


@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    await monitor_manager.connect(websocket)
    client_id = id(websocket)
    logger.info(f"Monitor client {client_id} connected")

    try:
        while True:
            try:
                await websocket.receive_text()
            except ValueError:
                pass
    except WebSocketDisconnect:
        monitor_manager.disconnect(websocket)
        logger.info(f"Monitor client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"Unexpected error with monitor client {client_id}: {e}", exc_info=True)
        monitor_manager.disconnect(websocket)
