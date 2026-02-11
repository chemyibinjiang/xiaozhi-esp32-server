import argparse
import asyncio
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, Tuple

from aiohttp import web

from config.config_loader import get_project_dir
from config.logger import setup_logging
from config.settings import load_config
from core.providers.llm.codex.codex import LLMProvider as CodexLLMProvider

TAG = __name__
logger = setup_logging()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_store_path(config: Dict, cli_store: str) -> str:
    if cli_store:
        store_path = cli_store
    else:
        store_path = str(
            config.get("codex_app", {}).get(
                "session_store",
                "data/codex_app/session_registry.json",
            )
        ).strip()
        if not store_path:
            store_path = "data/codex_app/session_registry.json"

    if os.path.isabs(store_path):
        return store_path
    return os.path.join(get_project_dir(), store_path)


def _resolve_codex_llm_name(config: Dict, cli_llm_name: str) -> Tuple[str, Dict]:
    llm_map = config.get("LLM", {}) or {}
    if not isinstance(llm_map, dict) or not llm_map:
        raise ValueError("LLM config is empty")

    preferred = str(cli_llm_name or "").strip()
    if not preferred:
        preferred = str(config.get("codex_app", {}).get("llm_name", "")).strip()

    if preferred:
        llm_cfg = llm_map.get(preferred)
        if not isinstance(llm_cfg, dict):
            raise ValueError(f"LLM '{preferred}' not found in config")
        if str(llm_cfg.get("type", "")).strip() != "codex":
            raise ValueError(f"LLM '{preferred}' type must be 'codex'")
        return preferred, llm_cfg

    selected_name = str(config.get("selected_module", {}).get("LLM", "")).strip()
    if selected_name:
        selected_cfg = llm_map.get(selected_name)
        if isinstance(selected_cfg, dict) and str(selected_cfg.get("type", "")).strip() == "codex":
            return selected_name, selected_cfg

    for name, llm_cfg in llm_map.items():
        if isinstance(llm_cfg, dict) and str(llm_cfg.get("type", "")).strip() == "codex":
            return str(name), llm_cfg

    raise ValueError("No LLM with type 'codex' found")


class SessionStore:
    def __init__(self, store_path: str):
        self.store_path = store_path
        self._lock = threading.Lock()

    def _load(self) -> Dict:
        if not os.path.exists(self.store_path):
            return {"bindings": {}}

        with open(self.store_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {"bindings": {}}
        bindings = raw.get("bindings")
        if not isinstance(bindings, dict):
            raw["bindings"] = {}
        return raw

    def _save(self, data: Dict):
        parent = os.path.dirname(self.store_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def resolve_or_create(self, device_id: str, user_id: str) -> Dict:
        normalized_device = str(device_id or "").strip()
        normalized_user = str(user_id or "").strip() or "test"
        if not normalized_device:
            raise ValueError("device_id is required")

        key = f"{normalized_device}::{normalized_user}"
        now = _utc_now()

        with self._lock:
            data = self._load()
            bindings = data.setdefault("bindings", {})
            item = bindings.get(key)

            created = False
            if not isinstance(item, dict):
                created = True
                chat_session_id = str(uuid.uuid4())
                model_session_key = f"codex:{chat_session_id}"
                item = {
                    "device_id": normalized_device,
                    "user_id": normalized_user,
                    "chat_session_id": chat_session_id,
                    "model_session_key": model_session_key,
                    "created_at": now,
                    "updated_at": now,
                }
            else:
                item["updated_at"] = now

            bindings[key] = item
            self._save(data)

        return {
            "device_id": normalized_device,
            "user_id": normalized_user,
            "chat_session_id": str(item["chat_session_id"]),
            "model_session_key": str(item["model_session_key"]),
            "created": created,
        }


class CodexAppService:
    def __init__(self, config: Dict, llm_name: str, llm_cfg: Dict, store_path: str):
        self.config = config
        self.llm_name = llm_name
        self.store = SessionStore(store_path)
        self.llm = CodexLLMProvider(llm_cfg)

    async def handle_health(self, request: web.Request):
        return web.json_response(
            {
                "ok": True,
                "llm_name": self.llm_name,
            }
        )

    async def handle_session_resolve_or_create(self, request: web.Request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {"success": False, "message": "invalid JSON body"},
                status=400,
            )

        device_id = str(body.get("device_id", "")).strip()
        user_id = str(body.get("user_id", "")).strip()
        if not user_id:
            user_id = str(
                self.config.get("session_registry", {}).get("default_user_id", "test")
            ).strip()
        if not user_id:
            user_id = "test"

        if not device_id:
            return web.json_response(
                {"success": False, "message": "device_id is required"},
                status=400,
            )

        try:
            result = await asyncio.to_thread(
                self.store.resolve_or_create,
                device_id,
                user_id,
            )
            result["success"] = True
            return web.json_response(result)
        except Exception as exc:
            logger.bind(tag=TAG).error(f"resolve_or_create failed: {exc}")
            return web.json_response(
                {"success": False, "message": str(exc)},
                status=500,
            )

    def _collect_codex_response(self, session_id: str, dialogue, kwargs: Dict):
        events = []
        text_parts = []

        for token in self.llm.response(session_id, dialogue, **kwargs):
            if isinstance(token, dict):
                events.append(token)
                continue
            if isinstance(token, tuple):
                if token and token[0]:
                    text_parts.append(str(token[0]))
                continue
            if token is None:
                continue
            text_parts.append(str(token))

        return {
            "session_id": session_id,
            "text": "".join(text_parts),
            "events": events,
        }

    async def handle_codex_response(self, request: web.Request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {"success": False, "message": "invalid JSON body"},
                status=400,
            )

        session_id = str(body.get("session_id", "")).strip()
        dialogue = body.get("dialogue", [])
        kwargs = body.get("kwargs", {}) or {}

        if not isinstance(dialogue, list):
            return web.json_response(
                {"success": False, "message": "dialogue must be a list"},
                status=400,
            )
        if not isinstance(kwargs, dict):
            kwargs = {}

        if not session_id:
            session_id = "utility"

        try:
            result = await asyncio.to_thread(
                self._collect_codex_response,
                session_id,
                dialogue,
                kwargs,
            )
            result["success"] = True
            return web.json_response(result)
        except Exception as exc:
            logger.bind(tag=TAG).error(f"codex response failed: {exc}")
            return web.json_response(
                {
                    "success": False,
                    "message": str(exc),
                },
                status=500,
            )

    def close(self):
        sessions = getattr(self.llm, "_sessions", {})
        if isinstance(sessions, dict):
            for session in sessions.values():
                try:
                    session.close()
                except Exception:
                    pass

    def create_app(self):
        app = web.Application()
        app.add_routes(
            [
                web.get("/health", self.handle_health),
                web.post(
                    "/session/resolve_or_create",
                    self.handle_session_resolve_or_create,
                ),
                web.post(
                    "/v1/llm/codex/response",
                    self.handle_codex_response,
                ),
            ]
        )

        async def _on_shutdown(_app):
            self.close()

        app.on_shutdown.append(_on_shutdown)
        return app


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone Codex app service for xiaozhi-server",
    )
    parser.add_argument("--host", default="", help="HTTP host, default from config")
    parser.add_argument("--port", type=int, default=0, help="HTTP port, default from config")
    parser.add_argument(
        "--llm-name",
        default="",
        help="LLM key in config['LLM'] to run (must be type=codex)",
    )
    parser.add_argument(
        "--session-store",
        default="",
        help="Session store JSON path, relative to project root if not absolute",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    codex_app_cfg = config.get("codex_app", {}) or {}

    host = str(args.host or codex_app_cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1"
    port = int(args.port or codex_app_cfg.get("port", 9001))
    store_path = _resolve_store_path(config, args.session_store)
    llm_name, llm_cfg = _resolve_codex_llm_name(config, args.llm_name)

    service = CodexAppService(config, llm_name, llm_cfg, store_path)
    logger.bind(tag=TAG).info(
        "codex app service starting: host={} port={} llm={} store={}",
        host,
        port,
        llm_name,
        store_path,
    )

    app = service.create_app()
    web.run_app(app, host=host, port=port)


if __name__ == "__main__":
    main()
