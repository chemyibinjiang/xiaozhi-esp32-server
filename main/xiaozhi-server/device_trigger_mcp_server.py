#!/usr/bin/env python3
"""
MCP server for triggering xiaozhi device actions through app.py HTTP endpoints.

Exposed tools:
  - xiaozhi_list_sessions
  - xiaozhi_debug_route_context
  - xiaozhi_take_photo
  - xiaozhi_preview_local_file

Design goal:
  Route tool calls to the correct device by device_id.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from datetime import datetime
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from mcp.server.fastmcp import Context, FastMCP

from trigger_preview_local_file import (
    trigger_preview_local_file as do_preview_local_file,
)
from trigger_take_photo import (
    build_photo_name,
    build_question_with_photo_name,
    list_sessions as list_device_sessions,
    trigger_take_photo as do_take_photo,
)

LOGGER = logging.getLogger("device_trigger_mcp_server")


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _extract_header_value_from_request(request: Dict[str, Any], header_name: str) -> str:
    target = _norm(header_name).lower()
    if not target:
        return ""

    direct = _norm(request.get(header_name, "") or request.get(header_name.lower(), "") or request.get(header_name.upper(), ""))
    if direct:
        return direct

    def _scan_headers(raw_headers: Any) -> str:
        if isinstance(raw_headers, dict):
            for key, value in raw_headers.items():
                if _decode_scalar(key).lower() == target:
                    return _norm(_decode_scalar(value))
            return ""
        if isinstance(raw_headers, (list, tuple)):
            for item in raw_headers:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    key = _decode_scalar(item[0]).lower()
                    if key == target:
                        return _norm(_decode_scalar(item[1]))
            return ""
        return ""

    for candidate in (request.get("headers"), _to_mapping(request.get("scope", {})).get("headers")):
        hit = _scan_headers(candidate)
        if hit:
            return hit
    return ""


def _normalize_streamable_http_path(path: str) -> str:
    """
    Build a redirect-free Starlette route for streamable-http.

    Why:
      Some clients call /mcp while others call /mcp/.
      With a strict "/mcp" route, "/mcp/" may hit a 307 redirect without
      Content-Type, which can break strict MCP clients.

    Behavior:
      - If caller already provides a Starlette path pattern (contains "{}"),
        keep it as-is.
      - Otherwise convert "/mcp" -> "/mcp{_tail:path}" so both /mcp and /mcp/
        are handled directly without redirect.
    """
    normalized = _norm(path) or "/mcp"
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    if "{" in normalized and "}" in normalized:
        return normalized

    if normalized == "/":
        return "/{_tail:path}"

    return f"{normalized.rstrip('/')}" + "{_tail:path}"


def _sanitize_device_for_path(device_id: str) -> str:
    value = _norm(device_id)
    if not value:
        return "unknown"
    chars: List[str] = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            chars.append(ch)
            continue
        if ch == ":":
            chars.append("-")
            continue
        chars.append("_")
    safe = "".join(chars).strip("._-")
    return safe or "unknown"


def _to_mapping(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            data = model_dump()
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    as_dict = getattr(obj, "__dict__", None)
    if isinstance(as_dict, dict):
        return as_dict
    return {}


def _add_unique(values: List[str], value: Any):
    normalized = _norm(value)
    if normalized and normalized not in values:
        values.append(normalized)


def _find_by_field(connections: List[Dict[str, Any]], field: str, value: str) -> List[Dict[str, Any]]:
    target = _norm(value)
    if not target:
        return []
    return [item for item in connections if _norm(item.get(field, "")) == target]


def _log_route_resolution(action: str, resolved: Dict[str, Any]) -> None:
    resolved_by = _norm(resolved.get("resolved_by", ""))
    route = resolved.get("connection", {})
    payload = (
        f"action={_norm(action) or 'unknown'} "
        f"resolved_by={resolved_by or 'unknown'} "
        f"session_id={_norm(route.get('session_id', ''))} "
        f"device_id={_norm(route.get('device_id', ''))} "
        f"chat_session_id={_norm(route.get('chat_session_id', ''))} "
        f"model_session_key={_norm(route.get('model_session_key', ''))}"
    )

    if resolved_by.startswith("auto_"):
        LOGGER.info("route_resolution mode=context_candidates %s", payload)
        return

    if resolved_by.startswith("explicit_"):
        LOGGER.info("route_resolution mode=explicit_selector %s", payload)
        return

    LOGGER.info("route_resolution mode=unknown %s", payload)


def _configure_logging(log_level: str) -> None:
    level = getattr(logging, _norm(log_level).upper(), logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )
    else:
        root_logger.setLevel(level)
    LOGGER.setLevel(level)


class RouteResolver:
    def __init__(self, base_url: str, list_timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.list_timeout = int(list_timeout)
        self._initialized_candidate_keys: Set[str] = set()
        self._candidate_lock = Lock()

    def list_sessions(self) -> Dict[str, Any]:
        return list_device_sessions(self.base_url, timeout=self.list_timeout)

    def _extract_transport_session_id(
        self,
        *,
        ctx: Optional[Context] = None,
        request: Optional[Dict[str, Any]] = None,
        session: Any = None,
    ) -> str:
        if request is None and ctx is not None:
            request_context = getattr(ctx, "request_context", None)
            request = _to_mapping(getattr(request_context, "request", None))
        if session is None and ctx is not None:
            request_context = getattr(ctx, "request_context", None)
            session = getattr(request_context, "session", None)

        request_map = request or {}
        transport_id = _extract_header_value_from_request(request_map, "mcp-session-id")
        if transport_id:
            return transport_id

        for attr_name in ("id", "session_id", "mcp_session_id"):
            value = _norm(getattr(session, attr_name, ""))
            if value:
                return value
        return ""


    def _context_snapshot(self, ctx: Optional[Context]) -> Dict[str, Any]:
        if ctx is None:
            return {"has_context": False}

        snap: Dict[str, Any] = {
            "has_context": True,
            "request_id": _norm(getattr(ctx, "request_id", "")),
            "client_id": _norm(getattr(ctx, "client_id", "")),
        }

        request_context = getattr(ctx, "request_context", None)
        meta = _to_mapping(getattr(request_context, "meta", None))
        request = _to_mapping(getattr(request_context, "request", None))
        session = getattr(request_context, "session", None)
        request_scope = _to_mapping(request.get("scope", {}))
        mcp_session_id = self._extract_transport_session_id(request=request, session=session)

        session_probe: Dict[str, Any] = {"type": type(session).__name__ if session is not None else ""}
        for key in ("id", "session_id", "mcp_session_id", "client_id"):
            value = _norm(getattr(session, key, ""))
            if value:
                session_probe[key] = value

        client_params = _to_mapping(getattr(session, "client_params", None))
        if client_params:
            client_info = _to_mapping(client_params.get("clientInfo", {}))
            session_probe["protocol_version"] = _norm(client_params.get("protocolVersion", ""))
            session_probe["client_info"] = {
                "name": _norm(client_info.get("name", "")),
                "version": _norm(client_info.get("version", "")),
                "title": _norm(client_info.get("title", "")),
            }

        snap["meta"] = {
            "progressToken": _norm(meta.get("progressToken", "")),
            "client_id": _norm(meta.get("client_id", "") or meta.get("clientId", "")),
        }
        snap["request"] = {
            "method": _norm(request_scope.get("method", "") or request.get("method", "")),
            "path": _norm(request_scope.get("path", "") or request.get("path", "")),
            "client": _json_safe(request_scope.get("client", request.get("client", ""))),
            "mcp_session_id": mcp_session_id,
        }
        snap["session"] = session_probe
        return snap

    def _context_candidates(self, ctx: Optional[Context]) -> Dict[str, List[str]]:
        candidates: Dict[str, List[str]] = {
            "device_ids": [],
        }

        _add_unique(candidates["device_ids"], os.getenv("XIAOZHI_DEVICE_ID", ""))

        if ctx is None:
            self._emit_candidates_initialized(ctx, candidates)
            return candidates

        request_context = getattr(ctx, "request_context", None)
        meta = _to_mapping(getattr(request_context, "meta", None))
        request = _to_mapping(getattr(request_context, "request", None))
        session = getattr(request_context, "session", None)
        session_map = _to_mapping(session)
        client_params = _to_mapping(getattr(session, "client_params", None))

        potential_sources = [meta, request, session_map, client_params]
        for source in potential_sources:
            if not source:
                continue
            _add_unique(candidates["device_ids"], source.get("device_id", ""))
            _add_unique(candidates["device_ids"], source.get("deviceId", ""))

        self._emit_candidates_initialized(ctx, candidates)
        return candidates

    def _candidate_init_key(self, ctx: Optional[Context], candidates: Dict[str, List[str]]) -> str:
        if ctx is None:
            return "no_context"

        request_context = getattr(ctx, "request_context", None)
        request = _to_mapping(getattr(request_context, "request", None))
        session = getattr(request_context, "session", None)
        mcp_session_id = self._extract_transport_session_id(request=request, session=session)
        if mcp_session_id:
            return f"transport_header:mcp-session-id:{mcp_session_id}"

        for attr_name in ("session_id", "id", "mcp_session_id", "client_id"):
            value = _norm(getattr(session, attr_name, ""))
            if value:
                return f"session_attr:{attr_name}:{value}"

        client_id = _norm(getattr(ctx, "client_id", ""))
        if client_id:
            return f"ctx_client_id:{client_id}"

        for bucket in ("device_ids",):
            values = candidates.get(bucket, [])
            if values:
                return f"{bucket}:{values[0]}"

        return "unknown_context"

    def _emit_candidates_initialized(self, ctx: Optional[Context], candidates: Dict[str, List[str]]) -> None:
        init_key = self._candidate_init_key(ctx, candidates)
        with self._candidate_lock:
            if init_key in self._initialized_candidate_keys:
                return
            self._initialized_candidate_keys.add(init_key)

        payload = {
            "event": "candidates_initialized",
            "init_key": init_key,
            "context_snapshot": self._context_snapshot(ctx),
            "candidates": candidates,
        }
        payload_text = json.dumps(_json_safe(payload), ensure_ascii=False)
        print(payload_text, flush=True)
        LOGGER.info("candidates_initialized %s", payload_text)

    @staticmethod
    def _single_or_error(matches: List[Dict[str, Any]], reason: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            return None, f"multiple matches for {reason}"
        return None, None

    def resolve_target(
        self,
        *,
        device_id: str = "",
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        sessions_resp = self.list_sessions()
        if not sessions_resp.get("success"):
            raise RuntimeError(f"failed to list sessions: {sessions_resp}")

        connections = sessions_resp.get("connections", [])
        if not isinstance(connections, list):
            connections = []
        if not connections:
            raise RuntimeError("no online sessions")

        explicit = {
            "device_id": _norm(device_id),
        }
        candidates = self._context_candidates(ctx)
        context_snapshot = self._context_snapshot(ctx)

        # 1) explicit routing first
        if explicit["device_id"]:
            matches = _find_by_field(connections, "device_id", explicit["device_id"])
            chosen, error = self._single_or_error(matches, f"device_id={explicit['device_id']}")
            if chosen:
                return self._build_resolution(chosen, "explicit_device_id", explicit, candidates, context_snapshot, len(connections))
            if error:
                raise RuntimeError(error)
            raise RuntimeError(f"device_id not online: {explicit['device_id']}")

        # 2) automatic context-driven resolution
        ordered_lookups: List[Tuple[str, str, Iterable[str]]] = [
            ("device_id", "device_id", candidates["device_ids"]),
        ]

        for source_name, field_name, values in ordered_lookups:
            for value in values:
                matches = _find_by_field(connections, field_name, value)
                chosen, error = self._single_or_error(matches, f"{source_name}->{field_name}={value}")
                if chosen:
                    return self._build_resolution(
                        chosen,
                        f"auto_{source_name}_to_{field_name}",
                        explicit,
                        candidates,
                        context_snapshot,
                        len(connections),
                    )
                if error:
                    raise RuntimeError(error)

        raise RuntimeError(
            "unable to resolve target connection automatically; provide device_id"
        )

    @staticmethod
    def _build_resolution(
        connection: Dict[str, Any],
        resolved_by: str,
        explicit: Dict[str, str],
        candidates: Dict[str, List[str]],
        context_snapshot: Dict[str, Any],
        online_count: int,
    ) -> Dict[str, Any]:
        return {
            "resolved_by": resolved_by,
            "online_count": int(online_count),
            "explicit_input": explicit,
            "auto_candidates": candidates,
            "context_snapshot": context_snapshot,
            "connection": {
                "session_id": _norm(connection.get("session_id", "")),
                "transport_session_id": _norm(connection.get("transport_session_id", "")),
                "device_id": _norm(connection.get("device_id", "")),
                "chat_session_id": _norm(connection.get("chat_session_id", "")),
                "model_session_key": _norm(connection.get("model_session_key", "")),
                "user_id": _norm(connection.get("user_id", "")),
                "client_ip": _norm(connection.get("client_ip", "")),
                "mcp_ready": bool(connection.get("mcp_ready", False)),
                "websocket_alive": bool(connection.get("websocket_alive", False)),
            },
        }


class PhotoPathTracker:
    def __init__(
        self,
        *,
        vision_dir: str,
        by_device_dir: str,
        detect_interval_ms: int = 200,
        enable_mirror: bool = True,
    ):
        self.vision_dir = os.path.abspath(_norm(vision_dir) or os.path.join("data", "vision"))
        self.by_device_dir = os.path.abspath(
            _norm(by_device_dir) or os.path.join("data", "vision_by_device")
        )
        self.detect_interval_s = max(0.05, int(detect_interval_ms) / 1000.0)
        self.enable_mirror = bool(enable_mirror)
        self._lock = Lock()
        os.makedirs(self.vision_dir, exist_ok=True)
        if self.enable_mirror:
            os.makedirs(self.by_device_dir, exist_ok=True)

    def _candidate_prefixes(self, device_id: str) -> List[str]:
        value = _norm(device_id)
        if not value:
            return []
        ordered: List[str] = []
        for item in (value.replace(":", "-"), _sanitize_device_for_path(value)):
            if item and item not in ordered:
                ordered.append(item)
        return ordered

    def _list_files_in_dir(self, directory: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if not os.path.isdir(directory):
            return items
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue
                    try:
                        stat = entry.stat()
                    except FileNotFoundError:
                        continue
                    items.append(
                        {
                            "local_path": os.path.abspath(entry.path),
                            "mtime": float(stat.st_mtime),
                            "size": int(stat.st_size),
                            "file_name": entry.name,
                        }
                    )
        except FileNotFoundError:
            return items
        return items

    def _collect_shared_candidates(self, device_id: str) -> List[Dict[str, Any]]:
        prefixes = self._candidate_prefixes(device_id)
        if not prefixes:
            return []
        items: List[Dict[str, Any]] = []
        for item in self._list_files_in_dir(self.vision_dir):
            name = _norm(item.get("file_name", ""))
            if not any(name.startswith(f"{prefix}_") for prefix in prefixes):
                continue
            item["source"] = "vision_dir"
            items.append(item)
        return items

    def _collect_mirrored_candidates(self, device_id: str) -> List[Dict[str, Any]]:
        safe_device = _sanitize_device_for_path(device_id)
        device_dir = os.path.join(self.by_device_dir, safe_device)
        items: List[Dict[str, Any]] = []
        for item in self._list_files_in_dir(device_dir):
            item["source"] = "by_device_dir"
            items.append(item)
        return items

    def _format_photo_meta(self, device_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
        mtime = float(item.get("mtime", 0.0))
        local_path = os.path.abspath(_norm(item.get("local_path", "")))
        file_name = _norm(item.get("file_name", os.path.basename(local_path)))
        return {
            "device_id": _norm(device_id),
            "local_path": local_path,
            "file_name": file_name,
            "source": _norm(item.get("source", "")),
            "size": int(item.get("size", 0)),
            "mtime": mtime,
            "mtime_iso": datetime.fromtimestamp(mtime).isoformat(timespec="seconds")
            if mtime > 0
            else "",
        }

    def find_latest(self, device_id: str) -> Optional[Dict[str, Any]]:
        target_device = _norm(device_id)
        if not target_device:
            return None
        candidates = self._collect_shared_candidates(target_device) + self._collect_mirrored_candidates(target_device)
        if not candidates:
            return None
        best = max(candidates, key=lambda item: float(item.get("mtime", 0.0)))
        return self._format_photo_meta(target_device, best)

    def wait_for_new_photo(
        self,
        *,
        device_id: str,
        baseline: Optional[Dict[str, Any]],
        max_wait_seconds: float,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        wait_seconds = max(0.0, float(max_wait_seconds))
        baseline_path = _norm((baseline or {}).get("local_path", ""))
        baseline_mtime = float((baseline or {}).get("mtime", 0.0))

        deadline = time.time() + wait_seconds
        while True:
            latest = self.find_latest(device_id)
            if latest:
                latest_path = _norm(latest.get("local_path", ""))
                latest_mtime = float(latest.get("mtime", 0.0))
                if not baseline_path:
                    return latest, "first_detected_for_device"
                if latest_path != baseline_path or latest_mtime > baseline_mtime + 1e-6:
                    return latest, "new_file_after_take_photo"

            if time.time() >= deadline:
                break
            time.sleep(self.detect_interval_s)

        latest = self.find_latest(device_id)
        if latest:
            return latest, "timeout_latest_snapshot"
        return None, "not_found_after_timeout"

    def mirror_to_device_dir(self, device_id: str, local_path: str) -> Tuple[str, str]:
        if not self.enable_mirror:
            return "", "mirror_disabled"
        src = os.path.abspath(_norm(local_path))
        if not src or not os.path.isfile(src):
            return "", "source_missing"

        safe_device = _sanitize_device_for_path(device_id)
        dst_dir = os.path.join(self.by_device_dir, safe_device)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.abspath(os.path.join(dst_dir, os.path.basename(src)))

        if src == dst:
            return dst, "already_in_device_dir"

        with self._lock:
            src_mtime = os.path.getmtime(src)
            if os.path.exists(dst):
                dst_mtime = os.path.getmtime(dst)
                if dst_mtime >= src_mtime:
                    return dst, "already_exists_newer_or_same"
            shutil.copy2(src, dst)
        return dst, "copied"

    def resolve_photo_after_take_photo(
        self,
        *,
        device_id: str,
        requested_photo_name: str,
        baseline: Optional[Dict[str, Any]],
        detect_timeout: float,
    ) -> Dict[str, Any]:
        latest, detected_by = self.wait_for_new_photo(
            device_id=device_id,
            baseline=baseline,
            max_wait_seconds=detect_timeout,
        )
        if not latest:
            return {
                "found": False,
                "device_id": _norm(device_id),
                "requested_photo_name": _norm(requested_photo_name),
                "detected_by": detected_by,
            }

        mirrored_path, mirror_state = self.mirror_to_device_dir(
            _norm(device_id),
            _norm(latest.get("local_path", "")),
        )
        latest["found"] = True
        latest["requested_photo_name"] = _norm(requested_photo_name)
        latest["detected_by"] = detected_by
        latest["mirrored_path"] = mirrored_path
        latest["mirror_state"] = mirror_state
        return latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCP server for xiaozhi device triggers")
    parser.add_argument("--server", default=os.getenv("XIAOZHI_SERVER", "http://127.0.0.1:8003"))
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default=os.getenv("MCP_TRANSPORT", "stdio"),
    )
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "8000")))
    parser.add_argument("--mount-path", default=os.getenv("MCP_MOUNT_PATH", "/"))
    parser.add_argument("--sse-path", default=os.getenv("MCP_SSE_PATH", "/sse"))
    parser.add_argument("--message-path", default=os.getenv("MCP_MESSAGE_PATH", "/messages/"))
    parser.add_argument("--streamable-http-path", default=os.getenv("MCP_STREAMABLE_HTTP_PATH", "/mcp"))
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=os.getenv("MCP_LOG_LEVEL", "INFO"),
    )
    parser.add_argument("--name", default=os.getenv("MCP_SERVER_NAME", "xiaozhi-device-trigger"))
    parser.add_argument("--list-timeout", type=int, default=10)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--default-take-photo-tool-name", default="self.camera.take_photo")
    parser.add_argument("--default-preview-tool-name", default="self.screen.preview_image")
    parser.add_argument(
        "--default-time-format",
        default="%Y%m%d_%H%M%S",
        help="timestamp format used when photo_name is passed with append_timestamp=true",
    )
    parser.add_argument(
        "--vision-dir",
        default=os.getenv("XIAOZHI_VISION_DIR", os.path.join("data", "vision")),
        help="vision image directory written by app.py",
    )
    parser.add_argument(
        "--vision-by-device-dir",
        default=os.getenv(
            "XIAOZHI_VISION_BY_DEVICE_DIR",
            os.path.join("data", "vision_by_device"),
        ),
        help="mirror directory grouped by device_id",
    )
    parser.add_argument(
        "--photo-detect-timeout",
        type=float,
        default=float(os.getenv("XIAOZHI_PHOTO_DETECT_TIMEOUT", "8")),
        help="seconds to wait for a new photo file after take_photo",
    )
    parser.add_argument(
        "--photo-detect-interval-ms",
        type=int,
        default=int(os.getenv("XIAOZHI_PHOTO_DETECT_INTERVAL_MS", "200")),
        help="poll interval in milliseconds when waiting for a new photo file",
    )
    parser.add_argument(
        "--disable-photo-mirror",
        action="store_true",
        help="disable mirroring photos to vision_by_device/<device_id>/",
    )
    return parser.parse_args()


def build_server(args: argparse.Namespace) -> FastMCP:
    streamable_http_path = _normalize_streamable_http_path(args.streamable_http_path)

    resolver = RouteResolver(
        base_url=args.server,
        list_timeout=args.list_timeout,
    )
    photo_tracker = PhotoPathTracker(
        vision_dir=args.vision_dir,
        by_device_dir=args.vision_by_device_dir,
        detect_interval_ms=args.photo_detect_interval_ms,
        enable_mirror=not bool(args.disable_photo_mirror),
    )

    mcp = FastMCP(
        name=args.name,
        host=args.host,
        port=args.port,
        mount_path=args.mount_path,
        sse_path=args.sse_path,
        message_path=args.message_path,
        streamable_http_path=streamable_http_path,
        json_response=True,
        log_level=args.log_level,
    )

    async def _list_tools_with_candidate_init_logging() -> List[Any]:
        # Trigger candidate initialization logging as soon as client asks for tools
        # (this is the earliest stable point after MCP session initialize).
        try:
            ctx = mcp.get_context()
            resolver._context_candidates(ctx)
        except Exception as exc:
            LOGGER.warning("failed to initialize candidates on list_tools: %s", exc)
        return await FastMCP.list_tools(mcp)

    mcp._mcp_server.list_tools()(_list_tools_with_candidate_init_logging)

    @mcp.tool(name="xiaozhi_list_sessions", description="List online xiaozhi websocket sessions.")
    def xiaozhi_list_sessions() -> Dict[str, Any]:
        return resolver.list_sessions()

    @mcp.tool(
        name="xiaozhi_debug_route_context",
        description="Show MCP context-derived routing candidates for current tool call.",
    )
    def xiaozhi_debug_route_context(ctx: Context) -> Dict[str, Any]:
        return {
            "success": True,
            "context_snapshot": resolver._context_snapshot(ctx),
            "auto_candidates": resolver._context_candidates(ctx),
            "server": args.server,
        }

    @mcp.tool(
        name="xiaozhi_take_photo",
        description=(
            "Take a photo on the resolved target device. "
            "device_id is optional and defaults to null; "
            "the server auto-resolves from MCP context and only needs explicit device_id when routing is ambiguous."
        ),
    )
    def xiaozhi_take_photo(
        question: str = "Please take a photo.",
        photo_name: str = "",
        append_timestamp: bool = True,
        time_format: str = "",
        device_id: Optional[str] = None,
        tool_name: str = "",
        timeout: int = 90,
        request_timeout: int = 120,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        try:
            resolved = resolver.resolve_target(
                device_id=device_id,
                ctx=ctx,
            )
            _log_route_resolution("xiaozhi_take_photo", resolved)
            route = resolved["connection"]
            baseline_photo = photo_tracker.find_latest(route["device_id"])

            final_photo_name = _norm(photo_name)
            if final_photo_name and append_timestamp:
                final_photo_name = build_photo_name(
                    final_photo_name,
                    _norm(time_format) or args.default_time_format,
                )

            final_question = build_question_with_photo_name(question, final_photo_name)
            result = do_take_photo(
                args.server,
                session_id=route["session_id"],
                device_id=route["device_id"],
                question=final_question,
                photo_name=final_photo_name,
                tool_name=_norm(tool_name) or args.default_take_photo_tool_name,
                tool_timeout=int(timeout),
                request_timeout=int(request_timeout),
            )
            photo_meta = photo_tracker.resolve_photo_after_take_photo(
                device_id=route["device_id"],
                requested_photo_name=final_photo_name,
                baseline=baseline_photo,
                detect_timeout=float(args.photo_detect_timeout),
            )
            return {
                "success": bool(result.get("success", False)),
                "route": resolved,
                "requested_photo_name": final_photo_name,
                "photo_meta": photo_meta,
                "result": result,
            }
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    @mcp.tool(
        name="xiaozhi_get_latest_photo",
        description=(
            "Get latest local photo metadata for target device. "
            "device_id is optional and defaults to null; "
            "auto-resolution uses MCP context."
        ),
    )
    def xiaozhi_get_latest_photo(
        device_id: Optional[str] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        try:
            resolved = resolver.resolve_target(
                device_id=device_id,
                ctx=ctx,
            )
            route = resolved["connection"]
            latest = photo_tracker.find_latest(route["device_id"])
            if not latest:
                return {
                    "success": False,
                    "route": resolved,
                    "message": "no local photo found for resolved device",
                }
            return {
                "success": True,
                "route": resolved,
                "photo_meta": latest,
            }
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    @mcp.tool(
        name="xiaozhi_preview_local_file",
        description=(
            "Preview a local image file on the resolved target device screen. "
            "device_id is optional and defaults to null; "
            "the server auto-resolves from MCP context and only needs explicit device_id when routing is ambiguous."
        ),
    )
    def xiaozhi_preview_local_file(
        file_path: str = "",
        device_id: Optional[str] = None,
        tool_name: str = "",
        timeout: int = 90,
        request_timeout: int = 120,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        try:
            resolved = resolver.resolve_target(
                device_id=device_id,
                ctx=ctx,
            )
            _log_route_resolution("xiaozhi_preview_local_file", resolved)
            route = resolved["connection"]
            resolved_file_path = _norm(file_path)
            latest_meta: Optional[Dict[str, Any]] = None
            if not resolved_file_path:
                latest_meta = photo_tracker.find_latest(route["device_id"])
                if not latest_meta:
                    raise RuntimeError(
                        "file_path is empty and no latest photo found for resolved device"
                    )
                resolved_file_path = _norm(latest_meta.get("local_path", ""))

            result = do_preview_local_file(
                args.server,
                file_path=resolved_file_path,
                session_id=route["session_id"],
                device_id=route["device_id"],
                tool_name=_norm(tool_name) or args.default_preview_tool_name,
                tool_timeout=int(timeout),
                request_timeout=int(request_timeout),
            )
            return {
                "success": bool(result.get("success", False)),
                "route": resolved,
                "file_path": resolved_file_path,
                "file_meta": latest_meta,
                "result": result,
            }
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    return mcp


def main() -> int:
    args = parse_args()
    _configure_logging(args.log_level)
    server = build_server(args)
    server.run(transport=args.transport, mount_path=args.mount_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
