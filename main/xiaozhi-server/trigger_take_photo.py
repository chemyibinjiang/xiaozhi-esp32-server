#!/usr/bin/env python3
"""
Trigger one device camera capture through app.py HTTP endpoints.

Functions:
  - list_sessions(base_url)
  - trigger_take_photo(base_url, ...)

CLI examples:
  python trigger_take_photo.py --server http://127.0.0.1:8003 --list-sessions
  python trigger_take_photo.py --server http://127.0.0.1:8003 --device-id 94:a9:90:28:e8:ec --question "请拍照"
"""

import argparse
import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


def _request_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return json.loads(body)
        except Exception:
            return {"success": False, "message": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def list_sessions(base_url: str, timeout: int = 10) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/mcp/device/sessions"
    return _request_json("GET", url, timeout=timeout)


def trigger_take_photo(
    base_url: str,
    *,
    session_id: str = "",
    device_id: str = "",
    question: str = "Please take a photo.",
    tool_name: str = "self.camera.take_photo",
    tool_timeout: int = 90,
    request_timeout: int = 120,
) -> Dict[str, Any]:
    if not session_id and not device_id:
        raise ValueError("session_id or device_id is required")

    payload: Dict[str, Any] = {
        "question": question,
        "timeout": int(tool_timeout),
        "tool_name": tool_name,
    }
    if session_id:
        payload["session_id"] = session_id
    if device_id:
        payload["device_id"] = device_id

    url = f"{base_url.rstrip('/')}/mcp/device/take_photo"
    return _request_json("POST", url, payload=payload, timeout=request_timeout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trigger one camera capture via app.py")
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8003",
        help="HTTP server base URL, e.g. http://127.0.0.1:8003",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="Only list current online sessions",
    )
    parser.add_argument("--session-id", default="", help="Target session_id")
    parser.add_argument("--device-id", default="", help="Target device_id")
    parser.add_argument(
        "--question",
        default="Please take a photo.",
        help="Question passed to self.camera.take_photo",
    )
    parser.add_argument(
        "--tool-name",
        default="self.camera.take_photo",
        help="Tool name to call (raw name, server will sanitize)",
    )
    parser.add_argument(
        "--tool-timeout",
        type=int,
        default=90,
        help="Tool call timeout seconds (sent to server)",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=120,
        help="HTTP request timeout seconds",
    )
    parser.add_argument(
        "--auto-pick-first",
        action="store_true",
        help="If no session/device provided, pick first online session",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_sessions:
        result = list_sessions(args.server, timeout=args.request_timeout)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("success") else 1

    session_id = args.session_id.strip()
    device_id = args.device_id.strip()

    if not session_id and not device_id and args.auto_pick_first:
        sessions = list_sessions(args.server, timeout=args.request_timeout)
        if not sessions.get("success"):
            print(json.dumps(sessions, ensure_ascii=False, indent=2))
            return 1
        items = sessions.get("connections", [])
        if not items:
            print(
                json.dumps(
                    {"success": False, "message": "no online sessions"},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 1
        first = items[0]
        session_id = str(first.get("session_id", "")).strip()
        device_id = str(first.get("device_id", "")).strip()

    try:
        result = trigger_take_photo(
            args.server,
            session_id=session_id,
            device_id=device_id,
            question=args.question,
            tool_name=args.tool_name,
            tool_timeout=args.tool_timeout,
            request_timeout=args.request_timeout,
        )
    except ValueError as e:
        print(json.dumps({"success": False, "message": str(e)}, ensure_ascii=False))
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
