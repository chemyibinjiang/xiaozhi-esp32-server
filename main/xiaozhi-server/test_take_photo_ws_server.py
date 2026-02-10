#!/usr/bin/env python3
"""
Fake websocket server to force device camera capture via MCP.

Typical usage (no app.py required):
  python test_take_photo_ws_server.py --ws-port 8000 --fake-vision

Device websocket stays unchanged:
  ws://<host-ip>:8000/xiaozhi/v1/
"""

import argparse
import asyncio
import json
import time
import uuid
from typing import Optional

from aiohttp import web
import websockets

from config.config_loader import load_config
from core.utils.auth import AuthToken
from core.utils.util import get_local_ip, get_vision_url


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fake websocket server that sends MCP self.camera.take_photo"
    )
    parser.add_argument("--ws-host", default="0.0.0.0", help="Fake websocket listen host")
    parser.add_argument("--ws-port", type=int, default=8000, help="Fake websocket listen port")
    parser.add_argument(
        "--question",
        default="Please take a photo and return one short sentence.",
        help="Question passed to self.camera.take_photo",
    )
    parser.add_argument(
        "--vision-url",
        default="",
        help="Vision URL sent in MCP initialize. Empty means auto.",
    )
    parser.add_argument(
        "--no-tools-list",
        action="store_true",
        help="Skip tools/list before tools/call",
    )
    parser.add_argument(
        "--fake-vision",
        action="store_true",
        help="Start local fake vision HTTP endpoint for Explain()",
    )
    parser.add_argument("--vision-host", default="0.0.0.0", help="Fake vision listen host")
    parser.add_argument("--vision-port", type=int, default=8003, help="Fake vision listen port")
    parser.add_argument(
        "--vision-path",
        default="/mcp/vision/explain",
        help="Fake vision endpoint path",
    )
    parser.add_argument(
        "--vision-response",
        default="photo done",
        help="Fake vision response text field",
    )
    parser.add_argument(
        "--print-binary",
        action="store_true",
        help="Print every binary frame (very noisy).",
    )
    parser.add_argument(
        "--binary-summary-interval",
        type=float,
        default=2.0,
        help="Seconds between binary traffic summaries when --print-binary is off.",
    )
    return parser.parse_args()


def build_server_hello(session_id: str):
    return {
        "type": "hello",
        "version": 1,
        "transport": "websocket",
        "session_id": session_id,
        "audio_params": {
            "format": "opus",
            "sample_rate": 16000,
            "channels": 1,
            "frame_duration": 60,
        },
    }


def ensure_auth_key(config: dict) -> str:
    # Keep behavior aligned with app.py so standalone run works.
    auth_key = config.get("server", {}).get("auth_key", "")
    if not auth_key or "你" in str(auth_key):
        auth_key = config.get("manager-api", {}).get("secret", "")
        if not auth_key or "你" in str(auth_key):
            auth_key = uuid.uuid4().hex
    config.setdefault("server", {})["auth_key"] = auth_key
    return auth_key


def build_initialize_payload(config: dict, device_id: str, vision_url: str):
    auth_key = ensure_auth_key(config)
    token = AuthToken(auth_key).generate_token(device_id)
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {},
                "vision": {"url": vision_url, "token": token},
            },
            "clientInfo": {"name": "FakeWsServer", "version": "1.0.0"},
        },
    }


async def send_mcp(ws, session_id: str, payload: dict):
    msg = {"type": "mcp", "session_id": session_id, "payload": payload}
    raw = json.dumps(msg, ensure_ascii=False)
    await ws.send(raw)
    print(f"[send] {raw}")


def parse_json(raw: str) -> Optional[dict]:
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def log_binary_summary(args, recv_state: dict, size: int):
    recv_state["frames"] += 1
    recv_state["bytes"] += size
    now = time.monotonic()
    interval = max(0.2, float(getattr(args, "binary_summary_interval", 2.0)))
    last = recv_state.get("last_report", now)
    if now - last < interval:
        return
    elapsed = max(1e-6, now - last)
    kbps = (recv_state["bytes"] * 8.0) / elapsed / 1000.0
    print(
        f"[recv][binary-summary] frames={recv_state['frames']} "
        f"bytes={recv_state['bytes']} rate={kbps:.1f}kbps"
    )
    recv_state["frames"] = 0
    recv_state["bytes"] = 0
    recv_state["last_report"] = now


async def recv_and_print(ws, args, recv_state: dict, timeout_sec: Optional[int] = None):
    if timeout_sec is None:
        raw = await ws.recv()
    else:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout_sec)
    if isinstance(raw, bytes):
        if args.print_binary:
            print(f"[recv][binary] len={len(raw)} first8={raw[:8].hex()}")
        else:
            log_binary_summary(args, recv_state, len(raw))
        return raw, None
    print(f"[recv] {raw}")
    return raw, parse_json(raw)


async def wait_mcp_result(ws, args, recv_state: dict, expected_id: int, timeout_sec: int):
    end_at = asyncio.get_running_loop().time() + timeout_sec
    while True:
        left = end_at - asyncio.get_running_loop().time()
        if left <= 0:
            raise TimeoutError(f"Timeout waiting MCP result id={expected_id}")
        _, obj = await recv_and_print(ws, args, recv_state, timeout_sec=int(left))
        if not isinstance(obj, dict):
            continue
        if obj.get("type") != "mcp":
            continue
        payload = obj.get("payload")
        if isinstance(payload, dict) and payload.get("id") == expected_id:
            return payload


def resolve_vision_url(args, config):
    if args.vision_url:
        return args.vision_url
    if args.fake_vision:
        ip = get_local_ip()
        return f"http://{ip}:{args.vision_port}{args.vision_path}"
    return get_vision_url(config)


def normalize_path(path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return path


async def start_fake_vision(args):
    vision_path = normalize_path(args.vision_path)

    async def handle_get(_request):
        return web.Response(text=f"fake vision ok: {vision_path}", content_type="text/plain")

    async def handle_post(request):
        question = ""
        image_size = 0
        try:
            if request.content_type.startswith("multipart/"):
                reader = await request.multipart()
                field1 = await reader.next()
                if field1 is not None:
                    question = await field1.text()
                field2 = await reader.next()
                if field2 is not None:
                    data = await field2.read()
                    image_size = len(data)
        except Exception:
            # Return success even when body parsing fails; capture path testing only.
            pass

        print(f"[vision] question='{question}' image_bytes={image_size}")
        resp = {
            "success": True,
            "action": "RESPONSE",
            "response": args.vision_response,
        }
        return web.Response(
            text=json.dumps(resp, ensure_ascii=False, separators=(",", ":")),
            content_type="application/json",
        )

    app = web.Application()
    app.add_routes([web.get(vision_path, handle_get), web.post(vision_path, handle_post)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.vision_host, args.vision_port)
    await site.start()
    print(f"[info] fake vision listening http://{args.vision_host}:{args.vision_port}{vision_path}")
    return runner


async def handle_client(ws, *, config, args, vision_url):
    request_path = ""
    device_id = "00:11:22:33:44:55"
    try:
        request_path = ws.request.path
        device_id = ws.request.headers.get("Device-Id", device_id)
    except Exception:
        pass
    print(f"[info] connected path={request_path} device_id={device_id}")
    recv_state = {"frames": 0, "bytes": 0, "last_report": time.monotonic()}

    # 1) Expect device hello
    _, first_obj = await recv_and_print(ws, args, recv_state, timeout_sec=15)
    if not isinstance(first_obj, dict) or first_obj.get("type") != "hello":
        raise RuntimeError("First frame is not hello")

    # 2) Reply server hello
    session_id = uuid.uuid4().hex
    server_hello = build_server_hello(session_id)
    await ws.send(json.dumps(server_hello, ensure_ascii=False))
    print(f"[send] {json.dumps(server_hello, ensure_ascii=False)}")

    # 3) Send MCP initialize
    await send_mcp(ws, session_id, build_initialize_payload(config, device_id, vision_url))
    await wait_mcp_result(ws, args, recv_state, expected_id=1, timeout_sec=20)

    # 4) Optional tools/list
    if not args.no_tools_list:
        await send_mcp(ws, session_id, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        await wait_mcp_result(ws, args, recv_state, expected_id=2, timeout_sec=20)

    # 5) Trigger photo
    call_id = 1001
    call_payload = {
        "jsonrpc": "2.0",
        "id": call_id,
        "method": "tools/call",
        "params": {
            "name": "self.camera.take_photo",
            "arguments": {"question": args.question},
        },
    }
    await send_mcp(ws, session_id, call_payload)
    result = await wait_mcp_result(ws, args, recv_state, expected_id=call_id, timeout_sec=90)
    print(f"[done] take_photo result: {json.dumps(result, ensure_ascii=False)}")

    # Keep receiving for observation.
    while True:
        await recv_and_print(ws, args, recv_state)


async def main():
    args = parse_args()
    config = load_config()
    vision_url = resolve_vision_url(args, config)
    print(f"[info] fake ws listening ws://{args.ws_host}:{args.ws_port}/xiaozhi/v1/")
    print(f"[info] vision_url used in MCP initialize: {vision_url}")

    vision_runner = None
    if args.fake_vision:
        vision_runner = await start_fake_vision(args)

    async def _handler(ws):
        try:
            await handle_client(ws, config=config, args=args, vision_url=vision_url)
        except Exception as e:
            print(f"[error] {e}")

    try:
        async with websockets.serve(_handler, args.ws_host, args.ws_port):
            await asyncio.Future()
    finally:
        if vision_runner is not None:
            await vision_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
