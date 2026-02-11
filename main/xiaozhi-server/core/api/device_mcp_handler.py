import json
import mimetypes
import os
import subprocess
import uuid
from aiohttp import web

from core.api.base_handler import BaseHandler
from core.providers.tools.device_mcp import call_mcp_tool
from core.utils.util import sanitize_tool_name, get_local_ip, is_valid_image_file

TAG = __name__


class DeviceMCPHandler(BaseHandler):
    def __init__(self, config: dict, ws_server):
        super().__init__(config)
        self.ws_server = ws_server
        data_dir = self.config.get("log", {}).get("data_dir", "data")
        self.preview_dir = os.path.join(data_dir, "preview_files")
        os.makedirs(self.preview_dir, exist_ok=True)

    def _json_response(self, body: dict, status: int = 200):
        return web.Response(
            text=json.dumps(body, ensure_ascii=False, separators=(",", ":")),
            content_type="application/json",
            status=status,
        )

    def _resolve_target_params(self, body: dict):
        session_id = str(body.get("session_id", "")).strip()
        device_id = str(body.get("device_id", "")).strip()
        return session_id, device_id

    async def _resolve_target_conn(self, session_id: str, device_id: str):
        if not session_id and not device_id:
            return None, None, self._json_response(
                {"success": False, "message": "session_id or device_id is required"},
                status=400,
            )

        conn = await self.ws_server.get_connection(
            session_id=session_id if session_id else None,
            device_id=device_id if device_id else None,
        )
        if not conn:
            return None, None, self._json_response(
                {"success": False, "message": "connection not found"},
                status=404,
            )

        mcp_client = getattr(conn, "mcp_client", None)
        if not mcp_client:
            return None, None, self._json_response(
                {"success": False, "message": "mcp client is not initialized"},
                status=409,
            )

        if not await mcp_client.is_ready():
            return None, None, self._json_response(
                {"success": False, "message": "mcp client is not ready"},
                status=409,
            )

        return conn, mcp_client, None

    def _read_json_body(self, body: dict):
        if not isinstance(body, dict):
            raise ValueError("request body must be json object")
        return body

    def _save_preview_file(self, source_file_path: str):
        source_file_path = os.path.abspath(source_file_path)
        if not os.path.exists(source_file_path):
            raise ValueError(f"file not found: {source_file_path}")
        if not os.path.isfile(source_file_path):
            raise ValueError(f"not a file: {source_file_path}")

        with open(source_file_path, "rb") as f:
            image_data = f.read()
        if not image_data:
            raise ValueError("file is empty")
        if not is_valid_image_file(image_data):
            raise ValueError("file is not a supported image")

        png_signature = b"\x89PNG\r\n\x1a\n"
        file_name = f"{uuid.uuid4().hex}.png"
        save_path = os.path.join(self.preview_dir, file_name)

        if image_data.startswith(png_signature):
            with open(save_path, "wb") as f:
                f.write(image_data)
            return file_name, save_path, len(image_data)

        # Firmware image decoder supports PNG in this project config.
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    source_file_path,
                    "-frames:v",
                    "1",
                    save_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            err = (e.stderr or e.stdout or "").strip()
            raise ValueError(f"failed to convert image to png: {err}") from e
        except FileNotFoundError as e:
            raise ValueError(
                "failed to convert image to png: ffmpeg is not installed or not in PATH"
            ) from e

        if not os.path.exists(save_path):
            raise ValueError("failed to convert image to png: output not generated")
        with open(save_path, "rb") as f:
            png_data = f.read()
        if not png_data.startswith(png_signature):
            raise ValueError("failed to convert image to png: invalid output format")

        return file_name, save_path, len(png_data)

    def _build_preview_url(self, file_name: str):
        port = int(self.config.get("server", {}).get("http_port", 8003))
        return f"http://{get_local_ip()}:{port}/mcp/device/local_files/{file_name}"

    def _build_preview_tool_candidates(self, tool_name_raw: str):
        candidates = []
        raw = (tool_name_raw or "").strip()
        if raw:
            candidates.append(raw)

        sanitized = sanitize_tool_name(raw) if raw else ""
        if sanitized and sanitized not in candidates:
            candidates.append(sanitized)

        dotted = raw.replace("_", ".") if raw else ""
        if dotted and dotted not in candidates:
            candidates.append(dotted)

        underscored = raw.replace(".", "_") if raw else ""
        if underscored and underscored not in candidates:
            candidates.append(underscored)

        for fixed_name in [
            "self.screen.preview_image",
            "self_screen_preview_image",
            "self.screen.preview_screen_shot",
            "self_screen_preview_screen_shot",
            "preview_screen_shot",
            "self.screen.preview_screenshot",
            "self_screen_preview_screenshot",
            "preview_screenshot",
        ]:
            if fixed_name not in candidates:
                candidates.append(fixed_name)

        return candidates

    async def handle_get(self, request):
        response = None
        try:
            if not self.ws_server:
                response = self._json_response(
                    {"success": False, "message": "ws server is not available"},
                    status=500,
                )
            else:
                connections = await self.ws_server.list_connections()
                response = self._json_response(
                    {"success": True, "connections": connections}
                )
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"list sessions failed: {e}")
            response = self._json_response(
                {"success": False, "message": "internal server error"},
                status=500,
            )
        finally:
            if response:
                self._add_cors_headers(response)
            return response

    async def handle_post(self, request):
        response = None
        try:
            if not self.ws_server:
                response = self._json_response(
                    {"success": False, "message": "ws server is not available"},
                    status=500,
                )
                return response

            try:
                body = await request.json()
            except Exception:
                response = self._json_response(
                    {"success": False, "message": "request body must be json"},
                    status=400,
                )
                return response

            self._read_json_body(body)
            session_id, device_id = self._resolve_target_params(body)
            question = str(body.get("question", "Please take a photo.")).strip()
            photo_name = str(body.get("photo_name", "")).strip()
            tool_name_raw = str(
                body.get("tool_name", "self.camera.take_photo")
            ).strip()
            tool_name = sanitize_tool_name(tool_name_raw)
            timeout = int(body.get("timeout", 90))
            if timeout <= 0:
                timeout = 90

            conn, mcp_client, error_resp = await self._resolve_target_conn(
                session_id, device_id
            )
            if error_resp:
                response = error_resp
                return response

            tool_args = {"question": question}
            if photo_name:
                tool_args["photo_name"] = photo_name

            result = await call_mcp_tool(
                conn,
                mcp_client,
                tool_name,
                tool_args,
                timeout=timeout,
            )

            response = self._json_response(
                {
                    "success": True,
                    "tool": tool_name_raw,
                    "tool_sanitized": tool_name,
                    "session_id": conn.session_id,
                    "device_id": conn.device_id,
                    "requested_photo_name": photo_name,
                    "result": result,
                }
            )
        except ValueError as e:
            response = self._json_response(
                {"success": False, "message": str(e)},
                status=400,
            )
        except TimeoutError:
            response = self._json_response(
                {"success": False, "message": "tool call timeout"},
                status=504,
            )
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"take_photo failed: {e}")
            response = self._json_response(
                {"success": False, "message": str(e)},
                status=500,
            )
        finally:
            if response:
                self._add_cors_headers(response)
            return response

    async def handle_preview_local_file_post(self, request):
        response = None
        try:
            if not self.ws_server:
                response = self._json_response(
                    {"success": False, "message": "ws server is not available"},
                    status=500,
                )
                return response

            try:
                body = await request.json()
            except Exception:
                response = self._json_response(
                    {"success": False, "message": "request body must be json"},
                    status=400,
                )
                return response

            self._read_json_body(body)
            session_id, device_id = self._resolve_target_params(body)
            file_path = str(body.get("file_path", "")).strip()
            if not file_path:
                response = self._json_response(
                    {"success": False, "message": "file_path is required"},
                    status=400,
                )
                return response

            timeout = int(body.get("timeout", 90))
            if timeout <= 0:
                timeout = 90

            tool_name_raw = str(
                body.get("tool_name", "self.screen.preview_image")
            ).strip()
            tool_name = sanitize_tool_name(tool_name_raw)

            conn, mcp_client, error_resp = await self._resolve_target_conn(
                session_id, device_id
            )
            if error_resp:
                response = error_resp
                return response

            file_name, save_path, size = self._save_preview_file(file_path)
            file_url = self._build_preview_url(file_name)
            self.logger.bind(tag=TAG).info(
                f"prepared preview file: src={file_path} saved={save_path} bytes={size}"
            )

            result = None
            used_tool = ""
            tried_tools = self._build_preview_tool_candidates(tool_name_raw)
            last_unknown_error = None
            for candidate_tool in tried_tools:
                try:
                    result = await call_mcp_tool(
                        conn,
                        mcp_client,
                        sanitize_tool_name(candidate_tool),
                        {"url": file_url},
                        timeout=timeout,
                        allow_unlisted=True,
                        raw_tool_name=candidate_tool,
                    )
                    used_tool = candidate_tool
                    break
                except Exception as e:
                    err_msg = str(e)
                    unknown_tool = (
                        "Unknown tool" in err_msg
                        or "工具" in err_msg and "不存在" in err_msg
                        or "tool" in err_msg and "not found" in err_msg
                    )
                    if unknown_tool:
                        last_unknown_error = e
                        self.logger.bind(tag=TAG).warning(
                            f"preview tool not found: {candidate_tool}, trying next"
                        )
                        continue
                    raise e

            if result is None:
                if last_unknown_error:
                    raise ValueError(
                        "no preview tool found on device, tried: "
                        + ", ".join(tried_tools)
                    )
                raise RuntimeError("no preview tool succeeded")

            response = self._json_response(
                {
                    "success": True,
                    "tool": used_tool if used_tool else tool_name_raw,
                    "tool_sanitized": sanitize_tool_name(
                        used_tool if used_tool else tool_name_raw
                    ),
                    "tried_tools": tried_tools,
                    "session_id": conn.session_id,
                    "device_id": conn.device_id,
                    "file_path": file_path,
                    "served_file": file_name,
                    "url": file_url,
                    "result": result,
                }
            )
        except ValueError as e:
            response = self._json_response(
                {"success": False, "message": str(e)},
                status=400,
            )
        except TimeoutError:
            response = self._json_response(
                {"success": False, "message": "tool call timeout"},
                status=504,
            )
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"preview_local_file failed: {e}")
            response = self._json_response(
                {"success": False, "message": str(e)},
                status=500,
            )
        finally:
            if response:
                self._add_cors_headers(response)
            return response

    async def handle_local_file_get(self, request):
        response = None
        try:
            file_name = request.match_info.get("file_name", "").strip()
            if not file_name:
                response = self._json_response(
                    {"success": False, "message": "missing file_name"},
                    status=400,
                )
                return response
            if "/" in file_name or "\\" in file_name or ".." in file_name:
                response = self._json_response(
                    {"success": False, "message": "invalid file_name"},
                    status=400,
                )
                return response

            file_path = os.path.abspath(os.path.join(self.preview_dir, file_name))
            preview_dir_abs = os.path.abspath(self.preview_dir)
            if not file_path.startswith(preview_dir_abs + os.sep):
                response = self._json_response(
                    {"success": False, "message": "invalid file path"},
                    status=400,
                )
                return response
            if not os.path.exists(file_path):
                response = self._json_response(
                    {"success": False, "message": "file not found"},
                    status=404,
                )
                return response

            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = "application/octet-stream"

            response = web.FileResponse(path=file_path)
            response.content_type = mime_type
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"serve local preview file failed: {e}")
            response = self._json_response(
                {"success": False, "message": "internal server error"},
                status=500,
            )
        finally:
            if response:
                self._add_cors_headers(response)
            return response
