import asyncio
import os
import threading
from typing import Any, Dict, Optional, Tuple

from config.logger import setup_logging
from core.providers.llm.base import LLMProviderBase
from core.providers.tools.server_mcp.mcp_client import ServerMCPClient

TAG = __name__
logger = setup_logging()


class _LoopRunner:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro, timeout: Optional[float] = None):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout)


class LLMProvider(LLMProviderBase):
    def __init__(self, config):
        self.command = config.get("command", "codex.cmd")
        self.args = list(config.get("args") or ["mcp-server"])
        self.env = config.get("env") or {}
        self.cwd = config.get("cwd") or config.get("workdir")
        self.prompt_mode = config.get("prompt_mode", "full_dialogue")
        self.resume_session = str(config.get("resume_session", False)).lower() in (
            "true",
            "1",
            "yes",
        )
        self.init_timeout = float(config.get("init_timeout", 20))

        self.sandbox = config.get("sandbox")
        self.approval_policy = config.get("approval_policy") or config.get(
            "approval-policy"
        )
        self.model = config.get("model")
        self.profile = config.get("profile")
        self.base_instructions = config.get("base_instructions") or config.get(
            "base-instructions"
        )
        self.developer_instructions = config.get("developer_instructions") or config.get(
            "developer-instructions"
        )
        self.compact_prompt = config.get("compact_prompt") or config.get("compact-prompt")
        self.codex_config = config.get("config")

        self._client: Optional[ServerMCPClient] = None
        self._client_lock = threading.Lock()
        self._loop_runner: Optional[_LoopRunner] = None

        self._session_map: Dict[str, str] = {}
        self._session_lock = threading.Lock()
        self._prompt_mode_state: Dict[str, bool] = {}
        self._prompt_mode_lock = threading.Lock()
        self._warned_missing_resume = False

    def _ensure_client(self) -> ServerMCPClient:
        if self._client:
            return self._client
        with self._client_lock:
            if self._client:
                return self._client
            if not self._loop_runner:
                self._loop_runner = _LoopRunner()
            env = os.environ.copy()
            env.update(self.env)
            client_config = {"command": self.command, "args": self.args, "env": env}
            client = ServerMCPClient(client_config)
            self._loop_runner.run(client.initialize(), timeout=self.init_timeout)
            self._client = client
            return client

    def _get_effective_prompt_mode(self, session_id: str) -> str:
        if self.prompt_mode != "first_full_then_last":
            return self.prompt_mode
        if not self.resume_session and not self._warned_missing_resume:
            logger.bind(tag=TAG).warning(
                "prompt_mode=first_full_then_last but resume_session is disabled; "
                "context will be lost after the first turn."
            )
            self._warned_missing_resume = True
        if not session_id:
            return "full_dialogue"
        with self._prompt_mode_lock:
            if not self._prompt_mode_state.get(session_id):
                return "full_dialogue"
            return "last_user"

    def _mark_prompt_sent(self, session_id: str) -> None:
        if self.prompt_mode != "first_full_then_last":
            return
        if not session_id:
            return
        with self._prompt_mode_lock:
            self._prompt_mode_state[session_id] = True

    def _build_prompt(self, dialogue, prompt_mode: str) -> str:
        if prompt_mode == "last_user":
            for msg in reversed(dialogue):
                if msg.get("role") == "user":
                    return msg.get("content", "")
            return ""

        parts = []
        for msg in dialogue:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content is None:
                content = ""
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            elif role == "tool":
                parts.append(f"Tool: {content}")
            else:
                parts.append(f"User: {content}")
        return "\n".join(parts)

    def _store_thread_id(self, session_id: str, thread_id: str) -> None:
        if not session_id or not thread_id:
            return
        with self._session_lock:
            self._session_map[session_id] = thread_id

    def _get_thread_id(self, session_id: str) -> Optional[str]:
        if not session_id:
            return None
        with self._session_lock:
            return self._session_map.get(session_id)

    def _extract_text(self, result) -> str:
        structured = getattr(result, "structuredContent", None)
        if structured is None and isinstance(result, dict):
            structured = result.get("structuredContent")
        if isinstance(structured, dict):
            for key in ("content", "message", "text", "output"):
                value = structured.get(key)
                if isinstance(value, str) and value:
                    return value

        content = getattr(result, "content", None)
        if content is None and isinstance(result, dict):
            content = result.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                text = getattr(item, "text", None)
                if text is None and isinstance(item, dict):
                    text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            return "".join(parts)
        return ""

    def _extract_thread_id(self, result) -> Optional[str]:
        structured = getattr(result, "structuredContent", None)
        if structured is None and isinstance(result, dict):
            structured = result.get("structuredContent")
        if isinstance(structured, dict):
            for key in ("threadId", "thread_id", "session_id", "conversationId"):
                value = structured.get(key)
                if isinstance(value, str) and value:
                    return value
        return None

    def _build_start_args(self, prompt: str) -> Dict[str, Any]:
        args: Dict[str, Any] = {"prompt": prompt}
        if self.approval_policy:
            args["approval-policy"] = self.approval_policy
        if self.sandbox:
            args["sandbox"] = self.sandbox
        if self.model:
            args["model"] = self.model
        if self.profile:
            args["profile"] = self.profile
        if self.base_instructions:
            args["base-instructions"] = self.base_instructions
        if self.developer_instructions:
            args["developer-instructions"] = self.developer_instructions
        if self.compact_prompt:
            args["compact-prompt"] = self.compact_prompt
        if isinstance(self.codex_config, dict):
            args["config"] = self.codex_config
        if self.cwd:
            args["cwd"] = self.cwd
        return args

    def response(self, session_id, dialogue, **kwargs):
        try:
            client = self._ensure_client()
        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to initialize Codex MCP client: {e}")
            yield "[Codex MCP error: failed to initialize]"
            return

        prompt_mode = self._get_effective_prompt_mode(session_id)
        prompt = self._build_prompt(dialogue, prompt_mode=prompt_mode)
        thread_id = self._get_thread_id(session_id) if self.resume_session else None

        try:
            if thread_id:
                tool_name = "codex-reply"
                tool_args = {"threadId": thread_id, "prompt": prompt}
            else:
                tool_name = "codex"
                tool_args = self._build_start_args(prompt)

            result = self._loop_runner.run(
                client.call_tool(tool_name, tool_args), timeout=None
            )
            new_thread_id = self._extract_thread_id(result)
            if new_thread_id:
                self._store_thread_id(session_id, new_thread_id)
            if prompt_mode == "full_dialogue":
                self._mark_prompt_sent(session_id)

            text = self._extract_text(result)
            if text:
                yield text
            else:
                yield ""
        except Exception as e:
            logger.bind(tag=TAG).error(f"Codex MCP call failed: {e}")
            yield "[Codex MCP error: call failed]"

