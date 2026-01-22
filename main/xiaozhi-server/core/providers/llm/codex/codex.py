import json
import os
import re
import subprocess
import threading
import time
from queue import Queue, Empty

from config.logger import setup_logging
from core.providers.llm.base import LLMProviderBase
from core.utils.llm_runtime import block_abort, unblock_abort
from core.utils.llm_stream import wrap_status
from .stderr_summary import summarize

TAG = __name__
logger = setup_logging()
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}\b"
)


class LLMProvider(LLMProviderBase):
    def __init__(self, config):
        self.command = config.get("command", "codex")
        self.args = list(config.get("args") or [])
        if not self.args:
            self.args = ["exec", "--json", "-"]
        self.prompt_mode = config.get("prompt_mode", "full_dialogue")
        self.prompt_via_stdin = str(config.get("prompt_via_stdin", True)).lower() in (
            "true",
            "1",
            "yes",
        )
        self.cwd = config.get("cwd") or config.get("workdir")
        self.env = config.get("env") or {}
        self.stderr_summary_enabled = str(
            config.get("stderr_summary_enabled", True)
        ).lower() in ("true", "1", "yes")
        self.stderr_summary_min_interval = float(
            config.get("stderr_summary_min_interval", 0)
        )
        self.resume_session = str(config.get("resume_session", False)).lower() in (
            "true",
            "1",
            "yes",
        )
        self._session_map = {}
        self._session_lock = threading.Lock()

    def _filter_args(self, args):
        filtered = []
        for arg in args:
            if arg in ("exec", "resume", "e"):
                continue
            if arg == "-":
                continue
            filtered.append(arg)
        return filtered

    def _build_command(self, resume_id: str = None) -> list:
        args = list(self.args)
        if resume_id:
            args = self._filter_args(args)
            if "--json" not in args:
                args.append("--json")
            if self.prompt_via_stdin and "-" not in args:
                args.append("-")
            return [self.command, "exec", "resume", resume_id] + args

        if not any(arg in ("exec", "e") for arg in args):
            args.insert(0, "exec")
        if "--json" not in args:
            args.append("--json")
        if self.prompt_via_stdin and "-" not in args:
            args.append("-")
        return [self.command] + args

    def _build_prompt(self, dialogue) -> str:
        if self.prompt_mode == "last_user":
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

    def _parse_event(self, line: str):
        if not line:
            return None
        payload = line.strip()
        if not payload:
            return None
        if payload.startswith("data:"):
            payload = payload[5:].strip()
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return payload

    def _find_uuid(self, value):
        if not value:
            return None
        if isinstance(value, str):
            match = _UUID_RE.search(value)
            return match.group(0) if match else None
        return None

    def _extract_session_id(self, event):
        if not event:
            return None
        if isinstance(event, str):
            return self._find_uuid(event)

        if isinstance(event, dict):
            for key in (
                "session_id",
                "sessionId",
                "conversation_id",
                "conversationId",
                "session",
                "conversation",
            ):
                candidate = self._extract_session_id(event.get(key))
                if candidate:
                    return candidate
            candidate = self._extract_session_id(event.get("id"))
            if candidate:
                return candidate
            return self._find_uuid(json.dumps(event, ensure_ascii=False))

        if isinstance(event, list):
            for item in event:
                candidate = self._extract_session_id(item)
                if candidate:
                    return candidate
        return None

    def _store_session_id(self, session_id: str, event) -> None:
        if not session_id or not event:
            return
        candidate = self._extract_session_id(event)
        if not candidate:
            return
        with self._session_lock:
            self._session_map[session_id] = candidate

    def _get_resume_id(self, session_id: str):
        if not session_id:
            return None
        with self._session_lock:
            return self._session_map.get(session_id)

    def _collect_text(self, value, items):
        if value is None:
            return
        if isinstance(value, str):
            if value:
                items.append(value)
            return
        if isinstance(value, list):
            for item in value:
                self._collect_text(item, items)
            return
        if isinstance(value, dict):
            for key in (
                "text",
                "content",
                "message",
                "output",
                "delta",
                "data",
                "response",
                "result",
                "final",
            ):
                if key in value:
                    self._collect_text(value[key], items)
            return

    def _extract_text_chunks(self, event):
        if event is None:
            return []
        if isinstance(event, str):
            return [event]

        items = []
        if isinstance(event, dict):
            event_type = event.get("type")
            if event_type:
                if event_type.endswith(".delta") or event_type.endswith(".text.delta"):
                    self._collect_text(event.get("delta") or event.get("text"), items)
                elif event_type.endswith(".error"):
                    self._collect_text(event.get("error") or event.get("message"), items)
                else:
                    self._collect_text(event, items)
            else:
                self._collect_text(event, items)
        return items

    def _event_to_text(self, event) -> str:
        chunks = self._extract_text_chunks(event)
        if not chunks:
            return ""
        return "\n".join(chunks).strip()

    def response(self, session_id, dialogue, **kwargs):
        prompt = self._build_prompt(dialogue)
        resume_id = self._get_resume_id(session_id) if self.resume_session else None
        command = self._build_command(resume_id=resume_id)
        if not self.prompt_via_stdin and prompt:
            command = command + [prompt]

        env = os.environ.copy()
        env.update(self.env)

        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=self.cwd,
                env=env,
            )
        except FileNotFoundError:
            logger.bind(tag=TAG).error("Codex CLI not found in PATH.")
            yield "[Codex CLI error: command not found]"
            return
        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to start Codex CLI: {e}")
            yield "[Codex CLI error: failed to start]"
            return

        if self.prompt_via_stdin and proc.stdin:
            try:
                proc.stdin.write(prompt)
                proc.stdin.close()
            except Exception as e:
                logger.bind(tag=TAG).error(f"Failed to write prompt: {e}")

        queue = Queue()

        def reader(stream, name):
            for line in iter(stream.readline, ""):
                queue.put((name, line))
            queue.put((name, None))

        threads = []
        if proc.stdout:
            threads.append(threading.Thread(target=reader, args=(proc.stdout, "stdout"), daemon=True))
        if proc.stderr:
            threads.append(threading.Thread(target=reader, args=(proc.stderr, "stderr"), daemon=True))
        for t in threads:
            t.start()

        stdout_done = False
        stderr_done = False
        stdout_started = False
        abort_blocked = False
        last_summary = None
        last_summary_time = 0.0

        try:
            while not (stdout_done and stderr_done):
                try:
                    name, line = queue.get(timeout=0.1)
                except Empty:
                    if proc.poll() is not None and stdout_done and stderr_done:
                        break
                    continue

                if line is None:
                    if name == "stdout":
                        stdout_done = True
                    elif name == "stderr":
                        stderr_done = True
                    continue

                event = self._parse_event(line)
                if event is not None:
                    self._store_session_id(session_id, event)
                if name == "stderr":
                    if stdout_started or not self.stderr_summary_enabled:
                        continue
                    stderr_text = self._event_to_text(event)
                    if not stderr_text:
                        continue
                    summary = summarize(stderr_text)
                    if not summary:
                        continue
                    summary = summary.strip()
                    if not summary:
                        continue
                    now = time.monotonic()
                    if (
                        self.stderr_summary_min_interval > 0
                        and now - last_summary_time < self.stderr_summary_min_interval
                    ):
                        continue
                    if summary == last_summary:
                        continue
                    if not abort_blocked:
                        block_abort(session_id)
                        abort_blocked = True
                    last_summary = summary
                    last_summary_time = now
                    yield wrap_status(summary)
                    continue

                for chunk in self._extract_text_chunks(event):
                    if not chunk:
                        continue
                    if not stdout_started:
                        stdout_started = True
                        if abort_blocked:
                            unblock_abort(session_id)
                            abort_blocked = False
                    yield chunk

            exit_code = proc.wait()
            if exit_code != 0 and not stdout_started:
                logger.bind(tag=TAG).error(f"Codex CLI exited with code {exit_code}")
                yield f"[Codex CLI error: exit code {exit_code}]"
        finally:
            if abort_blocked:
                unblock_abort(session_id)
            for t in threads:
                t.join(timeout=1)

    def response_with_functions(self, session_id, dialogue, functions=None, **kwargs):
        for chunk in self.response(session_id, dialogue, **kwargs):
            yield chunk, None
