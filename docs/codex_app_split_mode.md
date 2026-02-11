# Codex Split Mode (Standalone Codex Service)

This mode runs Codex in a separate process from `xiaozhi-server`.

## 1) Start standalone Codex service

```powershell
cd main/xiaozhi-server
python codex_app.py --host 127.0.0.1 --port 9001 --llm-name codex_app_server
```

`codex_app.py` exposes:
- `POST /session/resolve_or_create`
- `POST /v1/llm/codex/response`
- `GET /health`

## 2) Update `data/.config.yaml` for xiaozhi-server

Keep your existing codex config (used by `codex_app.py`), and add a remote LLM entry:

```yaml
selected_module:
  LLM: codex_remote_gateway

LLM:
  codex_app_server:
    type: codex
    # keep your original codex settings here

  codex_remote_gateway:
    type: codex_remote
    endpoint: "http://127.0.0.1:9001/v1/llm/codex/response"
    timeout: 300
    include_events: true

session_registry:
  endpoint: "http://127.0.0.1:9001/session/resolve_or_create"
  request_timeout: 3
  default_user_id: "test"
  enable_local_fallback: true
```

## 3) Start xiaozhi-server

```powershell
cd main/xiaozhi-server
python app.py
```

Now session mapping and model turns are both routed to standalone `codex_app.py`.

## Notes

- If `codex_app.py` is down, `codex_remote` calls fail.
- If `session_registry.enable_local_fallback=true`, session mapping can still fall back to local JSON when endpoint fails.
- To keep full decoupling, run `codex_app.py` as a separate long-lived process/service.
