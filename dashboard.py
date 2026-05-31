from __future__ import annotations

import json
import os
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from career_intelligence import generate_career_insights
from huggingface_client import DEFAULT_GEMMA_MODEL, request_chat_completion

DASHBOARD_HTML = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>AI Career Intelligence Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #f7f9fc; color: #111827; }
    h1 { margin-bottom: 8px; }
    .card { background: #fff; border: 1px solid #d1d5db; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
    textarea, input { width: 100%; padding: 8px; margin-top: 6px; margin-bottom: 12px; box-sizing: border-box; }
    button { padding: 10px 14px; border: 0; border-radius: 6px; background: #2563eb; color: #fff; cursor: pointer; }
    pre { white-space: pre-wrap; background: #111827; color: #f9fafb; padding: 12px; border-radius: 6px; overflow-x: auto; }
    small { color: #6b7280; }
  </style>
</head>
<body>
  <h1>Beyond the Resume: AI Career Intelligence</h1>
  <p>Run rule-based market intelligence and optional Gemma 4 advice through Hugging Face.</p>

  <div class=\"card\">
    <label>Target Role</label>
    <input id=\"target_role\" value=\"Data Scientist\" />

    <label>Skills (comma-separated)</label>
    <input id=\"skills\" value=\"python, sql, statistics\" />

    <label>Job Postings JSON (array of objects with title + skills)</label>
    <textarea id=\"jobs\" rows=\"7\">[
  {"title":"Senior Data Scientist","skills":["Python","SQL","MLOps","ML"]},
  {"title":"Data Scientist","skills":["Python","Deep Learning","Communication"]}
]</textarea>

    <label>Trend JSON (object with emerging_skills)</label>
    <textarea id=\"trends\" rows=\"4\">{"emerging_skills":{"GenAI":3,"MLOps":2}}</textarea>

    <label><input id=\"include_ai\" type=\"checkbox\" checked /> Include Gemma 4 AI recommendation</label>

    <label>Hugging Face Token (optional, fallback to HF_TOKEN env)</label>
    <input id=\"hf_token\" type=\"password\" placeholder=\"hf_xxx\" />

    <button onclick=\"submitPayload()\">Generate Insights</button>
    <small>Model default: google/gemma-4-31B-it:novita</small>
  </div>

  <div class=\"card\">
    <h3>Response</h3>
    <pre id=\"output\">{}</pre>
  </div>

  <script>
    async function submitPayload() {
      const payload = {
        profile: {
          target_role: document.getElementById('target_role').value,
          skills: document.getElementById('skills').value.split(',').map(s => s.trim()).filter(Boolean)
        },
        jobs: JSON.parse(document.getElementById('jobs').value || '[]'),
        trends: JSON.parse(document.getElementById('trends').value || '{}'),
        include_ai: document.getElementById('include_ai').checked,
        hf_token: document.getElementById('hf_token').value
      };

      const res = await fetch('/api/insights', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });

      const body = await res.json();
      document.getElementById('output').textContent = JSON.stringify(body, null, 2);
    }
  </script>
</body>
</html>
"""


def _build_ai_messages(profile: dict[str, Any], insights: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": "You are a practical career advisor. Keep guidance concise, specific, and actionable.",
        },
        {
            "role": "user",
            "content": (
                "Given this candidate profile and market analysis, provide a prioritized 90-day plan."
                f"\nProfile: {json.dumps(profile)}"
                f"\nInsights: {json.dumps(insights)}"
            ),
        },
    ]


def build_dashboard_response(
    payload: dict[str, Any],
    chat_completion_fn=request_chat_completion,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    profile = payload.get("profile") if isinstance(payload.get("profile"), dict) else {}
    jobs = payload.get("jobs") if isinstance(payload.get("jobs"), list) else []
    trends = payload.get("trends") if isinstance(payload.get("trends"), dict) else {}

    insights = generate_career_insights(profile=profile, jobs=jobs, trends=trends)
    response: dict[str, Any] = {
        "model": payload.get("model") or DEFAULT_GEMMA_MODEL,
        "insights": asdict(insights),
    }

    if payload.get("include_ai"):
        effective_env = env or os.environ
        hf_token = str(payload.get("hf_token") or effective_env.get("HF_TOKEN") or "").strip()
        if not hf_token:
            response["ai_error"] = "Missing Hugging Face token. Provide hf_token or set HF_TOKEN env variable."
        else:
            messages = _build_ai_messages(profile, response["insights"])
            try:
                response["ai_advice"] = chat_completion_fn(
                    messages=messages,
                    hf_token=hf_token,
                    model=response["model"],
                )
            except Exception as exc:  # noqa: BLE001
                response["ai_error"] = str(exc)

    return response


class DashboardHandler(BaseHTTPRequestHandler):
    def _json_response(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            body = DASHBOARD_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/health":
            self._json_response(HTTPStatus.OK, {"status": "ok"})
            return

        self._json_response(HTTPStatus.NOT_FOUND, {"error": "Not Found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/insights":
            self._json_response(HTTPStatus.NOT_FOUND, {"error": "Not Found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length > 0 else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Payload must be a JSON object.")
        except (json.JSONDecodeError, ValueError) as exc:
            self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        response = build_dashboard_response(payload)
        self._json_response(HTTPStatus.OK, response)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Dashboard available at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
