from __future__ import annotations

import json
import os
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from career_intelligence import generate_career_insights
from huggingface_client import DEFAULT_GEMMA_MODEL, request_chat_completion

REMOTEOK_JOBS_URL = "https://remoteok.com/api"
STACKEXCHANGE_TAGS_URL = "https://api.stackexchange.com/2.3/tags"
FALLBACK_JOBS = [
    {"title": "Senior Data Scientist", "skills": ["Python", "SQL", "MLOps", "ML"]},
    {"title": "Data Scientist", "skills": ["Python", "Deep Learning", "Communication"]},
]
FALLBACK_TRENDS = {"emerging_skills": {"GenAI": 3, "MLOps": 2}}

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

    <small>Market jobs and trend data are fetched automatically from open web sources.</small>

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
        auto_fetch_market_data: true,
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


def _fetch_json(url: str, headers: dict[str, str] | None = None, timeout: int = 10) -> Any:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def fetch_open_market_data() -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    warnings: list[str] = []

    jobs: list[dict[str, Any]] = []
    try:
        payload = _fetch_json(REMOTEOK_JOBS_URL, headers={"User-Agent": "career-intelligence-dashboard"})
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("position") or item.get("title") or "").strip()
                tags = item.get("tags")
                if not title or not isinstance(tags, list):
                    continue
                skills = [str(tag).strip() for tag in tags if isinstance(tag, str) and str(tag).strip()]
                if skills:
                    jobs.append({"title": title, "skills": skills})
                if len(jobs) >= 30:
                    break
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Unable to fetch jobs from RemoteOK: {exc}")
    if not jobs:
        jobs = FALLBACK_JOBS.copy()
        warnings.append("Using fallback jobs sample data.")

    trends: dict[str, Any] = {}
    try:
        term_counts: dict[str, int] = {}
        for term in ("ai", "machine-learning", "mlops", "data-science"):
            encoded_term = quote_plus(term)
            url = (
                f"{STACKEXCHANGE_TAGS_URL}?order=desc&sort=popular&site=stackoverflow"
                f"&pagesize=10&inname={encoded_term}"
            )
            payload = _fetch_json(url)
            items = payload.get("items") if isinstance(payload, dict) else []
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                count = item.get("count")
                if name and isinstance(count, int):
                    term_counts[name] = max(term_counts.get(name, 0), count)

        ranked = sorted(term_counts.items(), key=lambda pair: pair[1], reverse=True)[:10]
        if ranked:
            max_count = ranked[0][1]
            emerging_skills: dict[str, int] = {}
            for skill, count in ranked:
                weight = 1 if max_count <= 0 else max(1, min(5, round((count / max_count) * 5)))
                emerging_skills[skill] = weight
            trends = {"emerging_skills": emerging_skills}
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Unable to fetch trends from Stack Exchange: {exc}")
    if not trends:
        trends = FALLBACK_TRENDS.copy()
        warnings.append("Using fallback trend sample data.")

    return jobs, trends, warnings


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
    market_data_fetcher=fetch_open_market_data,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    profile = payload.get("profile") if isinstance(payload.get("profile"), dict) else {}
    jobs = payload.get("jobs") if isinstance(payload.get("jobs"), list) else []
    trends = payload.get("trends") if isinstance(payload.get("trends"), dict) else {}

    warnings: list[str] = []
    auto_fetch_market_data = bool(payload.get("auto_fetch_market_data", not jobs and not trends))
    if auto_fetch_market_data:
        fetched_jobs, fetched_trends, fetch_warnings = market_data_fetcher()
        warnings.extend(fetch_warnings)
        if not jobs:
            jobs = fetched_jobs
        if not trends:
            trends = fetched_trends

    insights = generate_career_insights(profile=profile, jobs=jobs, trends=trends)
    response: dict[str, Any] = {
        "model": payload.get("model") or DEFAULT_GEMMA_MODEL,
        "market_data_sources": {
            "jobs": REMOTEOK_JOBS_URL,
            "trends": STACKEXCHANGE_TAGS_URL,
        },
        "market_data_summary": {
            "jobs_count": len(jobs),
            "trend_skills_count": len(trends.get("emerging_skills", {})) if isinstance(trends, dict) else 0,
        },
        "insights": asdict(insights),
    }
    if warnings:
        response["market_data_warnings"] = warnings

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
