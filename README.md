# GDG

## Beyond the Resume: AI Career Intelligence

This repository includes a career intelligence platform that combines:
- Candidate profile data (target role + current skills)
- Open-web job posting requirements (market demand)
- Open-web labor-market trend signals (emerging skills)
- Optional Gemma 4 advisory output via Hugging Face chat completions

## Components

### 1) Rule-based intelligence engine
`/tmp/workspace/manavchaudhary1/GDG/career_intelligence.py`

`generate_career_insights(profile, jobs, trends)` returns:
- `market_alignment_score`
- `strengths`
- `skill_gaps`
- `recommendations`

### 2) Hugging Face Gemma integration
`/tmp/workspace/manavchaudhary1/GDG/huggingface_client.py`

Default model:
- `google/gemma-4-31B-it:novita`

Endpoint:
- `https://router.huggingface.co/v1/chat/completions`

### 3) Usage dashboard (UI + API)
`/tmp/workspace/manavchaudhary1/GDG/dashboard.py`

- UI at `GET /`
- Health check at `GET /health`
- Analysis endpoint at `POST /api/insights`

## Run dashboard

```bash
cd /tmp/workspace/manavchaudhary1/GDG
export HF_TOKEN=your_huggingface_token
python dashboard.py
```

Open:
- `http://127.0.0.1:8000`

## API payload example

```json
{
  "profile": {
    "target_role": "Data Scientist",
    "skills": ["python", "sql", "statistics"]
  },
  "auto_fetch_market_data": true,
  "include_ai": true,
  "model": "google/gemma-4-31B-it:novita"
}
```

When `auto_fetch_market_data` is true, the dashboard fetches:
- Jobs from `https://remoteok.com/api`
- Trend signals from `https://api.stackexchange.com/2.3/tags`

## Hugging Face curl reference

```bash
curl https://router.huggingface.co/v1/chat/completions \
  -H "Authorization: ******" \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": [{"type": "text", "text": "Describe this image in one sentence."}, {"type": "image_url", "image_url": {"url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"}}]}],
    "model": "google/gemma-4-31B-it:novita",
    "stream": false
  }'
```

## Run tests

```bash
python -m unittest discover -s tests -v
```
