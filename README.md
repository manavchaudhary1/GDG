# GDG

## Beyond the Resume: AI Career Intelligence

This repository now includes a lightweight career intelligence engine that combines:
- Candidate profile data (target role + current skills)
- Job posting requirements (market demand)
- Labor-market trend signals (emerging skills)

### Core capability
`generate_career_insights(profile, jobs, trends)` returns:
- **market_alignment_score**: match quality with relevant market demand
- **strengths**: in-demand skills already present in the candidate profile
- **skill_gaps**: highest priority missing skills
- **recommendations**: actionable next steps

### Quick example
```python
from career_intelligence import generate_career_insights

profile = {"target_role": "Data Scientist", "skills": ["python", "sql"]}
jobs = [{"title": "Data Scientist", "skills": ["python", "sql", "mlops", "genai"]}]
trends = {"emerging_skills": {"genai": 3}}

insights = generate_career_insights(profile, jobs, trends)
print(insights)
```

### Run tests
```bash
python -m unittest discover -s tests -v
```
