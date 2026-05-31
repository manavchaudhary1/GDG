import unittest

from career_intelligence import generate_career_insights


class CareerInsightsTests(unittest.TestCase):
    def test_generates_skill_gaps_from_jobs_and_trends(self):
        profile = {
            "target_role": "Data Scientist",
            "skills": ["python", "sql", "statistics"],
        }
        jobs = [
            {"title": "Senior Data Scientist", "skills": ["Python", "SQL", "ML", "MLOps"]},
            {"title": "Data Scientist", "skills": ["Python", "Deep Learning", "Communication"]},
            {"title": "Backend Engineer", "skills": ["Go", "Kubernetes"]},
        ]
        trends = {"emerging_skills": {"GenAI": 3, "MLOps": 2}}

        insights = generate_career_insights(profile, jobs, trends)

        self.assertEqual(insights.target_role, "Data Scientist")
        self.assertIn("python", insights.strengths)
        self.assertIn("mlops", insights.skill_gaps)
        self.assertIn("genai", insights.skill_gaps)
        self.assertGreaterEqual(insights.market_alignment_score, 40)
        self.assertTrue(insights.recommendations)

    def test_handles_empty_market_data(self):
        insights = generate_career_insights(
            profile={"skills": ["python"]},
            jobs=[],
            trends={},
        )

        self.assertEqual(insights.market_alignment_score, 0)
        self.assertEqual(insights.skill_gaps, [])
        self.assertEqual(insights.recommendations, ["Maintain momentum by deepening expertise and applying consistently to well-matched roles."])


if __name__ == "__main__":
    unittest.main()
