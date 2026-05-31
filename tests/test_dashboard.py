import unittest

from dashboard import build_dashboard_response


class DashboardResponseTests(unittest.TestCase):
    def test_returns_rule_based_insights_without_ai(self):
        payload = {
            "profile": {"target_role": "Data Scientist", "skills": ["python", "sql"]},
            "jobs": [{"title": "Data Scientist", "skills": ["python", "sql", "mlops"]}],
            "trends": {"emerging_skills": {"genai": 2}},
            "include_ai": False,
        }

        response = build_dashboard_response(payload)

        self.assertIn("insights", response)
        self.assertNotIn("ai_advice", response)
        self.assertEqual(response["model"], "google/gemma-4-31B-it:novita")

    def test_includes_ai_advice_when_token_available(self):
        payload = {
            "profile": {"target_role": "Data Scientist", "skills": ["python", "sql"]},
            "jobs": [{"title": "Data Scientist", "skills": ["python", "sql", "mlops"]}],
            "trends": {},
            "include_ai": True,
            "hf_token": "hf_test",
        }

        called = {}

        def fake_chat(messages, hf_token, model):
            called["messages"] = messages
            called["hf_token"] = hf_token
            called["model"] = model
            return "Do project A, B, C"

        response = build_dashboard_response(payload, chat_completion_fn=fake_chat)

        self.assertEqual(response["ai_advice"], "Do project A, B, C")
        self.assertEqual(called["hf_token"], "hf_test")
        self.assertTrue(called["messages"])
        self.assertIn("compatible", called["messages"][1]["content"].lower())

    def test_reports_error_if_ai_requested_without_token(self):
        payload = {
            "profile": {"skills": ["python"]},
            "jobs": [],
            "trends": {},
            "include_ai": True,
            "auto_fetch_market_data": False,
        }

        response = build_dashboard_response(payload, env={})

        self.assertIn("ai_error", response)
        self.assertIn("Missing Hugging Face token", response["ai_error"])

    def test_does_not_auto_fetch_market_data_when_inputs_absent(self):
        payload = {
            "profile": {"target_role": "Data Scientist", "skills": ["python"]},
            "include_ai": False,
        }

        response = build_dashboard_response(payload)

        self.assertIn("insights", response)
        self.assertEqual(response["market_data_summary"]["jobs_count"], 0)
        self.assertEqual(response["market_data_summary"]["trend_skills_count"], 0)

    def test_reports_disabled_auto_fetch_when_requested(self):
        payload = {
            "profile": {"target_role": "Data Scientist", "skills": ["python"]},
            "include_ai": False,
            "auto_fetch_market_data": True,
        }

        response = build_dashboard_response(payload)

        self.assertIn("market_data_warnings", response)
        self.assertIn("disabled", response["market_data_warnings"][0].lower())


if __name__ == "__main__":
    unittest.main()
