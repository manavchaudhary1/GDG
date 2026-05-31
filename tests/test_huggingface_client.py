import json
import unittest

from huggingface_client import request_chat_completion


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class HuggingFaceClientTests(unittest.TestCase):
    def test_request_chat_completion_parses_string_content(self):
        captured = {}

        def fake_requester(request, timeout):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["auth"] = request.headers.get("Authorization")
            captured["timeout"] = timeout
            return _FakeResponse({"choices": [{"message": {"content": "Actionable answer"}}]})

        output = request_chat_completion(
            messages=[{"role": "user", "content": "hello"}],
            hf_token="hf_test",
            requester=fake_requester,
        )

        self.assertEqual(output, "Actionable answer")
        self.assertEqual(captured["body"]["model"], "google/gemma-4-31B-it:novita")
        self.assertEqual(captured["body"]["stream"], False)
        self.assertEqual(captured["timeout"], 30)
        self.assertIn("Bearer", captured["auth"])

    def test_request_chat_completion_parses_multimodal_content(self):
        def fake_requester(request, timeout):
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"type": "text", "text": "Line one"},
                                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                                    {"type": "text", "text": "Line two"},
                                ]
                            }
                        }
                    ]
                }
            )

        output = request_chat_completion(
            messages=[{"role": "user", "content": "hello"}],
            hf_token="hf_test",
            requester=fake_requester,
        )

        self.assertEqual(output, "Line one\nLine two")


if __name__ == "__main__":
    unittest.main()
