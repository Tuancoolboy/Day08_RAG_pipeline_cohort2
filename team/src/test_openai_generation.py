import os
import unittest
from unittest.mock import patch

from team.src.config import ConfigError, load_openai_settings
from team.src.generation import build_messages, generate_with_openai
from team.src.pipeline import answer_question
from team.src.retrieval import retrieve
from team.src.vector_store import build_vector_index, vector_search, vector_store_status

try:
    from fastapi.testclient import TestClient
    from team.api import app
except ImportError:  # pragma: no cover - dependency is declared in requirements.txt
    TestClient = None
    app = None


class OpenAIConfigTests(unittest.TestCase):
    def test_missing_api_key_raises_config_error(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-xxx"}, clear=True):
            with self.assertRaises(ConfigError):
                load_openai_settings()

    def test_load_openai_settings_from_environment(self):
        env = {
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL": "test-model",
            "OPENAI_TIMEOUT_SECONDS": "5",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = load_openai_settings()

        self.assertEqual(settings.api_key, "sk-test")
        self.assertEqual(settings.model, "test-model")
        self.assertEqual(settings.timeout_seconds, 5.0)


class OpenAIGenerationTests(unittest.TestCase):
    def test_build_messages_includes_query_and_context(self):
        sources = [
            {
                "content": "Dieu 249 quy dinh hinh phat ve tang tru trai phep chat ma tuy.",
                "score": 0.91,
                "metadata": {"source": "Bo luat Hinh su", "year": "2015"},
            }
        ]

        messages = build_messages("Dieu 249 quy dinh gi?", sources)

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Dieu 249 quy dinh gi?", messages[1]["content"])
        self.assertIn("[Bo luat Hinh su, 2015]", messages[1]["content"])

    def test_generate_with_openai_missing_key_returns_safe_response(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-xxx"}, clear=True):
            result = generate_with_openai("Test question?", [])

        self.assertIn("Missing OPENAI_API_KEY", result["answer"])
        self.assertEqual(result["metadata"]["error"], "config_error")


class RetrievalPipelineTests(unittest.TestCase):
    def test_retrieve_returns_sources_for_article_249(self):
        results = retrieve("Dieu 249 tang tru trai phep chat ma tuy", top_k=3)

        self.assertGreater(len(results), 0)
        self.assertIn("content", results[0])
        self.assertIn("score", results[0])
        self.assertIn("metadata", results[0])

    def test_answer_question_returns_standard_shape_without_openai(self):
        result = answer_question("Cac hinh thuc cai nghien ma tuy la gi?", use_openai=False)

        self.assertIn("answer", result)
        self.assertIn("sources", result)
        self.assertIn("metadata", result)
        self.assertGreater(len(result["sources"]), 0)

    def test_vector_store_builds_and_searches(self):
        index = build_vector_index()
        results = vector_search("Dieu 251 mua ban trai phep chat ma tuy", top_k=3)
        status = vector_store_status()

        self.assertGreater(index["record_count"], 0)
        self.assertGreater(len(results), 0)
        self.assertTrue(status["exists"])
        self.assertIn("embedding_model", status)

    def test_retrieve_vector_mode_returns_sources(self):
        results = retrieve("cac hinh thuc cai nghien ma tuy", top_k=3, mode="hybrid_vector")

        self.assertGreater(len(results), 0)
        self.assertIn("score", results[0])


@unittest.skipIf(TestClient is None, "FastAPI is not installed")
class FastAPITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("hybrid_vector", data["retrieval_modes"])

    def test_chat_endpoint_returns_contract_without_openai(self):
        response = self.client.post(
            "/api/chat",
            json={
                "query": "Điều 249 quy định gì?",
                "top_k": 3,
                "retrieval_mode": "hybrid_vector",
                "use_reranking": True,
                "use_openai": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertIn("sources", data)
        self.assertIn("metadata", data)
        self.assertGreater(len(data["sources"]), 0)
        self.assertIn("score", data["sources"][0])


if __name__ == "__main__":
    unittest.main()
