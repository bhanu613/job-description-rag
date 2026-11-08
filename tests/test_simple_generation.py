import json
import unittest
from pathlib import Path
from unittest.mock import patch
import rag

P = [{"chunk_id": "A-C000", "text": "Python is required.", "metadata": {"source_id": "A"}}]


def fake(payload, reason="stop"):
    def chat(**kwargs):
        return {"message": {"content": json.dumps(payload)}, "done_reason": reason}
    return chat


class SimpleGenerationTests(unittest.TestCase):
    def test_answer_and_reference(self):
        r = rag.generate_answer("What skill?", P, chat_fn=fake({"answer": "Python", "source_ids": ["A-C000"]}))
        self.assertEqual((r.status, r.answer, r.source_ids, r.attempts), ("answered", "Python", ["A-C000"], 1))

    def test_abstention(self):
        r = rag.generate_answer("Salary?", P, chat_fn=fake({"answer": rag.MISSING_MESSAGE, "source_ids": []}))
        self.assertEqual(r.status, "insufficient_evidence")

    def test_no_context_no_model_call(self):
        def forbidden(**kwargs): raise AssertionError("Model must not be called")
        self.assertEqual(rag.generate_answer("Question", [], chat_fn=forbidden).status, "no_context")

    def test_empty_input(self):
        self.assertEqual(rag.generate_answer(" ", P).status, "invalid_input")

    def test_invalid_question_type(self):
        self.assertEqual(rag.generate_answer(None, P).status, "invalid_input")

    def test_unknown_reference(self):
        with self.assertRaises(ValueError):
            rag.validate_response({"answer": "Python", "source_ids": ["B"]}, P)

    def test_no_references_for_factual_answer(self):
        with self.assertRaises(ValueError):
            rag.validate_response({"answer": "Python", "source_ids": []}, P)

    def test_abstention_with_reference(self):
        with self.assertRaises(ValueError):
            rag.validate_response({"answer": rag.MISSING_MESSAGE, "source_ids": ["A-C000"]}, P)

    def test_duplicate_references_deduplicated(self):
        answer, ids = rag.validate_response({"answer": "Python", "source_ids": ["A-C000", "A-C000"]}, P)
        self.assertEqual(ids, ["A-C000"])

    def test_wrong_answer_type(self):
        with self.assertRaises(ValueError):
            rag.validate_response({"answer": 3, "source_ids": ["A-C000"]}, P)

    def test_empty_answer(self):
        with self.assertRaises(ValueError):
            rag.validate_response({"answer": " ", "source_ids": ["A-C000"]}, P)

    def test_wrong_reference_container(self):
        with self.assertRaises(ValueError):
            rag.validate_response({"answer": "Python", "source_ids": "A-C000"}, P)

    def test_wrong_reference_type(self):
        with self.assertRaises(ValueError):
            rag.validate_response({"answer": "Python", "source_ids": [4]}, P)

    def test_extra_fields(self):
        with self.assertRaises(ValueError):
            rag.validate_response({"answer": "Python", "source_ids": ["A-C000"], "claims": []}, P)

    def test_missing_fields(self):
        with self.assertRaises(ValueError): rag.validate_response({"answer": "Python"}, P)

    def test_malformed_json(self):
        def malformed(**kwargs): return {"message": {"content": "oops"}}
        self.assertEqual(rag.generate_answer("Q", P, chat_fn=malformed).status, "validation_failed")

    def test_missing_message_field(self):
        def malformed(**kwargs): return {}
        self.assertEqual(rag.generate_answer("Q", P, chat_fn=malformed).status, "validation_failed")

    def test_service_failure(self):
        def offline(**kwargs): raise ConnectionError("Offline")
        self.assertEqual(rag.generate_answer("Q", P, chat_fn=offline).status, "service_error")

    def test_truncation_rejected(self):
        r=rag.generate_answer("Q", P, chat_fn=fake({"answer":"Python", "source_ids":["A-C000"]}, "length"))
        self.assertEqual(r.status, "validation_failed")

    def test_exactly_one_model_call(self):
        calls=[]
        def chat(**kwargs):
            calls.append(kwargs)
            return fake({"answer":"Python", "source_ids":["A-C000"]})(**kwargs)
        rag.generate_answer("Q", P, chat_fn=chat)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["options"], {"temperature":0,"num_ctx":4096,"num_predict":384})

    def test_no_semantic_guarantee(self):
        # Format/source-membership checks cannot reject an unsupported claim by themselves.
        r=rag.generate_answer("Salary?", P, chat_fn=fake({"answer":"One million euros", "source_ids":["A-C000"]}))
        self.assertEqual(r.status, "answered")

    def test_absolute_database_path(self):
        self.assertTrue(rag.DB_PATH.is_absolute())
        self.assertEqual(rag.DB_PATH, rag.ROOT / "chroma_data")

    def test_empty_retrieval_input_rejected_before_service(self):
        with self.assertRaises(ValueError): rag.retrieve(" ")

    def test_invalid_top_k_rejected_before_service(self):
        for value in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError): rag.retrieve("Q", value)


if __name__ == "__main__":
    unittest.main()
