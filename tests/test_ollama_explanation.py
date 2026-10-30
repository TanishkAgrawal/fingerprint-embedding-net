import json
import unittest
from io import BytesIO
from unittest.mock import patch

from src.retrieval.ollama_explanation import explain_comparison


class OllamaExplanationTests(unittest.TestCase):
    def setUp(self):
        self.matches = [{'Rank': 1, 'Image ID': 12, 'Cosine Similarity': .92,
                         'Same finger': True, 'Fingerprint ID': '102_Left_middle'}]

    @patch('src.retrieval.ollama_explanation.urllib.request.urlopen')
    def test_sends_grounded_comparison_to_local_model(self, send):
        send.return_value = BytesIO(json.dumps({'response': ' A descriptive result. '}).encode())
        result = explain_comparison(self.matches, {'Image ID': 7}, ' llama3.2 ')
        self.assertEqual(result, 'A descriptive result.')
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, 'http://127.0.0.1:11434/api/generate')
        payload = json.loads(request.data)
        self.assertEqual(payload['model'], 'llama3.2')
        self.assertFalse(payload['stream'])
        facts = json.loads(payload['prompt'].split('Facts:\n')[1])
        self.assertEqual(facts['ranked_comparisons'], self.matches)
        self.assertIsNone(facts['threshold'])

    @patch('src.retrieval.ollama_explanation.urllib.request.urlopen')
    def test_unavailable_service_propagates_to_ui_handler(self, send):
        send.side_effect = OSError('Connection refused')
        with self.assertRaises(OSError):
            explain_comparison(self.matches, {}, 'llama3.2')

    @patch('src.retrieval.ollama_explanation.urllib.request.urlopen')
    def test_empty_response_is_rejected(self, send):
        send.return_value = BytesIO(b'{"response": ""}')
        with self.assertRaisesRegex(ValueError, 'empty or invalid'):
            explain_comparison(self.matches, {}, 'llama3.2')

    def test_missing_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'model name'):
            explain_comparison(self.matches, {}, ' ')


if __name__ == '__main__':
    unittest.main()
