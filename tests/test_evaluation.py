import unittest
from src.evaluation.metrics import compute_bleu, compute_rouge, compute_bertscore

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.generated = "Right lung shows pneumonia."
        self.reference = "Right lung has pneumonia."

    def test_bleu(self):
        score = compute_bleu(self.generated, self.reference)
        self.assertGreater(score, 0.5)

    def test_rouge(self):
        scores = compute_rouge(self.generated, self.reference)
        self.assertGreater(scores["rouge1"].fmeasure, 0.5)

    def test_bertscore(self):
        scores = compute_bertscore(self.generated, self.reference)
        self.assertGreater(scores["f1"], 0.5)

if __name__ == "__main__":
    unittest.main()