import unittest
import os
from src.utils.io_utils import read_text

class TestGeneration(unittest.TestCase):
    def setUp(self):
        self.output_dir = "data/outputs/primary_reports"
        self.formatted_dir = "data/outputs/formatted_reports"

    def test_primary_generation(self):
        self.assertTrue(os.path.exists(self.output_dir))
        for filename in os.listdir(self.output_dir):
            if filename.endswith(".txt"):
                report = read_text(os.path.join(self.output_dir, filename))
                self.assertIn("Findings", report)
                self.assertIn("Impression", report)

    def test_secondary_generation(self):
        self.assertTrue(os.path.exists(self.formatted_dir))
        for filename in os.listdir(self.formatted_dir):
            if filename.endswith(".txt"):
                report = read_text(os.path.join(self.formatted_dir, filename))
                self.assertIn("Findings", report)
                self.assertIn("Impression", report)

if __name__ == "__main__":
    unittest.main()