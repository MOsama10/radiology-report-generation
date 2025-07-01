import unittest
import os
from src.preprocessing.preprocess import preprocess_scene_graph, validate_scene_graph
from src.utils.io_utils import read_json

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.sample_data = {
            "image_id": "test",
            "patient_id": 123,
            "gender": "F",
            "age_decile": "70-80",
            "reason_for_exam": "Pneumonia",
            "viewpoint": "AP",
            "StudyDateTime": "2025-07-01",
            "attributes": [
                {
                    "name": "Right lung",
                    "attributes": [[
                        "anatomicalfinding|yes|lung opacity",
                        "disease|yes|pneumonia"
                    ]],
                    "phrases": ["Right lung shows pneumonia."]
                }
            ]
        }
        self.schema_path = "src/preprocessing/schema.json"

    def test_validate_scene_graph(self):
        schema = read_json(self.schema_path)
        self.assertTrue(validate_scene_graph(self.sample_data, schema))

    def test_preprocess_scene_graph(self):
        output = preprocess_scene_graph(self.sample_data)
        self.assertIn("Right lung: lung opacity, pneumonia", output)

if __name__ == "__main__":
    unittest.main()