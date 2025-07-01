import json
import os
import yaml
import jsonschema
from collections import defaultdict
from jsonschema.exceptions import ValidationError
from src.utils.io_utils import read_json, write_text

def load_schema(schema_path):
    """Load JSON schema for validation."""
    return read_json(schema_path)

def validate_scene_graph(data, schema):
    """Validate scene graph JSON against schema."""
    try:
        jsonschema.validate(data, schema)
        return True
    except ValidationError as e:
        print(f"Validation error: {e}")
        return False

def preprocess_scene_graph(data):
    """Transform scene graph JSON into structured text for LLM."""
    # Extract metadata
    metadata = {
        "patient_id": data.get("patient_id", "Unknown"),
        "gender": data.get("gender", "Unknown"),
        "age_decile": data.get("age_decile", "Unknown"),
        "reason_for_exam": data.get("reason_for_exam", "Unknown"),
        "viewpoint": data.get("viewpoint", "Unknown"),
        "study_date": data.get("StudyDateTime", "Unknown")
    }

    # Group objects and attributes
    object_attributes = defaultdict(list)
    for attr in data.get("attributes", []):
        obj_name = attr.get("name", "Unknown")
        findings = []
        for attr_list in attr.get("attributes", []):
            findings.extend([a.split("|")[2] for a in attr_list if a.split("|")[1] == "yes"])
        phrases = attr.get("phrases", [])
        object_attributes[obj_name].append({"findings": findings, "phrases": phrases})

    # Convert to structured text
    prompt_lines = []
    prompt_lines.append(f"**Patient Context**: A {metadata['age_decile']}-year-old {metadata['gender']} with {metadata['reason_for_exam']}, {metadata['viewpoint']} view, studied on {metadata['study_date']}.")
    prompt_lines.append("**Findings**:")
    for obj_name, attrs in object_attributes.items():
        for attr in attrs:
            findings_str = ", ".join(attr["findings"]) if attr["findings"] else "No significant findings"
            phrases_str = " ".join(attr["phrases"]) if attr["phrases"] else "No additional details"
            prompt_lines.append(f"- {obj_name}: {findings_str}. Phrase: {phrases_str}")

    return "\n".join(prompt_lines)

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    raw_dir = config['data']['raw_dir']
    processed_dir = config['data']['processed_dir']
    schema_path = config['preprocessing']['schema_file']

    # Load schema
    schema = load_schema(schema_path)

    # Ensure output directory exists
    os.makedirs(processed_dir, exist_ok=True)

    # Process each JSON file in raw_dir
    for filename in os.listdir(raw_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(raw_dir, filename)
            output_path = os.path.join(processed_dir, f"processed_{filename.replace('.json', '.txt')}")

            # Load and validate scene graph
            data = read_json(input_path)
            if not validate_scene_graph(data, schema):
                print(f"Skipping {filename} due to validation failure")
                continue

            # Preprocess and save output
            processed_text = preprocess_scene_graph(data)
            write_text(output_path, processed_text)
            print(f"Processed {filename} -> {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess scene graph JSON files.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)