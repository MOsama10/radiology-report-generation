import json
import yaml
import jsonschema
from collections import defaultdict
from jsonschema.exceptions import ValidationError
from src.utils.io_utils import read_json, write_text
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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
    # Extract metadata with defaults
    metadata = {
        "patient_id": data.get("patient_id", "Unknown"),
        "gender": data.get("gender", "Unknown"),
        "age_decile": data.get("age_decile", "Unknown"),
        "reason_for_exam": data.get("reason_for_exam", "Unknown"),
        "viewpoint": data.get("viewpoint", "Unknown"),
        "study_date": data.get("StudyDateTime", data.get("study_date", "Unknown"))
    }

    # Group objects and attributes
    object_attributes = defaultdict(list)
    
    # Process attributes safely
    attributes = data.get("attributes", [])
    for attr in attributes:
        obj_name = attr.get("name", "Unknown Object")
        findings = []
        
        # Process attribute lists
        attr_lists = attr.get("attributes", [])
        for attr_list in attr_lists:
            if isinstance(attr_list, list):
                for a in attr_list:
                    if isinstance(a, str) and "|" in a:
                        parts = a.split("|")
                        if len(parts) >= 3 and parts[1] == "yes":
                            findings.append(parts[2])
        
        # Get phrases
        phrases = attr.get("phrases", [])
        if not isinstance(phrases, list):
            phrases = []
        
        # Get additional cues
        comparison_cues = attr.get("comparison_cues", [])
        temporal_cues = attr.get("temporal_cues", [])
        severity_cues = attr.get("severity_cues", [])
        
        object_attributes[obj_name].append({
            "findings": findings, 
            "phrases": phrases,
            "comparison_cues": comparison_cues,
            "temporal_cues": temporal_cues,
            "severity_cues": severity_cues
        })

    # Convert to structured text
    prompt_lines = []
    prompt_lines.append(f"**Patient Context**: A {metadata['age_decile']}-year-old {metadata['gender']} with {metadata['reason_for_exam']}, {metadata['viewpoint']} view, studied on {metadata['study_date']}.")
    prompt_lines.append("**Anatomical Findings**:")
    
    for obj_name, attrs in object_attributes.items():
        for attr in attrs:
            findings_str = ", ".join(attr["findings"]) if attr["findings"] else "No significant findings"
            phrases_str = " ".join(attr["phrases"]) if attr["phrases"] else ""
            
            # Add cues if available
            cues = []
            if attr.get("comparison_cues"):
                cues.extend([f"Comparison: {', '.join(str(c) for c in attr['comparison_cues'])}"])
            if attr.get("temporal_cues"):
                cues.extend([f"Temporal: {', '.join(str(c) for c in attr['temporal_cues'])}"])
            if attr.get("severity_cues"):
                cues.extend([f"Severity: {', '.join(str(c) for c in attr['severity_cues'])}"])
            
            cues_str = " | ".join(cues) if cues else ""
            
            line = f"- {obj_name}: {findings_str}"
            if phrases_str:
                line += f" | Phrase: {phrases_str}"
            if cues_str:
                line += f" | {cues_str}"
            
            prompt_lines.append(line)

    return "\n".join(prompt_lines)

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    raw_dir = config['data']['raw_dir']
    processed_dir = config['data']['processed_dir']
    schema_path = config['preprocessing']['schema_file']

    # Create directories if they don't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load schema
    if os.path.exists(schema_path):
        schema = load_schema(schema_path)
    else:
        print(f"Warning: Schema file not found at {schema_path}, skipping validation")
        schema = None

    # Check if raw directory exists
    if not os.path.exists(raw_dir):
        print(f"Error: Raw data directory {raw_dir} does not exist")
        print("Please create the directory and add your scene graph JSON files")
        return

    # Process each JSON file in raw_dir
    json_files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]
    
    if not json_files:
        print(f"No JSON files found in {raw_dir}")
        print("Please add scene graph JSON files to process")
        return
    
    for filename in json_files:
        input_path = os.path.join(raw_dir, filename)
        output_path = os.path.join(processed_dir, f"processed_{filename.replace('.json', '.txt')}")

        try:
            # Load scene graph
            data = read_json(input_path)
            
            # Validate if schema available
            if schema and not validate_scene_graph(data, schema):
                print(f"Skipping {filename} due to validation failure")
                continue

            # Preprocess and save output
            processed_text = preprocess_scene_graph(data)
            write_text(output_path, processed_text)
            print(f"Processed {filename} -> {output_path}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess scene graph JSON files.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
