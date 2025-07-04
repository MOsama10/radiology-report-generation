import json
import yaml
import jsonschema
from collections import defaultdict, Counter
from jsonschema.exceptions import ValidationError
from src.utils.io_utils import read_json, write_text
import sys
import os
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def load_schema(schema_path):
    """Load JSON schema for validation."""
    if os.path.exists(schema_path):
        return read_json(schema_path)
    return None

def validate_scene_graph(data, schema):
    """Validate scene graph JSON against schema."""
    if schema is None:
        return True
    try:
        jsonschema.validate(data, schema)
        return True
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        return False

def extract_findings_from_triplets(triplets, threshold=0.01):
    """Extract and organize findings from triplets with confidence scoring."""
    findings = defaultdict(list)
    
    for triplet in triplets:
        subject = triplet.get("subject", "")
        predicate = triplet.get("predicate", "")
        obj = triplet.get("object", "")
        score = triplet.get("score", 0.0)
        
        # Filter by confidence threshold
        if score < threshold:
            continue
            
        # Clean up object names
        clean_object = clean_finding_name(obj)
        
        # Organize by anatomical region
        if subject and predicate and clean_object:
            finding_info = {
                "finding": clean_object,
                "predicate": predicate,
                "confidence": round(score, 3),
                "raw_object": obj
            }
            findings[subject].append(finding_info)
    
    return findings

def clean_finding_name(finding):
    """Clean and standardize finding names."""
    # Remove prefixes
    finding = re.sub(r'^(anatomicalfinding_|technicalassessment_|nlp_)', '', finding)
    
    # Replace underscores with spaces
    finding = finding.replace('_', ' ')
    
    # Capitalize appropriately
    finding = finding.title()
    
    return finding

def group_findings_by_region(findings):
    """Group findings by anatomical regions and consolidate similar findings."""
    regional_findings = defaultdict(lambda: defaultdict(list))
    
    for subject, finding_list in findings.items():
        # Group positive and negative findings
        positive_findings = []
        negative_findings = []
        
        for finding_info in finding_list:
            if finding_info["predicate"] == "has":
                positive_findings.append(finding_info)
            elif finding_info["predicate"] == "does_not_have":
                negative_findings.append(finding_info)
        
        # Sort by confidence
        positive_findings.sort(key=lambda x: x["confidence"], reverse=True)
        negative_findings.sort(key=lambda x: x["confidence"], reverse=True)
        
        regional_findings[subject]["positive"] = positive_findings
        regional_findings[subject]["negative"] = negative_findings
    
    return regional_findings

def generate_clinical_interpretation(regional_findings):
    """Generate clinical interpretation from findings."""
    interpretations = []
    
    for region, finding_types in regional_findings.items():
        region_summary = []
        
        # Process positive findings
        positive = finding_types.get("positive", [])
        if positive:
            # Get most confident positive findings
            top_positive = [f for f in positive if f["confidence"] > 0.01][:3]
            if top_positive:
                findings_text = ", ".join([f["finding"] for f in top_positive])
                confidence_avg = sum([f["confidence"] for f in top_positive]) / len(top_positive)
                region_summary.append(f"shows {findings_text} (confidence: {confidence_avg:.2f})")
        
        # Process negative findings (only high confidence ones)
        negative = finding_types.get("negative", [])
        high_conf_negative = [f for f in negative if f["confidence"] > 0.01]
        if high_conf_negative and not positive:
            region_summary.append("appears normal")
        
        if region_summary:
            interpretations.append(f"**{region.title()}**: {' and '.join(region_summary)}")
    
    return interpretations

def create_enhanced_prompt(data):
    """Create enhanced prompt from triplet-based scene graph data."""
    
    # Extract basic metadata
    image_name = data.get("image_name", "Unknown")
    parameters = data.get("parameters", {})
    
    # Extract findings from triplets
    triplets = data.get("triplets", [])
    # threshold = parameters.get("obj_threshold", 0.01)
    threshold = 0.01
    
    logger.info(f"Processing {len(triplets)} triplets with threshold {threshold}")
    
    findings = extract_findings_from_triplets(triplets, threshold)
    regional_findings = group_findings_by_region(findings)
    
    # Get available subjects, predicates, and objects
    subjects = data.get("subjects", [])
    predicates = data.get("predicates", [])
    objects = data.get("objects", [])
    
    # Build enhanced prompt
    prompt_lines = []
    
    # Header with metadata
    prompt_lines.append("**RADIOLOGY SCENE GRAPH ANALYSIS**")
    prompt_lines.append(f"**Image ID**: {image_name}")
    prompt_lines.append(f"**Analysis Parameters**: Object threshold: {threshold}")
    prompt_lines.append("")
    
    # Summary statistics
    total_positive = sum(len(f.get("positive", [])) for f in regional_findings.values())
    total_negative = sum(len(f.get("negative", [])) for f in regional_findings.values())
    prompt_lines.append(f"**Finding Summary**: {total_positive} positive findings, {total_negative} negative findings across {len(regional_findings)} anatomical regions")
    prompt_lines.append("")
    
    # Detailed findings by region
    prompt_lines.append("**ANATOMICAL FINDINGS BY REGION**:")
    
    for region, finding_types in regional_findings.items():
        prompt_lines.append(f"\n**{region.upper()}**:")
        
        # Positive findings
        positive = finding_types.get("positive", [])
        if positive:
            prompt_lines.append("  Positive findings:")
            for finding in positive[:10]:  # Limit to top 5
                prompt_lines.append(f"    - {finding['finding']} (confidence: {finding['confidence']})")
        
        # High-confidence negative findings
        negative = [f for f in finding_types.get("negative", []) if f["confidence"] > 0.025]
        if negative and len(negative) <= 3:  # Only show if few negative findings
            prompt_lines.append("  Notable negative findings:")
            for finding in negative:
                prompt_lines.append(f"    - No {finding['finding']} (confidence: {finding['confidence']})")
    
    # Clinical interpretation
    interpretations = generate_clinical_interpretation(regional_findings)
    if interpretations:
        prompt_lines.append("\n**CLINICAL INTERPRETATION**:")
        for interpretation in interpretations:
            prompt_lines.append(f"  {interpretation}")
    
    # Technical details
    prompt_lines.append("\n**TECHNICAL DETAILS**:")
    prompt_lines.append(f"  - Available anatomical regions: {', '.join(subjects)}")
    prompt_lines.append(f"  - Finding types analyzed: {len(objects)} different findings")
    prompt_lines.append(f"  - Confidence threshold applied: {threshold}")
    
    # Top findings summary
    all_positive_findings = []
    for region_findings in regional_findings.values():
        all_positive_findings.extend(region_findings.get("positive", []))
    
    if all_positive_findings:
        all_positive_findings.sort(key=lambda x: x["confidence"], reverse=True)
        top_findings = all_positive_findings[:5]
        
        prompt_lines.append("\n**HIGHEST CONFIDENCE FINDINGS**:")
        for i, finding in enumerate(top_findings, 1):
            prompt_lines.append(f"  {i}. {finding['finding']} (confidence: {finding['confidence']})")
    
    return "\n".join(prompt_lines)

def create_legacy_format_prompt(data):
    """Create prompt in legacy format for compatibility."""
    
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

def detect_input_format(data):
    """Detect the format of input data (triplet-based vs legacy)."""
    
    if "triplets" in data and isinstance(data["triplets"], list):
        return "triplet"
    elif "attributes" in data and isinstance(data["attributes"], list):
        return "legacy"
    else:
        logger.warning("Unknown input format, defaulting to triplet format")
        return "triplet"

def preprocess_scene_graph(data):
    """Enhanced preprocessing that handles multiple input formats."""
    
    input_format = detect_input_format(data)
    logger.info(f"Detected input format: {input_format}")
    
    if input_format == "triplet":
        return create_enhanced_prompt(data)
    else:
        return create_legacy_format_prompt(data)

def main(config_path):
    """Main preprocessing function with enhanced capabilities."""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    raw_dir = config['data']['raw_dir']
    processed_dir = config['data']['processed_dir']
    schema_path = config['preprocessing']['schema_file']

    # Create directories if they don't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load schema
    schema = load_schema(schema_path)
    if schema:
        logger.info("Schema loaded successfully")
    else:
        logger.warning(f"Schema file not found at {schema_path}, skipping validation")

    # Check if raw directory exists
    if not os.path.exists(raw_dir):
        logger.error(f"Raw data directory {raw_dir} does not exist")
        logger.info("Please create the directory and add your scene graph JSON files")
        return

    # Process each JSON file in raw_dir
    json_files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]
    
    if not json_files:
        logger.warning(f"No JSON files found in {raw_dir}")
        logger.info("Please add scene graph JSON files to process")
        return
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    for filename in json_files:
        input_path = os.path.join(raw_dir, filename)
        output_path = os.path.join(processed_dir, f"processed_{filename.replace('.json', '.txt')}")

        try:
            logger.info(f"Processing {filename}...")
            
            # Load scene graph
            data = read_json(input_path)
            
            # Validate if schema available
            if schema and not validate_scene_graph(data, schema):
                logger.warning(f"Validation failed for {filename}, but continuing processing")

            # Enhanced preprocessing
            processed_text = preprocess_scene_graph(data)
            
            # Save output
            write_text(output_path, processed_text)
            logger.info(f"Successfully processed {filename} -> {os.path.basename(output_path)}")
            
            # Log some statistics
            lines = processed_text.split('\n')
            logger.info(f"Generated {len(lines)} lines of processed text")
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            continue

    logger.info("Enhanced preprocessing completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced preprocessing for scene graph JSON files.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    main(args.config)
