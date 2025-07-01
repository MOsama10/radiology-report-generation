import os
import yaml
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils.io_utils import read_text, write_text
from src.utils.llm_utils import load_llm, generate_text_enhanced

import re
import logging

logger = logging.getLogger(__name__)

def extract_report_sections(report_text):
    """Extract structured sections from the primary report."""
    
    sections = {
        'patient_info': '',
        'findings': '',
        'impression': ''
    }
    
    # Extract patient information
    patient_match = re.search(r'\*\*Patient\*\*:([^\n]*)', report_text, re.IGNORECASE)
    if patient_match:
        sections['patient_info'] = patient_match.group(1).strip()
    
    # Extract findings section
    findings_match = re.search(r'\*\*Findings\*\*:(.*?)(?=\*\*Impression\*\*|\*\*IMPRESSION\*\*|$)', 
                              report_text, re.IGNORECASE | re.DOTALL)
    if findings_match:
        sections['findings'] = findings_match.group(1).strip()
    
    # Extract impression section
    impression_match = re.search(r'\*\*Impression\*\*:(.*?)$', 
                                report_text, re.IGNORECASE | re.DOTALL)
    if impression_match:
        sections['impression'] = impression_match.group(1).strip()
    
    return sections

def create_structured_prompt(sections, original_report):
    """Create a structured prompt for the secondary LLM."""
    
    structured_prompt = f"""
Reformat and summarize the following radiology report while preserving ALL factual information. 
Create a concise, well-structured report with clear sections.

ORIGINAL REPORT:
{original_report}

INSTRUCTIONS:
1. Preserve all medical findings and clinical details
2. Organize into clear "Findings" and "Impression" sections  
3. Use bullet points for clarity where appropriate
4. Maintain professional medical language
5. Do not add new information not present in the original
6. Do not omit any significant findings

REQUIRED OUTPUT FORMAT:
**Radiology Report**

**Findings**:
[Organized findings by anatomical region]

**Impression**:
[Key clinical conclusions numbered as points]
"""
    
    return structured_prompt

def validate_output_quality(original_report, formatted_report):
    """Validate that the formatted report maintains quality and completeness."""
    
    quality_score = 0
    issues = []
    
    # Check for required sections
    if '**Findings**' in formatted_report or '**FINDINGS**' in formatted_report:
        quality_score += 1
    else:
        issues.append("Missing Findings section")
    
    if '**Impression**' in formatted_report or '**IMPRESSION**' in formatted_report:
        quality_score += 1
    else:
        issues.append("Missing Impression section")
    
    # Check for content preservation (basic word overlap)
    original_words = set(original_report.lower().split())
    formatted_words = set(formatted_report.lower().split())
    overlap_ratio = len(original_words.intersection(formatted_words)) / len(original_words)
    
    if overlap_ratio > 0.6:  # At least 60% word overlap
        quality_score += 1
    else:
        issues.append(f"Low content overlap: {overlap_ratio:.2f}")
    
    # Check length reasonableness
    if 0.3 <= len(formatted_report) / len(original_report) <= 1.2:
        quality_score += 1
    else:
        issues.append("Unreasonable length ratio")
    
    return quality_score, issues

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load configuration
    input_dir = os.path.join(config['data']['output_dir'], "primary_reports")
    output_dir = os.path.join(config['data']['output_dir'], "formatted_reports")
    prompt_path = os.path.join(config['data']['prompt_dir'], "secondary_llm_prompt.txt")
    model_config = config['models']['secondary_llm']
    device = config['hardware']['device']

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if input directory has files
    if not os.path.exists(input_dir):
        logger.error(f"Input directory {input_dir} does not exist. Run primary generation first.")
        return
    
    report_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    if not report_files:
        logger.warning(f"No report files found in {input_dir}")
        return

    # Load model and tokenizer
    model, tokenizer = load_llm(
        model_config,
        model_config["quantization"],
        device
    )


    # Load prompt template (fallback)
    try:
        prompt_template = read_text(prompt_path)
    except:
        logger.warning("Could not load secondary prompt template, using default")
        prompt_template = "Summarize and reformat the following report:"

    logger.info(f"Processing {len(report_files)} primary reports...")

    for filename in report_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"formatted_{filename}")

        try:
            # Read primary report
            input_report = read_text(input_path)
            logger.info(f"Formatting: {filename}")

            # Extract sections for better analysis
            sections = extract_report_sections(input_report)
            
            # Create structured prompt
            structured_prompt = create_structured_prompt(sections, input_report)

            # Generate formatted report
            formatted_report = generate_text_enhanced(
                model=model,
                tokenizer=tokenizer, 
                prompt=structured_prompt,
                model_config=model_config,
                device=device
            )


            # Validate output quality
            quality_score, issues = validate_output_quality(input_report, formatted_report)
            
            if quality_score >= 3:  # Good quality threshold
                write_text(output_path, formatted_report)
                logger.info(f"Generated formatted report: {filename} -> {output_path}")
            else:
                logger.warning(f"Quality issues for {filename}: {issues}")
                # Save anyway but with warning in filename
                warning_path = output_path.replace('.txt', '_WARNING.txt')
                write_text(warning_path, formatted_report)
                logger.info(f"Saved with warning: {warning_path}")
                
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            continue

    logger.info("Secondary generation completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced summarization and reformatting of radiology reports.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)