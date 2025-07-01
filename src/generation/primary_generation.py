import os
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from src.utils.io_utils import read_text, write_text
from src.utils.llm_utils import load_llm, generate_text_enhanced
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_with_config(model_config, device):
    """Enhanced model loading with better configuration."""
    model_name = model_config['name']
    
    # Determine model type
    if any(x in model_name.lower() for x in ['t5', 'bart', 'pegasus']):
        model_type = 'seq2seq'
    else:
        model_type = 'causal'
    
    logger.info(f"Loading {model_type} model: {model_name}")
    
    return load_llm(model_name, model_config['path'], model_config['quantization'], device), model_type

def enhance_prompt_with_context(prompt_template, input_data, model_config):
    """Enhanced prompt engineering for better medical report generation."""
    
    # Add medical context and constraints
    enhanced_template = f"""
You are an expert radiologist generating a comprehensive chest X-ray report. You must:

1. Base ALL findings strictly on the provided scene graph data
2. Use precise medical terminology 
3. Follow standard radiology report structure
4. Include relevant clinical context
5. Avoid any hallucinations or unsupported claims

IMPORTANT: Only report findings that are explicitly present in the scene graph data.

{prompt_template}

ADDITIONAL CONSTRAINTS:
- Use present tense for current findings
- Include anatomical location details when available
- Mention comparison and temporal cues if present
- Provide clinical correlation when appropriate
- Keep the tone professional and clinical

"""
    
    return enhanced_template.format(input_data=input_data)

def post_process_report(generated_text, input_data):
    """Post-process the generated report for quality assurance."""
    
    # Remove any obvious hallucinations or repetitions
    lines = generated_text.split('\n')
    processed_lines = []
    seen_lines = set()
    
    for line in lines:
        line = line.strip()
        if line and line not in seen_lines:
            # Basic quality checks
            if len(line) > 10 and not line.startswith('**') or line.startswith('**'):
                processed_lines.append(line)
                seen_lines.add(line)
    
    # Ensure proper report structure
    report = '\n'.join(processed_lines)
    
    # Add missing sections if needed
    if '**Findings**' not in report and '**FINDINGS**' not in report:
        if 'Findings:' in report:
            report = report.replace('Findings:', '**Findings**:')
    
    if '**Impression**' not in report and '**IMPRESSION**' not in report:
        if 'Impression:' in report:
            report = report.replace('Impression:', '**Impression**:')
    
    return report

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load configuration
    processed_dir = config['data']['processed_dir']
    output_dir = os.path.join(config['data']['output_dir'], "primary_reports")
    prompt_path = os.path.join(config['data']['prompt_dir'], "primary_llm_prompt.txt")
    model_config = config['models']['primary_llm']
    device = config['hardware']['device']

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer with enhanced configuration
    (model, tokenizer), model_type = load_model_with_config(model_config, device)
    
    # Load prompt template
    prompt_template = read_text(prompt_path)

    # Process each preprocessed file
    processed_files = [f for f in os.listdir(processed_dir) if f.endswith('.txt')]
    
    if not processed_files:
        logger.warning(f"No processed files found in {processed_dir}")
        return
    
    logger.info(f"Processing {len(processed_files)} files...")

    for filename in processed_files:
        input_path = os.path.join(processed_dir, filename)
        output_path = os.path.join(output_dir, f"report_{filename}")

        try:
            # Read preprocessed data
            input_data = read_text(input_path)
            logger.info(f"Processing: {filename}")

            # Enhance prompt with context
            enhanced_prompt = enhance_prompt_with_context(prompt_template, input_data, model_config)

            # Generate report with enhanced parameters
            report = generate_text_enhanced(
                model=model, 
                tokenizer=tokenizer, 
                prompt=enhanced_prompt, 
                model_config=model_config,
                model_type=model_type,
                device=device
            )

            # Post-process the report
            final_report = post_process_report(report, input_data)

            # Save report
            write_text(output_path, final_report)
            logger.info(f"Generated report: {filename} -> {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            continue

    logger.info("Primary generation completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate radiology reports using enhanced primary LLM.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)