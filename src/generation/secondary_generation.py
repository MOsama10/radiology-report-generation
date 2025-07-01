import os
import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.utils.io_utils import read_text, write_text
from src.utils.llm_utils import load_llm, generate_text

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

    # Load model and tokenizer
    model, tokenizer = load_llm(model_config['name'], model_config['path'], model_config['quantization'], device)

    # Load prompt template
    prompt_template = read_text(prompt_path)

    # Process each primary report
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"formatted_{filename}")

            # Read primary report
            input_report = read_text(input_path)

            # Prepare prompt
            prompt = prompt_template.format(input_report=input_report)

            # Generate summarized report
            formatted_report = generate_text(model, tokenizer, prompt, max_tokens=model_config['max_tokens'], device=device)

            # Save formatted report
            write_text(output_path, formatted_report)
            print(f"Generated formatted report for {filename} -> {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Summarize and reformat radiology reports using secondary LLM.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)