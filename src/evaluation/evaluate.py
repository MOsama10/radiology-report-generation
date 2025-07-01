import os
import yaml
from src.utils.io_utils import read_text
from src.evaluation.metrics import compute_bleu, compute_rouge, compute_bertscore

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load configuration
    generated_dir = os.path.join(config['data']['output_dir'], "formatted_reports")
    ground_truth_dir = config['data']['ground_truth_dir']
    output_dir = config['evaluation']['output_dir']
    metrics = config['evaluation']['metrics']

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize results
    results = {metric: [] for metric in metrics}

    # Evaluate each generated report
    for filename in os.listdir(generated_dir):
        if filename.endswith(".txt"):
            generated_path = os.path.join(generated_dir, filename)
            ground_truth_path = os.path.join(ground_truth_dir, f"ground_truth_{filename.replace('formatted_report_', '')}")

            if not os.path.exists(ground_truth_path):
                print(f"No ground truth found for {filename}")
                continue

            # Read reports
            generated_report = read_text(generated_path)
            ground_truth_report = read_text(ground_truth_path)

            # Compute metrics
            if "bleu" in metrics:
                bleu_score = compute_bleu(generated_report, ground_truth_report)
                results["bleu"].append((filename, bleu_score))
            if "rouge" in metrics:
                rouge_scores = compute_rouge(generated_report, ground_truth_report)
                results["rouge"].append((filename, rouge_scores))
            if "bertscore" in metrics:
                bertscore = compute_bertscore(generated_report, ground_truth_report)
                results["bertscore"].append((filename, bertscore))

    # Save results
    output_path = os.path.join(output_dir, "evaluation_results.txt")
    with open(output_path, 'w') as f:
        for metric, scores in results.items():
            f.write(f"{metric.upper()} Scores:\n")
            for filename, score in scores:
                f.write(f"{filename}: {score}\n")
            f.write("\n")
    print(f"Evaluation results saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate generated radiology reports.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)