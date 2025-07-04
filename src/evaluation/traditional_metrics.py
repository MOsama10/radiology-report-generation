# import os
# import yaml
# from src.utils.io_utils import read_text
# from src.evaluation.metrics import compute_bleu, compute_rouge, compute_bertscore

# def main(config_path):
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)

#     # Load configuration
#     generated_dir = os.path.join(config['data']['output_dir'], "primary_reports")
#     ground_truth_dir = config['data']['ground_truth_dir']
#     output_dir = config['evaluation']['output_dir']
#     metrics = config['evaluation']['metrics']

#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # Initialize results
#     results = {metric: [] for metric in metrics}

#     # Evaluate each generated report
#     for filename in os.listdir(generated_dir):
#         if filename.endswith(".txt"):
#             generated_path = os.path.join(generated_dir, filename)
#             ground_truth_path = os.path.join(ground_truth_dir, f"ground_truth_{filename.replace('report_processed_', '')}")

#             if not os.path.exists(ground_truth_path):
#                 print(f"No ground truth found for {filename}")
#                 continue

#             # Read reports
#             generated_report = read_text(generated_path)
#             ground_truth_report = read_text(ground_truth_path)

#             # Compute metrics
#             if "bleu" in metrics:
#                 bleu_score = compute_bleu(generated_report, ground_truth_report)
#                 results["bleu"].append((filename, bleu_score))
#             if "rouge" in metrics:
#                 rouge_scores = compute_rouge(generated_report, ground_truth_report)
#                 results["rouge"].append((filename, rouge_scores))
#             if "bertscore" in metrics:
#                 bertscore = compute_bertscore(generated_report, ground_truth_report)
#                 results["bertscore"].append((filename, bertscore))

#     # Save results
#     output_path = os.path.join(output_dir, "evaluation_results.txt")
#     with open(output_path, 'w') as f:
#         for metric, scores in results.items():
#             f.write(f"{metric.upper()} Scores:\n")
#             for filename, score in scores:
#                 f.write(f"{filename}: {score}\n")
#             f.write("\n")
#     print(f"Evaluation results saved to {output_path}")

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Evaluate generated radiology reports.")
#     parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
#     args = parser.parse_args()
#     main(args.config)
import os
from src.utils.io_utils import read_text
from src.evaluation.metrics import (
    compute_bleu,
    compute_rouge,
    compute_bertscore,
    compute_meteor,
    compute_semantic_similarity,
)

def main():
    # ✅ EDIT THESE PATHS DIRECTLY:
    generated_path = r"data\outputs\primary_reports\report_processed_00f11709-ce2637cf-d7e74371-0fe48659-f3c88f78_SceneGraph.txt"
    ground_truth_path = r"data\ground_truth\ground_truth_s53547436.txt"
    output_dir = r"data\evaluation"
    metrics = ["bleu", "rouge", "bertscore", "meteor", "semantic"]

    # Validate files exist
    if not os.path.exists(generated_path):
        raise FileNotFoundError(f"Generated file not found: {generated_path}")
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Load report contents
    generated_report = read_text(generated_path)
    ground_truth_report = read_text(ground_truth_path)

    results = {}

    # Compute BLEU
    if "bleu" in metrics:
        bleu_score = compute_bleu(generated_report, ground_truth_report)
        results["bleu"] = round(bleu_score, 4)

    # Compute ROUGE
    if "rouge" in metrics:
        rouge_scores = compute_rouge(generated_report, ground_truth_report)
        rouge_scores = {k: round(v, 4) for k, v in rouge_scores.items()}
        results["rouge"] = rouge_scores

    # Compute BERTScore
    if "bertscore" in metrics:
        bertscore = compute_bertscore(generated_report, ground_truth_report)
        bertscore = {k: round(v, 4) for k, v in bertscore.items()}
        results["bertscore"] = bertscore

    # Compute METEOR
    if "meteor" in metrics:
        meteor_score = compute_meteor(generated_report, ground_truth_report)
        results["meteor"] = round(meteor_score, 4)

    # Compute Semantic Similarity
    if "semantic" in metrics:
        semantic_score = compute_semantic_similarity(generated_report, ground_truth_report)
        results["semantic"] = round(semantic_score, 4)

    # Save results to file
    output_path = os.path.join(output_dir, "evaluation_results.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for metric, score in results.items():
            f.write(f"{metric.upper()} Score:\n")
            if isinstance(score, dict):
                for k, v in score.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{score}\n")
            f.write("\n")

    print(f"\n✅ Evaluation results saved to {output_path}")

if __name__ == "__main__":
    main()
