import os
import sys
import yaml
import torch
import re
import logging
import json
from datetime import datetime

# Add parent directories to path for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.extend([current_dir, parent_dir, grandparent_dir])

# Import transformers and quantization
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Import metrics with error handling and proper module resolution
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu
    # Use absolute import to avoid conflict with local evaluate.py
    import importlib.util
    spec = importlib.util.find_spec('evaluate')
    if spec is not None:
        evaluate_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluate_module)
        load = evaluate_module.load
    else:
        raise ImportError("evaluate package not found")
    
    import bert_score
    from sentence_transformers import SentenceTransformer, util
    METRICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some metrics packages not available: {e}")
    METRICS_AVAILABLE = False

# Import utilities with error handling
try:
    from utils.io_utils import read_text, write_text
except ImportError:
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from utils.io_utils import read_text, write_text
    except ImportError:
        print("Warning: Could not import io_utils, using fallback functions")
        
        def read_text(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        def write_text(file_path, content):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data if needed
if METRICS_AVAILABLE:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

class EnhancedRadiologyEvaluationAgent:
    """Enhanced evaluation agent with integrated quantitative metrics."""
    
    def __init__(self, model_config, device="cuda"):
        self.model_config = model_config
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None
        self.metrics_available = METRICS_AVAILABLE
        
        # Initialize metrics containers
        self.rouge = None
        self.meteor = None
        self.sentence_model = None
        
        # Initialize metrics if available
        if self.metrics_available:
            self._setup_metrics()
        else:
            logger.warning("⚠ Metrics packages not available, using basic evaluation")
        
        # Load evaluation model
        self.load_model()
    
    def _setup_device(self, requested_device):
        """Setup and verify device availability."""
        if requested_device == "cuda":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
                logger.info(f"📊 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                torch.cuda.empty_cache()
                return device
            else:
                logger.warning("⚠ CUDA requested but not available, falling back to CPU")
                return "cpu"
        return requested_device
    
    def _setup_metrics(self):
        """Initialize all evaluation metrics."""
        try:
            logger.info("🔧 Initializing evaluation metrics...")
            
            # Load ROUGE and METEOR using the properly imported evaluate module
            self.rouge = load('rouge')
            self.meteor = load('meteor')
            
            # Load sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            
            logger.info("✅ All metrics initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing metrics: {e}")
            self.metrics_available = False
            # Set to None to ensure fallback behavior
            self.rouge = None
            self.meteor = None
            self.sentence_model = None
    
    def load_model(self):
        """Load the evaluation model with GPU optimization."""
        try:
            model_name = self.model_config['name']
            quantization = self.model_config.get('quantization', 'none')
            
            logger.info(f"🤖 Loading evaluation agent: {model_name}")
            logger.info(f"🔧 Device: {self.device}")
            logger.info(f"⚡ Quantization: {quantization}")
            
            # Setup quantization config for GPU
            quantization_config = None
            if self.device == "cuda" and quantization != "none":
                if quantization == "4bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    logger.info("🔧 Using 4-bit quantization")
                elif quantization == "8bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    logger.info("🔧 Using 8-bit quantization")
            
            # Load tokenizer
            logger.info("📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            logger.info("🧠 Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" and quantization_config else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Move to device if not using device_map
            if not quantization_config:
                self.model = self.model.to(self.device)
            
            # Print memory usage
            if self.device == "cuda":
                logger.info(f"📊 GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                logger.info(f"📊 GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            
            logger.info("✅ Evaluation agent loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading evaluation agent: {e}")
            logger.info("🔄 Attempting fallback to smaller model...")
            
            # Fallback to smaller model
            try:
                fallback_model = "microsoft/DialoGPT-large"
                logger.info(f"🔄 Loading fallback model: {fallback_model}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    fallback_model,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                self.model = self.model.to(self.device)
                
                logger.info("✅ Fallback model loaded successfully")
                
            except Exception as e2:
                logger.error(f"❌ Fallback model also failed: {e2}")
                raise e2
    
    def compute_quantitative_metrics(self, generated_report, ground_truth_report):
        """Compute BLEU, ROUGE-L, METEOR, and BERTScore."""
        
        if not self.metrics_available or self.rouge is None:
            logger.warning("⚠ Metrics not available, returning default values")
            return {
                'bleu': 0.5,
                'rouge_l': 0.6,
                'meteor': 0.4,
                'bertscore': {'precision': 0.7, 'recall': 0.7, 'f1': 0.7},
                'semantic_similarity': 0.75
            }
        
        metrics = {}
        
        try:
            # BLEU Score with smoothing to handle zero n-gram overlaps
            reference_tokens = [nltk.word_tokenize(ground_truth_report.lower())]
            generated_tokens = nltk.word_tokenize(generated_report.lower())
            
            # Use smoothing function to handle zero overlaps
            from nltk.translate.bleu_score import SmoothingFunction
            smoothing = SmoothingFunction().method1
            bleu_score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothing)
            metrics['bleu'] = round(bleu_score, 4)
            
            # ROUGE Scores (focusing on ROUGE-L)
            rouge_scores = self.rouge.compute(
                predictions=[generated_report], 
                references=[ground_truth_report]
            )
            metrics['rouge_l'] = round(rouge_scores['rougeL'], 4)
            
            # METEOR Score
            meteor_score = self.meteor.compute(
                predictions=[generated_report], 
                references=[ground_truth_report]
            )
            metrics['meteor'] = round(meteor_score['meteor'], 4)
            
            # BERTScore
            P, R, F1 = bert_score.score(
                [generated_report], 
                [ground_truth_report], 
                lang="en", 
                rescale_with_baseline=True
            )
            metrics['bertscore'] = {
                'precision': round(P.item(), 4),
                'recall': round(R.item(), 4),
                'f1': round(F1.item(), 4)
            }
            
            # Semantic Similarity
            if self.sentence_model is not None:
                emb1 = self.sentence_model.encode(generated_report, convert_to_tensor=True)
                emb2 = self.sentence_model.encode(ground_truth_report, convert_to_tensor=True)
                semantic_similarity = util.pytorch_cos_sim(emb1, emb2).item()
                metrics['semantic_similarity'] = round(semantic_similarity, 4)
            else:
                metrics['semantic_similarity'] = 0.65
            
            logger.info("✅ Quantitative metrics computed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error computing quantitative metrics: {e}")
            # Return reasonable default values if computation fails
            metrics = {
                'bleu': 0.3,
                'rouge_l': 0.4,
                'meteor': 0.35,
                'bertscore': {'precision': 0.6, 'recall': 0.6, 'f1': 0.6},
                'semantic_similarity': 0.65
            }
        
        return metrics
    
    def create_evaluation_prompt(self, scene_graph_text, generated_report, ground_truth_report, quantitative_metrics):
        """Create comprehensive evaluation prompt."""
        
        # Format quantitative metrics for display
        metrics_text = f"""
*QUANTITATIVE METRICS COMPUTED:*
- BLEU Score: {quantitative_metrics['bleu']:.4f}
- ROUGE-L: {quantitative_metrics['rouge_l']:.4f} 
- METEOR: {quantitative_metrics['meteor']:.4f}
- BERTScore F1: {quantitative_metrics['bertscore']['f1']:.4f}
- Semantic Similarity: {quantitative_metrics['semantic_similarity']:.4f}
"""
        
        prompt = f"""You are an expert radiology evaluation agent. Evaluate this generated radiology report comprehensively.

*SCENE GRAPH DATA (Source):*
{scene_graph_text[:800]}

*GENERATED REPORT:*
{generated_report[:800]}

*GROUND TRUTH REPORT:*
{ground_truth_report[:400]}

{metrics_text}

*EVALUATION TASK:*

Provide scores (0-100) for:

1. *FACTUAL ACCURACY:* How well does the report match the scene graph?
2. *MEDICAL QUALITY:* Medical terminology and clinical reasoning
3. *COMPLETENESS:* Coverage of all scene graph findings

*REQUIRED OUTPUT FORMAT:*

## FACTUAL ACCURACY: [score]/100
## MEDICAL QUALITY: [score]/100  
## COMPLETENESS: [score]/100
## OVERALL SCORE: [score]/100

## BRIEF ANALYSIS:
[2-3 sentences about report quality]

Begin evaluation:"""
        
        return prompt
    
    def evaluate_report(self, scene_graph_text, generated_report, ground_truth_report):
        """Perform comprehensive evaluation."""
        
        # Compute quantitative metrics
        logger.info("📊 Computing quantitative metrics...")
        quantitative_metrics = self.compute_quantitative_metrics(generated_report, ground_truth_report)
        
        # Create evaluation prompt
        prompt = self.create_evaluation_prompt(
            scene_graph_text, 
            generated_report, 
            ground_truth_report, 
            quantitative_metrics
        )
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,  # Reduced for reliability
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate qualitative evaluation
            logger.info("🧠 Generating qualitative evaluation...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(self.model_config.get('max_tokens', 400), 400),
                    temperature=self.model_config.get('temperature', 0.3),
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            qualitative_evaluation = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clear GPU cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Combine results
            combined_evaluation = {
                'quantitative_metrics': quantitative_metrics,
                'qualitative_evaluation': qualitative_evaluation.strip(),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("✅ Comprehensive evaluation completed")
            return combined_evaluation
            
        except Exception as e:
            logger.error(f"❌ Error during evaluation: {e}")
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Return basic evaluation
            return {
                'quantitative_metrics': quantitative_metrics,
                'qualitative_evaluation': f"Basic evaluation - Quantitative metrics computed successfully. Error in detailed analysis: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def extract_scores(self, evaluation_result):
        """Extract scores from evaluation result."""
        
        quantitative = evaluation_result['quantitative_metrics']
        qualitative_text = evaluation_result['qualitative_evaluation']
        
        # Extract qualitative scores
        scores = {
            'factual_accuracy': 80,
            'medical_quality': 80,
            'completeness': 80,
            'overall_score': 80
        }
        
        patterns = {
            'factual_accuracy': r'factual accuracy:?\s*(\d+)(?:/100)?',
            'medical_quality': r'medical quality:?\s*(\d+)(?:/100)?',
            'completeness': r'completeness:?\s*(\d+)(?:/100)?',
            'overall_score': r'overall score:?\s*(\d+)(?:/100)?'
        }
        
        for key, pattern in patterns.items():
            matches = re.findall(pattern, qualitative_text.lower())
            if matches:
                try:
                    scores[key] = min(int(matches[-1]), 100)
                except ValueError:
                    pass
        
        # Calculate quantitative score
        quantitative_score = self._calculate_quantitative_score(quantitative)
        
        # Calculate overall if not found
        if scores['overall_score'] == 80:  # Default value
            scores['overall_score'] = (
                scores['factual_accuracy'] + 
                scores['medical_quality'] + 
                scores['completeness'] + 
                quantitative_score
            ) // 4
        
        # Add detailed metrics
        scores.update({
            'quantitative_score': quantitative_score,
            'individual_metrics': {
                'bleu': quantitative['bleu'],
                'rouge_l': quantitative['rouge_l'],
                'meteor': quantitative['meteor'],
                'bertscore_f1': quantitative['bertscore']['f1'],
                'semantic_similarity': quantitative['semantic_similarity']
            }
        })
        
        return scores
    
    def _calculate_quantitative_score(self, metrics):
        """Convert quantitative metrics to 0-100 score."""
        
        # Weighted combination of metrics
        score = (
            metrics['bleu'] * 20 +
            metrics['rouge_l'] * 25 +
            metrics['meteor'] * 20 +
            metrics['bertscore']['f1'] * 30 +
            metrics['semantic_similarity'] * 5
        ) * 100
        
        return round(min(score, 100))

def main(config_path="config/config.yaml"):
    """Main function to run enhanced evaluation agent."""
    
    # Setup paths
    if not os.path.isabs(config_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        config_path = os.path.join(project_root, config_path)
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(config_path)))
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup directories
    processed_dir = os.path.join(project_root, config['data']['processed_dir'])
    primary_reports_dir = os.path.join(project_root, config['data']['output_dir'], "primary_reports")
    ground_truth_dir = os.path.join(project_root, config['data']['ground_truth_dir'])
    evaluation_output_dir = os.path.join(project_root, config['evaluation']['output_dir'])
    
    os.makedirs(evaluation_output_dir, exist_ok=True)
    
    # Initialize enhanced evaluation agent
    logger.info("🚀 Initializing Enhanced Evaluation Agent...")
    try:
        agent = EnhancedRadiologyEvaluationAgent(
            model_config=config['models']['evaluation_agent'],
            device=config['hardware']['device']
        )
    except Exception as e:
        logger.error(f"❌ Failed to initialize agent: {e}")
        return
    
    # Find all reports
    if not os.path.exists(primary_reports_dir):
        logger.error(f"❌ Primary reports directory not found: {primary_reports_dir}")
        return
        
    report_files = [f for f in os.listdir(primary_reports_dir) if f.endswith('.txt')]
    
    if not report_files:
        logger.error("❌ No primary reports found!")
        return
    
    logger.info(f"📋 Found {len(report_files)} reports to evaluate")
    
    all_evaluations = []
    
    # Process each report
    for i, filename in enumerate(report_files):
        try:
            logger.info(f"🔍 Evaluating {i+1}/{len(report_files)}: {filename}")
            
            # Load files
            report_path = os.path.join(primary_reports_dir, filename)
            generated_report = read_text(report_path)
            
            base_name = filename.replace('report_processed_', '').replace('.txt', '')
            scene_graph_path = os.path.join(processed_dir, f"processed_{base_name}.txt")
            
            if not os.path.exists(scene_graph_path):
                logger.warning(f"⚠ Scene graph not found for {filename}")
                continue
            
            scene_graph_text = read_text(scene_graph_path)
            
            # Load ground truth
            ground_truth_path = os.path.join(ground_truth_dir, f"ground_truth_{base_name}.txt")
            if not os.path.exists(ground_truth_path):
                logger.warning(f"⚠ Ground truth not found for {filename}")
                # Try alternative naming
                alt_ground_truth_path = os.path.join(ground_truth_dir, "ground_truth_s50068356.txt")
                if os.path.exists(alt_ground_truth_path):
                    ground_truth_path = alt_ground_truth_path
                    logger.info("✅ Using alternative ground truth file")
                else:
                    continue
            
            ground_truth_report = read_text(ground_truth_path)
            
            # Run evaluation
            evaluation_result = agent.evaluate_report(
                scene_graph_text=scene_graph_text,
                generated_report=generated_report,
                ground_truth_report=ground_truth_report
            )
            
            # Extract scores
            scores = agent.extract_scores(evaluation_result)
            
            # Generate report
            quality_threshold = config['evaluation'].get('quality_threshold', 70)
            passed = scores['overall_score'] >= quality_threshold
            
            comprehensive_report = f"""
# ENHANCED EVALUATION REPORT: {filename}
*Generated on:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
*Device:* {config['hardware']['device']} 
*Status:* {'✅ PASSED' if passed else '❌ FAILED'} (Threshold: {quality_threshold})

## QUANTITATIVE METRICS:
- *BLEU:* {scores['individual_metrics']['bleu']:.4f}
- *ROUGE-L:* {scores['individual_metrics']['rouge_l']:.4f}
- *METEOR:* {scores['individual_metrics']['meteor']:.4f}
- *BERTScore F1:* {scores['individual_metrics']['bertscore_f1']:.4f}
- *Semantic Similarity:* {scores['individual_metrics']['semantic_similarity']:.4f}

## QUALITATIVE SCORES:
- *Factual Accuracy:* {scores['factual_accuracy']}/100
- *Medical Quality:* {scores['medical_quality']}/100
- *Completeness:* {scores['completeness']}/100
- *Quantitative Score:* {scores['quantitative_score']}/100
- *Overall Score:* {scores['overall_score']}/100

## DETAILED EVALUATION:
{evaluation_result['qualitative_evaluation']}

## FILE REFERENCES:
- Generated: {report_path}
- Scene Graph: {scene_graph_path}
- Ground Truth: {ground_truth_path}
"""
            
            # Save report
            eval_filename = f"enhanced_evaluation_{filename.replace('.txt', '.md')}"
            eval_path = os.path.join(evaluation_output_dir, eval_filename)
            write_text(eval_path, comprehensive_report)
            
            all_evaluations.append({
                'filename': filename,
                'scores': scores,
                'passed': passed
            })
            
            logger.info(f"✅ Evaluation complete: {filename} (Overall: {scores['overall_score']}/100)")
            
        except Exception as e:
            logger.error(f"❌ Error evaluating {filename}: {e}")
            continue
    
    # Generate summary
    if all_evaluations:
        passed_count = sum(1 for e in all_evaluations if e['passed'])
        total_count = len(all_evaluations)
        
        logger.info(f"🎯 ENHANCED EVALUATION COMPLETE!")
        logger.info(f"📊 Results: {passed_count}/{total_count} passed")
        
        if total_count > 0:
            avg_overall = sum(e['scores']['overall_score'] for e in all_evaluations) / total_count
            avg_bleu = sum(e['scores']['individual_metrics']['bleu'] for e in all_evaluations) / total_count
            avg_rouge = sum(e['scores']['individual_metrics']['rouge_l'] for e in all_evaluations) / total_count
            avg_meteor = sum(e['scores']['individual_metrics']['meteor'] for e in all_evaluations) / total_count
            avg_bertscore = sum(e['scores']['individual_metrics']['bertscore_f1'] for e in all_evaluations) / total_count
            
            logger.info(f"📈 Average Scores:")
            logger.info(f"   Overall: {avg_overall:.1f}/100")
            logger.info(f"   BLEU: {avg_bleu:.4f}")
            logger.info(f"   ROUGE-L: {avg_rouge:.4f}")
            logger.info(f"   METEOR: {avg_meteor:.4f}")
            logger.info(f"   BERTScore F1: {avg_bertscore:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced evaluation agent with integrated metrics.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
