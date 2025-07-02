# # scripts/pipeline_controller.py
# """
# Enhanced Pipeline Controller
# Provides programmatic control over the radiology pipeline
# """

# import os
# import sys
# import yaml
# import argparse
# import logging
# from datetime import datetime

# # Add src to path
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# from preprocessing.preprocess import main as preprocess_main
# from generation.primary_generation import main as generation_main
# from evaluation.evaluation_agent import main as evaluation_main

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class PipelineController:
#     """Controller for the enhanced radiology pipeline."""
    
#     def __init__(self, config_path="config/config.yaml"):
#         self.config_path = config_path
#         with open(config_path, 'r') as f:
#             self.config = yaml.safe_load(f)
        
#         logger.info("Pipeline Controller initialized")
#         logger.info(f"Primary LLM: {self.config['models']['primary_llm']['name']}")
#         logger.info(f"Evaluation Agent: {self.config['models']['evaluation_agent']['name']}")
    
#     def run_preprocessing(self):
#         """Run preprocessing step."""
#         logger.info("🔄 STEP 1: Running preprocessing...")
#         try:
#             preprocess_main(self.config_path)
#             logger.info("✅ Preprocessing completed successfully")
#             return True
#         except Exception as e:
#             logger.error(f"❌ Preprocessing failed: {e}")
#             return False
    
#     def run_generation(self):
#         """Run report generation step."""
#         logger.info("🔄 STEP 2: Running report generation...")
#         try:
#             generation_main(self.config_path)
#             logger.info("✅ Report generation completed successfully")
#             return True
#         except Exception as e:
#             logger.error(f"❌ Report generation failed: {e}")
#             return False
    
#     def run_evaluation(self):
#         """Run evaluation agent step."""
#         logger.info("🔄 STEP 3: Running evaluation agent...")
#         try:
#             evaluation_main(self.config_path)
#             logger.info("✅ Evaluation completed successfully")
#             return True
#         except Exception as e:
#             logger.error(f"❌ Evaluation failed: {e}")
#             return False
    
#     def run_full_pipeline(self):
#         """Run the complete pipeline."""
#         logger.info("🚀 STARTING FULL PIPELINE")
#         logger.info("Pipeline: Scene Graph → Primary LLM → Evaluation Agent")
        
#         start_time = datetime.now()
        
#         # Step 1: Preprocessing
#         if not self.run_preprocessing():
#             return False
        
#         # Step 2: Generation  
#         if not self.run_generation():
#             return False
        
#         # Step 3: Evaluation
#         if not self.run_evaluation():
#             return False
        
#         end_time = datetime.now()
#         duration = end_time - start_time
        
#         logger.info(f"🎯 PIPELINE COMPLETED SUCCESSFULLY!")
#         logger.info(f"Total duration: {duration}")
        
#         return True
    
#     def run_step(self, step_name):
#         """Run a specific pipeline step."""
#         steps = {
#             'preprocess': self.run_preprocessing,
#             'generate': self.run_generation,
#             'evaluate': self.run_evaluation,
#             'full': self.run_full_pipeline
#         }
        
#         if step_name not in steps:
#             logger.error(f"Unknown step: {step_name}")
#             logger.info(f"Available steps: {list(steps.keys())}")
#             return False
        
#         return steps[step_name]()

# def main():
#     parser = argparse.ArgumentParser(description="Enhanced Radiology Pipeline Controller")
#     parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
#     parser.add_argument("--step", type=str, default="full", 
#                       choices=['preprocess', 'generate', 'evaluate', 'full'],
#                       help="Pipeline step to run")
#     parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
#     args = parser.parse_args()
    
#     if args.verbose:
#         logging.getLogger().setLevel(logging.DEBUG)
    
#     # Initialize controller
#     controller = PipelineController(args.config)
    
#     # Run specified step
#     success = controller.run_step(args.step)
    
#     if success:
#         print(f"\n✅ {args.step.upper()} step completed successfully!")
#     else:
#         print(f"\n❌ {args.step.upper()} step failed!")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()

# # Usage examples:
# # python scripts/pipeline_controller.py --step full
# # python scripts/pipeline_controller.py --step preprocess  
# # python scripts/pipeline_controller.py --step generate
# # python scripts/pipeline_controller.py --step evaluate
import os
import sys
import yaml
import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineController:
    """Controller for the enhanced radiology pipeline."""
    
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self.project_root = Path(__file__).parent.parent
        
        # Verify config exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info("Pipeline Controller initialized")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Primary LLM: {self.config['models']['primary_llm']['name']}")
        if 'evaluation_agent' in self.config['models']:
            logger.info(f"Evaluation Agent: {self.config['models']['evaluation_agent']['name']}")
    
    def run_subprocess(self, script_path, step_name):
        """Run a Python script as subprocess to avoid import issues."""
        try:
            python_exe = sys.executable
            script_full_path = self.project_root / script_path
            
            if not script_full_path.exists():
                logger.error(f"Script not found: {script_full_path}")
                return False
            
            # Set PYTHONPATH to include project root
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{self.project_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
            
            cmd = [python_exe, str(script_full_path), "--config", self.config_path]
            
            logger.info(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                check=True,
                env=env  # Pass modified environment
            )
            
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"Warnings: {result.stderr}")
            
            logger.info(f"✅ {step_name} completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {step_name} failed with exit code {e.returncode}")
            if e.stdout:
                print(f"Output: {e.stdout}")
            if e.stderr:
                print(f"Error: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"❌ {step_name} failed: {e}")
            return False
    
    def run_preprocessing(self):
        """Run preprocessing step."""
        logger.info("🔄 STEP 1: Running preprocessing...")
        return self.run_subprocess("src/preprocessing/preprocess.py", "Preprocessing")
    
    def run_generation(self):
        """Run report generation step."""
        logger.info("🔄 STEP 2: Running report generation...")
        return self.run_subprocess("src/generation/primary_generation.py", "Report generation")
    
    def run_evaluation(self):
        """Run evaluation agent step."""
        logger.info("🔄 STEP 3: Running evaluation agent...")
        return self.run_subprocess("src/evaluation/evaluation_agent.py", "Evaluation agent")
    
    def run_traditional_metrics(self):
        """Run traditional metrics evaluation."""
        logger.info("🔄 STEP 4: Running traditional metrics...")
        return self.run_subprocess("src/evaluation/evaluate.py", "Traditional metrics")
    
    def run_full_pipeline(self):
        """Run the complete pipeline."""
        logger.info("🚀 STARTING FULL PIPELINE")
        logger.info("Pipeline: Scene Graph → Primary LLM → Evaluation Agent")
        
        start_time = datetime.now()
        
        # Step 1: Preprocessing
        if not self.run_preprocessing():
            logger.error("Pipeline stopped due to preprocessing failure")
            return False
        
        # Step 2: Generation  
        if not self.run_generation():
            logger.error("Pipeline stopped due to generation failure")
            return False
        
        # Step 3: Evaluation
        if not self.run_evaluation():
            logger.warning("Evaluation failed, but continuing...")
        
        # Step 4: Traditional metrics (optional)
        if not self.run_traditional_metrics():
            logger.warning("Traditional metrics failed, but pipeline considered successful")
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"🎯 PIPELINE COMPLETED!")
        logger.info(f"Total duration: {duration}")
        
        # Show results summary
        self.show_results_summary()
        
        return True
    
    def show_results_summary(self):
        """Show summary of pipeline results."""
        try:
            output_dir = self.config['data']['output_dir']
            
            # Count generated reports
            primary_reports_dir = os.path.join(output_dir, "primary_reports")
            if os.path.exists(primary_reports_dir):
                report_count = len([f for f in os.listdir(primary_reports_dir) if f.endswith('.txt')])
                logger.info(f"📊 Generated {report_count} reports")
            
            # Check evaluation results
            if 'evaluation_agent' in self.config['models']:
                eval_dir = self.config.get('evaluation', {}).get('output_dir', 'data/outputs/evaluation_reports')
                if os.path.exists(eval_dir):
                    summary_file = os.path.join(eval_dir, "pipeline_summary.md")
                    if os.path.exists(summary_file):
                        logger.info(f"📋 Evaluation summary: {summary_file}")
            
            # Show key output locations
            logger.info("📁 Key outputs:")
            logger.info(f"  - Generated reports: {primary_reports_dir}")
            if 'evaluation_agent' in self.config['models']:
                logger.info(f"  - Evaluation reports: {eval_dir}")
            
        except Exception as e:
            logger.warning(f"Could not generate results summary: {e}")
    
    def run_step(self, step_name):
        """Run a specific pipeline step."""
        steps = {
            'preprocess': self.run_preprocessing,
            'generate': self.run_generation,
            'evaluate': self.run_evaluation,
            'metrics': self.run_traditional_metrics,
            'full': self.run_full_pipeline
        }
        
        if step_name not in steps:
            logger.error(f"Unknown step: {step_name}")
            logger.info(f"Available steps: {list(steps.keys())}")
            return False
        
        return steps[step_name]()

def main():
    parser = argparse.ArgumentParser(description="Enhanced Radiology Pipeline Controller")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--step", type=str, default="full", 
                      choices=['preprocess', 'generate', 'evaluate', 'metrics', 'full'],
                      help="Pipeline step to run")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize controller
        controller = PipelineController(args.config)
        
        # Run specified step
        success = controller.run_step(args.step)
        
        if success:
            print(f"\n✅ {args.step.upper()} step completed successfully!")
        else:
            print(f"\n❌ {args.step.upper()} step failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline controller failed: {e}")
        print(f"\n❌ Pipeline controller failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()