Radiology Report Generation Project
This project generates radiology reports from scene graph data using large language models (LLMs). The pipeline includes preprocessing, primary report generation, summarization/reformatting, and evaluation.
Setup

Run scripts/setup_env.bat to create a virtual environment and install dependencies.
Place raw scene graph JSON files in data/raw/.
Configure model paths and settings in config/config.yaml.
Run scripts/run_pipeline.bat to execute the pipeline.

Directory Structure

config/: Configuration files and LLM prompts.
data/: Raw and processed data, ground truth, and outputs.
models/: LLM model weights.
src/: Source code for preprocessing, generation, evaluation, and utilities.
tests/: Unit tests for pipeline components.
scripts/: Scripts for setup and pipeline execution.

Requirements

Python 3.8+
NVIDIA GPU (e.g., RTX 4060) for CUDA support
See requirements.txt for Python dependencies

Usage

Place scene graph JSON files in data/raw/.
Run scripts/run_pipeline.bat to process data, generate reports, and evaluate results.
Check data/outputs/ for generated reports and evaluation metrics.
