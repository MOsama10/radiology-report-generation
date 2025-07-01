@echo off
call venv\Scripts\activate
python src\preprocessing\preprocess.py --config config\config.yaml
python src\generation\primary_generation.py --config config\config.yaml
python src\generation\secondary_generation.py --config config\config.yaml
python src\evaluation\evaluate.py --config config\config.yaml