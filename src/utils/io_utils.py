import json
import os

def read_json(file_path):
    """Read JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_text(file_path, content):
    """Write text to file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def read_text(file_path):
    """Read text from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()