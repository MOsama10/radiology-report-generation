from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from transformers import BitsAndBytesConfig

def load_llm(model_name, model_path, quantization, device):
    """Load LLM and tokenizer with quantization."""
    quantization_config = None
    if quantization == "4bit":
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    if "t5" in model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if device == "cuda" else None
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if device == "cuda" else None
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_tokens, device):
    """Generate text using LLM."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)