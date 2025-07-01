from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
import torch
import os
import logging

logger = logging.getLogger(__name__)

def load_llm(model_name, model_path, quantization, device):
    """Enhanced LLM loading with better configuration and error handling."""
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Quantization: {quantization}")
    
    # Check if CUDA is available
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Enhanced quantization configuration
    quantization_config = None
    if quantization == "4bit" and device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit" and device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    try:
        # Load tokenizer first
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"  # Better for generation
        )
        
        # Add special tokens if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Determine model type and load accordingly
        logger.info("Loading model...")
        if any(x in model_name.lower() for x in ['t5', 'bart', 'pegasus', 'flan']):
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if device == "cuda" and quantization_config else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if device == "cuda" and quantization_config else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        
        # Move to device if not using device_map
        if device != "cuda" or quantization_config is None:
            model = model.to(device)
        
        logger.info(f"Model loaded successfully on {device}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Trying fallback model...")
        
        # Fallback to smaller model
        fallback_models = [
            "microsoft/DialoGPT-medium",
            "t5-small", 
            "distilgpt2"
        ]
        
        for fallback_model in fallback_models:
            try:
                logger.info(f"Trying fallback: {fallback_model}")
                tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                if 't5' in fallback_model:
                    model = AutoModelForSeq2SeqLM.from_pretrained(fallback_model)
                else:
                    model = AutoModelForCausalLM.from_pretrained(fallback_model)
                
                model = model.to(device)
                logger.info(f"Fallback model {fallback_model} loaded successfully")
                return model, tokenizer
            except:
                continue
        
        raise Exception("All model loading attempts failed")

def generate_text_enhanced(model, tokenizer, prompt, model_config, model_type, device):
    """Enhanced text generation with better parameters and error handling."""
    
    try:
        # Enhanced generation configuration
        generation_config = GenerationConfig(
            max_new_tokens=model_config.get('max_tokens', 512),
            temperature=model_config.get('temperature', 0.3),
            top_p=model_config.get('top_p', 0.85),
            top_k=model_config.get('top_k', 50),
            repetition_penalty=model_config.get('repetition_penalty', 1.1),
            length_penalty=model_config.get('length_penalty', 1.0),
            do_sample=True,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            output_scores=False,
            return_dict_in_generate=False
        )
        
        # Tokenize input with proper attention handling
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,  # Adjust based on model capacity
            padding=True,
            add_special_tokens=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with enhanced configuration
        logger.info("Generating text...")
        with torch.no_grad():
            if model_type == 'seq2seq':
                # For T5, BART, etc.
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    generation_config=generation_config
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            else:
                # For GPT-like models
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    generation_config=generation_config
                )
                
                # Remove input prompt from output
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs[0][input_length:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Post-process the generated text
        generated_text = post_process_generated_text(generated_text)
        
        logger.info("Text generation completed successfully")
        return generated_text.strip()
        
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        return f"Error generating text: {str(e)}"

def post_process_generated_text(text):
    """Post-process generated text to improve quality."""
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove repetitive patterns (basic)
    lines = text.split('\n')
    unique_lines = []
    seen_lines = set()
    
    for line in lines:
        line = line.strip()
        if line and line not in seen_lines:
            unique_lines.append(line)
            seen_lines.add(line)
    
    # Rejoin lines
    text = '\n'.join(unique_lines)
    
    # Basic formatting improvements
    text = text.replace('**Findings**:', '\n**Findings**:\n')
    text = text.replace('**Impression**:', '\n**Impression**:\n')
    text = text.replace('**Patient**:', '\n**Patient**:')
    
    return text
