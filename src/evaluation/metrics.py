# import nltk
# from nltk.translate.bleu_score import sentence_bleu
# from evaluate import load
# import bert_score

# # Download required NLTK data
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# def compute_bleu(generated, reference):
#     """Compute BLEU score."""
#     # Tokenize properly
#     reference_tokens = [nltk.word_tokenize(reference.lower())]
#     generated_tokens = nltk.word_tokenize(generated.lower())
#     return sentence_bleu(reference_tokens, generated_tokens)

# def compute_rouge(generated, reference):
#     """Compute ROUGE scores using evaluate library."""
#     rouge = load('rouge')
#     scores = rouge.compute(predictions=[generated], references=[reference])
#     return scores

# def compute_bertscore(generated, reference):
#     """Compute BERTScore."""
#     P, R, F1 = bert_score.score([generated], [reference], lang="en", rescale_with_baseline=True)
#     return {"precision": P.item(), "recall": R.item(), "f1": F1.item()}
import nltk
from nltk.translate.bleu_score import sentence_bleu
from evaluate import load
import bert_score
from sentence_transformers import SentenceTransformer, util

# Download NLTK resources if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load models globally to avoid reloading each call
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
meteor = load("meteor")
rouge = load("rouge")

def compute_bleu(generated, reference):
    """Compute BLEU score."""
    reference_tokens = [nltk.word_tokenize(reference.lower())]
    generated_tokens = nltk.word_tokenize(generated.lower())
    return sentence_bleu(reference_tokens, generated_tokens)

def compute_rouge(generated, reference):
    """Compute ROUGE scores."""
    return rouge.compute(predictions=[generated], references=[reference])

def compute_meteor(generated, reference):
    """Compute METEOR score."""
    return meteor.compute(predictions=[generated], references=[reference])["meteor"]

def compute_bertscore(generated, reference):
    """Compute BERTScore with baseline rescaling."""
    P, R, F1 = bert_score.score([generated], [reference], lang="en", rescale_with_baseline=True)
    return {"precision": round(P.item(), 4), "recall": round(R.item(), 4), "f1": round(F1.item(), 4)}

def compute_semantic_similarity(generated, reference):
    """Compute cosine similarity between sentence embeddings."""
    emb1 = sentence_model.encode(generated, convert_to_tensor=True)
    emb2 = sentence_model.encode(reference, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return similarity
