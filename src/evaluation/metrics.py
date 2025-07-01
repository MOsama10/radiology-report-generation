from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import bert_score

def compute_bleu(generated, reference):
    """Compute BLEU score."""
    reference = [reference.split()]
    generated = generated.split()
    return sentence_bleu(reference, generated)

def compute_rouge(generated, reference):
    """Compute ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, generated)

def compute_bertscore(generated, reference):
    """Compute BERTScore."""
    P, R, F1 = bert_score.score([generated], [reference], lang="en", rescale_with_baseline=True)
    return {"precision": P.item(), "recall": R.item(), "f1": F1.item()}