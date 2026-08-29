"""
metrics.py
Shared evaluation utilities for the Moldova fact-checking project.
Covers classification metrics, NLG metrics (ROUGE, BERTScore, cosine sim),
and retrieval metrics (question recall, evidence recall).
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

LABEL_NAMES = ["false", "partially_true", "true"]


def classification_metrics(
    y_true: Sequence[int | str],
    y_pred: Sequence[int | str],
    label_names: List[str] = LABEL_NAMES,
) -> dict:
    """Return accuracy, macro-F1, and per-class F1 as a dict."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, labels=label_names, zero_division=0
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=label_names,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": {
            lbl: float(per_class_f1[i]) for i, lbl in enumerate(label_names)
        },
        "classification_report": report,
    }


def print_classification_metrics(metrics: dict) -> None:
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Macro-F1 : {metrics['macro_f1']:.4f}")
    print("Per-class F1:")
    for lbl, val in metrics["per_class_f1"].items():
        print(f"  {lbl:20s}: {val:.4f}")


# ---------------------------------------------------------------------------
# NLG metrics
# ---------------------------------------------------------------------------


def rouge_scores(
    predictions: List[str],
    references: List[str],
) -> dict:
    """Compute ROUGE-1 and ROUGE-L using the `rouge_score` library."""
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise ImportError("Install rouge-score: pip install rouge-score") from exc

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    r1_f, rl_f = [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_f.append(scores["rouge1"].fmeasure)
        rl_f.append(scores["rougeL"].fmeasure)
    return {
        "rouge1_f1": float(np.mean(r1_f)),
        "rougeL_f1": float(np.mean(rl_f)),
    }


def bertscore_scores(
    predictions: List[str],
    references: List[str],
    model_type: str = "bert-base-multilingual-cased",
    lang: str = "ro",
    device: Optional[str] = None,
) -> dict:
    """Compute multilingual BERTScore (precision, recall, F1)."""
    try:
        from bert_score import score as bs_score
    except ImportError as exc:
        raise ImportError("Install bert-score: pip install bert-score") from exc

    P, R, F1 = bs_score(
        predictions,
        references,
        model_type=model_type,
        lang=lang,
        device=device,
        verbose=False,
    )
    return {
        "bertscore_precision": float(P.mean()),
        "bertscore_recall": float(R.mean()),
        "bertscore_f1": float(F1.mean()),
    }


def cosine_similarity_scores(
    predictions: List[str],
    references: List[str],
    model_name: str = "intfloat/multilingual-e5-base",
    device: Optional[str] = None,
) -> dict:
    """Compute mean cosine similarity between prediction and reference embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:
        raise ImportError(
            "Install sentence-transformers: pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_name, device=device)
    pred_embs = model.encode(predictions, normalize_embeddings=True, batch_size=32)
    ref_embs = model.encode(references, normalize_embeddings=True, batch_size=32)
    sims = (pred_embs * ref_embs).sum(axis=1)  # dot of L2-normalised = cosine
    return {"cosine_similarity": float(np.mean(sims))}


def nlg_metrics(
    predictions: List[str],
    references: List[str],
    include_bertscore: bool = True,
    include_cosine: bool = True,
    device: Optional[str] = None,
) -> dict:
    """Aggregate all NLG metrics into a single dict."""
    results = rouge_scores(predictions, references)
    if include_bertscore:
        results.update(bertscore_scores(predictions, references, device=device))
    if include_cosine:
        results.update(cosine_similarity_scores(predictions, references, device=device))
    return results


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lowercase and remove punctuation for fuzzy recall matching."""
    return re.sub(r"[^\w\s]", "", text.lower())


def question_recall(
    generated_questions: List[List[str]],
    reference_questions: List[List[str]],
    threshold: float = 0.5,
) -> float:
    """
    For each sample compute what fraction of reference sub-questions are
    'covered' by at least one generated question (token-overlap >= threshold).
    Returns macro-averaged recall across samples.
    """
    sample_recalls = []
    for gen_qs, ref_qs in zip(generated_questions, reference_questions):
        if not ref_qs:
            continue
        gen_tokens = [set(_normalise(q).split()) for q in gen_qs]
        covered = 0
        for ref_q in ref_qs:
            ref_tok = set(_normalise(ref_q).split())
            for gen_tok in gen_tokens:
                overlap = len(ref_tok & gen_tok) / max(len(ref_tok), 1)
                if overlap >= threshold:
                    covered += 1
                    break
        sample_recalls.append(covered / len(ref_qs))
    return float(np.mean(sample_recalls)) if sample_recalls else 0.0


def evidence_recall(
    retrieved_snippets: List[List[str]],
    reference_evidence: List[str],
    threshold: float = 0.3,
) -> float:
    """
    For each sample, check whether the gold evidence string is covered
    (token overlap >= threshold) by any retrieved snippet.
    Returns the fraction of samples where evidence was recalled.
    """
    hits = []
    for snippets, gold in zip(retrieved_snippets, reference_evidence):
        gold_tok = set(_normalise(gold).split())
        found = any(
            len(gold_tok & set(_normalise(s).split())) / max(len(gold_tok), 1)
            >= threshold
            for s in snippets
        )
        hits.append(int(found))
    return float(np.mean(hits)) if hits else 0.0


def retrieval_metrics(
    retrieved_snippets: List[List[str]],
    reference_evidence: List[str],
    generated_questions: Optional[List[List[str]]] = None,
    reference_questions: Optional[List[List[str]]] = None,
) -> dict:
    """
    Compute retrieval metrics.

    evidence_recall is always computed.

    question_recall requires BOTH generated_questions and reference_questions.
    The datasets used in this project (EuroVerdict, PolyTruth, dezinformare-ro)
    do not ship gold sub-questions, so question_recall is only computed when
    a caller explicitly provides both — it is never silently skipped with a
    misleading 0.
    """
    results = {
        "evidence_recall": evidence_recall(retrieved_snippets, reference_evidence)
    }
    if generated_questions is not None and reference_questions is not None:
        results["question_recall"] = question_recall(
            generated_questions, reference_questions
        )
    elif generated_questions is not None and reference_questions is None:
        import warnings
        warnings.warn(
            "generated_questions supplied but reference_questions is None — "
            "question_recall cannot be computed and is omitted from results.",
            UserWarning,
            stacklevel=2,
        )
    return results
