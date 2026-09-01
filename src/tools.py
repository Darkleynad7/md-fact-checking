"""
tools.py
Search and retrieval tool helpers for the multi-hop RAG agent pipeline.
Provides BM25 and dense (multilingual-e5-base) retrieval over an evidence corpus,
plus a sub-question decomposition helper.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# BM25 retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Thin wrapper around rank_bm25.BM25Okapi.
    Tokenises Romanian text, supports top-k retrieval.
    """

    def __init__(self, corpus: List[str]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError("Install rank-bm25: pip install rank-bm25") from exc
        self._corpus = corpus
        self._tokenised = [self._tokenise(doc) for doc in corpus]
        self._index = BM25Okapi(self._tokenised)

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return [(doc_text, score)] sorted by descending BM25 score."""
        tokens = self._tokenise(query)
        scores = self._index.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self._corpus[i], float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Dense retriever (multilingual-e5-base)
# ---------------------------------------------------------------------------

class DenseRetriever:
    """
    FAISS-backed dense retriever using multilingual-e5-base embeddings.
    Falls back to a brute-force cosine scan when faiss is unavailable.
    """

    MODEL_NAME = "intfloat/multilingual-e5-base"

    def __init__(
        self,
        corpus: List[str],
        device: Optional[str] = None,
        batch_size: int = 64,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers: pip install sentence-transformers"
            ) from exc

        self._corpus = corpus
        self._model = SentenceTransformer(self.MODEL_NAME, device=device)
        print(f"[DenseRetriever] Encoding {len(corpus)} documents …")
        self._embeddings: np.ndarray = self._model.encode(
            [f"passage: {c}" for c in corpus],
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        self._index = self._build_index()

    def _build_index(self):
        try:
            import faiss

            dim = self._embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(self._embeddings.astype(np.float32))
            return index
        except ImportError:
            return None  # fall back to numpy cosine search

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return [(doc_text, score)] sorted by descending cosine similarity."""
        q_emb = self._model.encode(
            [f"query: {query}"], normalize_embeddings=True
        ).astype(np.float32)

        if self._index is not None:
            scores, indices = self._index.search(q_emb, top_k)
            return [
                (self._corpus[int(i)], float(s))
                for i, s in zip(indices[0], scores[0])
            ]
        # Brute-force fallback
        sims = (self._embeddings @ q_emb.T).squeeze()
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self._corpus[i], float(sims[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Hybrid retriever (BM25 + dense RRF fusion)
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Reciprocal Rank Fusion of BM25 and dense scores.
    """

    def __init__(
        self,
        corpus: List[str],
        device: Optional[str] = None,
        rrf_k: int = 60,
    ) -> None:
        self._corpus = corpus
        self._bm25 = BM25Retriever(corpus)
        self._dense = DenseRetriever(corpus, device=device)
        self._rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        bm25_results = self._bm25.retrieve(query, top_k=top_k * 2)
        dense_results = self._dense.retrieve(query, top_k=top_k * 2)

        scores: Dict[str, float] = {}

        def rrf(rank: int) -> float:
            return 1.0 / (self._rrf_k + rank + 1)

        for rank, (doc, _) in enumerate(bm25_results):
            scores[doc] = scores.get(doc, 0.0) + rrf(rank)
        for rank, (doc, _) in enumerate(dense_results):
            scores[doc] = scores.get(doc, 0.0) + rrf(rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ---------------------------------------------------------------------------
# Sub-question decomposition helper
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = """\
<|system|>
Ești un verificator de fapte expert.<|end|>
<|user|>
Descompune afirmația de mai jos în {n} sub-întrebări factuale care, dacă sunt răspunzute, permit verificarea completă a afirmației.
Returnează DOAR o listă JSON de strings (sub-întrebările), fără explicații suplimentare.

Afirmație: {claim}

Sub-întrebări JSON:<|end|>
<|assistant|>
"""


def decompose_claim(
    claim: str,
    llm_fn,
    n: int = 3,
) -> List[str]:
    """
    Use a callable `llm_fn(prompt: str) -> str` to decompose a claim into
    sub-questions. Returns a list of question strings.
    `llm_fn` should accept a plain string prompt and return the model's text reply.
    """
    prompt = DECOMPOSE_PROMPT.format(claim=claim, n=n)
    raw = llm_fn(prompt)
    # Parse the first JSON array found in the response
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            questions = json.loads(match.group(0))
            return [str(q) for q in questions if q]
        except json.JSONDecodeError:
            pass
    # Fallback: split on newlines / numbering
    lines = [
        re.sub(r"^\s*[\d\-\.\)]+\s*", "", line).strip()
        for line in raw.splitlines()
        if line.strip()
    ]
    return [l for l in lines if l][:n]


# ---------------------------------------------------------------------------
# Evidence aggregator
# ---------------------------------------------------------------------------

def aggregate_evidence(snippets: List[Tuple[str, float]], max_tokens: int = 200) -> str:
    """
    Concatenate retrieved snippets into a single evidence string.
    Hard-capped at 1000 chars and removes brackets to perfectly match the
    LoRA training distribution from Notebook 3.
    """
    parts = []
    total = 0
    for i, (text, score) in enumerate(snippets, 1):
        words = text.split()
        if total + len(words) > max_tokens:
            break
        parts.append(text.strip())
        total += len(words)
    # Join with spaces (no brackets) and strictly truncate to 1000 chars exactly like training
    return " ".join(parts)[:1000]


# ---------------------------------------------------------------------------
# CoT verification prompt
# ---------------------------------------------------------------------------

VERIFY_PROMPT = """\
<|system|>
Ești un expert verificator de fapte pentru Republica Moldova. Bazează-te EXCLUSIV pe dovezile primite.<|end|>
<|user|>
Analizează afirmația: {claim}

Dovezi multiple:
{evidence}

Gândire pas cu pas (Chain-of-Thought):
1. Extrage informațiile din dovezi care susțin afirmația.
2. Extrage informațiile din dovezi care contrazic afirmația.
3. Sintetizează și alege verdictul cel mai obiectiv.

Verdict final: [ADEVĂRAT / FALS / PARȚIAL_ADEVĂRAT]
Justificare: [Max 3 propoziții în română]<|end|>
<|assistant|>
"""


def build_verify_prompt(claim: str, evidence: str) -> str:
    return VERIFY_PROMPT.format(claim=claim, evidence=evidence)


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------

LABEL_MAP = {
    "adevărat": "true",
    "adevarat": "true",
    "fals": "false",
    "parțial_adevărat": "partially_true",
    "partial_adevarat": "partially_true",
    "parțial adevărat": "partially_true",
    "partial adevarat": "partially_true",
}


def parse_verdict(model_output: str) -> Tuple[str, str]:
    """
    Extract (label, justification) from model output.
    Returns label as one of: 'true', 'false', 'partially_true', or 'unknown'.
    """
    text = model_output.strip()
    
    # Isolate the verdict section from Chain-of-Thought
    verdict_text = text
    for marker in ["Verdict final:", "Verdict:", "Verdict final (", "verdictul final este"]:
        if marker.lower() in text.lower():
            verdict_text = text.lower().split(marker.lower())[-1]
            break

    # Fix substring collision: sort keys descending by length
    label = "unknown"
    sorted_labels = sorted(LABEL_MAP.items(), key=lambda x: len(x[0]), reverse=True)
    
    for ro_key, en_val in sorted_labels:
        if ro_key in verdict_text.lower():
            label = en_val
            break

    # Extract justification after "Justificare:" marker
    just_match = re.search(r"Justificare[:\s]+(.+)", text, re.DOTALL | re.IGNORECASE)
    justification = just_match.group(1).strip() if just_match else text
    return label, justification
