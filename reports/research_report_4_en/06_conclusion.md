# 6. Conclusion 

## 6.1 Findings
The progression from Sequence Classification to Parameter-Efficient Generative LLMs highlights a fundamental boundary in Automated Fact-Checking.
1. **Discriminative Encoders (XLM-R, RemBERT)** are highly efficient and reach localized mathematical perfection on binary tasks. However, in heavily skewed real-world databases (e.g. Stopfals.md), they collapse under extreme Class Imbalance, sacrificing nuanced inference to maximize aggregate mathematical scoring.
2. **Generative LoRA Architectures (Llama-3.1-8B)** demonstrate a resilient foundational intelligence. They easily achieve >85% balanced semantic evaluation when provided human-curated evidence.
3. **Autonomous RAG Retrieval is highly viable.** By combining a Sparse BM25 index with a Dense E5 semantic vector space, the autonomous framework successfully fetched the correct journalistic evidence at an 88.75% recall rate, pushing the Verification accuracy to a peak of 87.57%.

## 6.2 Architectural Caveats
The transition to Autonomous multi-hop verification exposed critical flaws in contemporary RAG doctrines when applied to low-resource political fact-checking:
* **The Danger of Sub-Question Decomposition:** Breaking down complex claims artificially fragmented the semantic context. Providing the full, un-decomposed propaganda narrative to the Hybrid Retriever produced significantly sharper contextual evidence.
* **Out-of-Distribution Sensitivity:** LLM Verification weights tuned via LoRA are mathematically hyper-sensitive. Slight changes in the text format caused by algorithmic concatenation or `Chain-of-Thought` instructions entirely disabled the adapter logic. Masking the RAG output to mimic the exact bounds of the training distribution (1000 characters limitation, zero brackets) is strictly required.
* **The Failure of Semantic Search:** Dense vectors fail in factual verification because they index thematic similarity, not factual ground-truth directionality. Only by fusing them with strict lexical token-matchers (BM25) can an LLM safely navigate propaganda. 

This research successfully proves that localized, moderately-sized LLMs can autonomously identify, query, and verify political misinformation with fidelity approaching human journalistic standards.
