# 3. Hybrid RAG Architecture & Vector Interfacing

To grant the Llama instance autonomous access to factual archives without connecting to real-time external APIs, an offline vector database was synthesized comprising roughly 1,671 individual journalistic documents from `StopFals.md`.

## 3.1 The Hybrid Retrieval Paradigm
A major vulnerability in low-resource language Natural Language Processing is extreme morphological inflection (Romanian contains complex diacritics and verb conjugations). 
To ensure maximum factual recall, a dual-layer retrieval system was coded:
1. **Lexical (BM25Okapi):** Utilizes sparse Bag-of-Words matrices weighted by Term/Inverse-Document Frequencies. This ensures that highly specific identifiers (e.g., proper nouns, dates, exact geographical locations like "Găgăuzia") are instantly mathematically locked.
2. **Dense Semantic (Multilingual-E5-Base):** Utilizes the Microsoft E5 Transformer vector space to map contextual meaning. This allows the system to fetch evidence even if the propaganda replaces literal keywords with malicious synonyms.

## 3.2 Solving the Dense Memory Collapse
During initial engineering iterations, the FAISS Embedded Dense Retriever suffered heavily degraded semantic matching. Deep algorithmic debugging revealed a crucial API constraint inside the internal E5 network architecture. The E5 model requires strict instruction boundaries embedded directly into its dense arrays to orient its spatial mapping.
* **The Optimization:** The codebase was structurally patched to violently inject the string `"passage: "` into the multidimensional array prior to encoding the corpus, and identically prefix all runtime Llama queries with `"query: "`.
* **The Result:** This MLOps fix completely stabilized the internal FAISS topology. As validated in the logs, the hybrid agent's `evidence_recall` metric skyrocketed to an exceptional **88.75%**.

## 3.3 Sub-Question Decomposition
Instead of brutally throwing the raw propaganda claim into the Retrieval space, the pipeline initializes the zero-shot base `Llama` model and orders it to break down the claim into 3 distinct atomic queries. 
For example, if the propaganda asserts *"NATO invaded Ukraine and Georgia"*, Llama successfully generates mathematical sub-queries such as:
1. *Did NATO invade Ukraine?*
2. *Did NATO invade Georgia in 2008?*

This logic simulates a human journalist checking disparate facts before synthesizing a conclusion. The results from all queries are processed in parallel through a Reciprocal Rank Fusion (RRF) algorithm to extract the top snippets.
