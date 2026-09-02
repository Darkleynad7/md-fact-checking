# 1. Introduction: Towards Autonomous Fact-Checking

## 1.1 The Knowledge Bottleneck in Automated Fact-Checking
The surge of disinformation within Eastern Europe naturally necessitates scalable, automated verification systems. However, a critical limitation in current Natural Language Processing (NLP) solutions is the assumption of a "Gold Evidence" paragraph. Classical SOTA frameworks measure how precisely a foundational model can identify propaganda *if and only if* a human expert has already manually sourced, extracted, and provided the perfectly correlated factual text. 

For an Automated Fact-Checking (AFC) system to be deployed effectively in real-world journalism or social media filtering, it cannot rely on human-fed evidence. It must act as fully autonomous agents: reading a fake news claim, querying large databases (e.g., historical archives, Wikipedia, verified journalist hubs), extracting the relevant nuance, and formally publishing a verifiable conclusion.

## 1.2 Research Objectives for the Multi-Hop RAG Agent
This study formally investigates the feasibility of transforming a localized, offline Large Language Model (Llama-3.1-8B-Instruct) into a fully Autonomous Multi-Hop Agent targeting the Romanian informational sphere.
The primary objectives of this phase are:
1. **Implementation of Hybrid RAG:** Construct a Dual-Retrieval Augmented Generation pipeline combining Lexical (BM25) and Dense (Multilingual-E5) vector spaces.
2. **Sub-Question Decomposition:** Evaluate whether Llama can autonomously break down complex propaganda into mathematically searchable atomic queries.
3. **Out-of-Distribution (OOD) Recovery:** Diagnose and solve the catastrophic architectural collapse observed when feeding chaotic multi-hop retrieved data into strict Instruction-Tuned (LoRA) prediction layers.
4. **Agentic Ablation Studies:** Quantitatively isolate the necessity of Dense vs. Sparse retrieval methodologies in low-resource (Romanian) languages.
