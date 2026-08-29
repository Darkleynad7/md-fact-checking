# Project Plan: Automated Fact-Checking for the Republic of Moldova

## 1. Project Overview
This project aims to build, evaluate, and compare multiple Machine Learning and AI agent pipelines for automated fact-checking of Romanian-language claims (specifically targeting the Republic of Moldova). The outputs will form the basis of a dissertation report comparing foundational discriminative models against generative Large Language Models (LLMs) and multi-hop retrieval agents.

## 2. Directory Structure
Ensure the project adheres to the following structure:
```text
project_root/
│
├── notebooks/
│   ├── 01_data_ingestion_and_preprocessing.ipynb
│   ├── 02_discriminative_baseline_xlmr.ipynb
│   ├── 03_generative_lora_verdict.ipynb
│   ├── 04_multihop_rag_agent.ipynb
│   └── 05_evaluation_and_ablation.ipynb
│
├── data/
│   ├── raw/                  
│   └── processed/            
│
├── src/
│   ├── scraper.py            
│   ├── tools.py              
│   └── metrics.py            
│
└── project_plan.md
```

## 3. Data Sources

The dataset must consist of Romanian language claims and evidence:
 - EuroVerdict: Extract the Romanian subset containing claims, fact-checking articles, and manually written verdicts.  
 - PolyTruth: Extract the Romanian subset (approx. 1,000 statements) which pairs false claims with factual corrections.  
 - Hugging Face: Retrieve the rares127/dezinformare-ro dataset.
 - Scraping Fallback: If more data is needed, implement a scraper for stopfals.md.
 
##  4. Phase-by-Phase Implementation 
### Phase 1: Data Ingestion & Preprocessing (01_data_ingestion_and_preprocessing.ipynb)      
  - Task: Load EuroVerdict, PolyTruth, and Hugging Face datasets into pandas DataFrames.  
  
  - Cleaning: Remove HTML tags, resolve missing metadata, and standardize Romanian diacritics (ș/ş, ț/ţ).  
  - Schema: Standardize columns to claim_id, claim_text, evidence_text, veracity_label, and justification.  
  - Splitting: Create stratified 80/10/10 train/validation/test splits. Ensure no article overlaps across splits to prevent temporal or factual data leakage. Save to data/processed/.  
### Phase 2: Discriminative Transformer Baselines (02_discriminative_baseline_xlmr.ipynb)
 - Task: Fine-tune encoder-only multilingual models (xlm-roberta-base and google/rembert) for 3-way classification.  
 - Architecture: Add a sequence classification head on top of the [CLS] token. 
 - Hyperparameters: AdamW optimizer, learning rate $2 \times 10^{-5}$, linear warmup ratio 0.03, batch size 16.  
 - Output: Export models, validation loss, multi-class Accuracy, and Macro-F1.  
 
### Phase 3: Generative LoRA Verdict Generation (03_generative_lora_verdict.ipynb)
 - Task: Implement Parameter-Efficient Fine-Tuning (LoRA) on meta-llama/Llama-3.1-8B-Instruct.  
 - Configuration: 4-bit quantization, LoRA rank $r=16$, alpha $\alpha=16$, dropout $0.05$, targeting attention modules.  
 - Prompting: Use the EuroVerdict Romanian prompt structure instructing the agent to act as an expert fact-checker, evaluate claims based solely on context, and output a verdict and justification.  
 - Baselines: Include Zero-Shot and 5-step Chain-of-Thought (CoT) prompting on the un-tuned base model for baseline comparisons.  

### Phase 4: Multi-Hop Retrieval & Agent Pipeline (04_multihop_rag_agent.ipynb)
 - Task: Build a Retrieval-Augmented Generation (RAG) agent combining dynamic sub-question generation, tool-calling, and CoT verification.  
 - Retrieval: Index evidence using multilingual-e5-base embeddings and BM25.  
 - Workflow:
  - Decompose: LLM breaks the complex claim into 2–4 factual sub-questions.  
  - Retrieve: Agent executes search tools to gather facts for each sub-question.  
  - Synthesize: Aggregate snippets into an evidence history.  
  - Verify: Run a CoT reasoning template to assign the final veracity label and justification in Romanian.  

### Phase 5: Evaluation & Ablation (05_evaluation_and_ablation.ipynb)
 - Task: Evaluate all pipelines on the held-out test set.  
 - Classification Metrics: Calculate Accuracy, Macro-F1, and per-class F1.  
 - NLG Metrics: Calculate ROUGE-1, ROUGE-L, multilingual BERTScore, and Cosine Similarity for justifications.  
 - Retrieval Metrics: Calculate Question Recall and Evidence Recall.  
 - Outputs: Generate comparative plots and LaTeX tables for the dissertation report. Save failure cases for error analysis.