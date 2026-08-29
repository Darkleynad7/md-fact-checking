# Moldova Automated Fact-Checking

Automated fact-checking pipeline for Romanian-language claims from the Republic of Moldova, built as a dissertation project comparing discriminative transformers, generative LLMs, and multi-hop RAG agents.

---

## Project Structure

```
md-fact-checking/
│
├── notebooks/
│   ├── 01_data_ingestion_and_preprocessing.ipynb
│   ├── 02_discriminative_baseline_xlmr.ipynb
│   ├── 03_generative_lora_verdict.ipynb
│   ├── 04_multihop_rag_agent.ipynb
│   └── 05_evaluation_and_ablation.ipynb
│
├── data/
│   ├── raw/          ← raw data downloads / scrape output
│   ├── processed/    ← cleaned splits + model predictions
│   ├── models/       ← saved model checkpoints
│   └── figures/      ← plots and LaTeX tables
│
├── src/
│   ├── scraper.py    ← stopfals.md scraper
│   ├── tools.py      ← BM25 / dense / hybrid retriever, CoT prompt builders
│   └── metrics.py    ← classification, NLG, and retrieval metric utilities
│
├── requirements.txt
└── project_plan.md
```

---

## Pipelines

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_data_ingestion_and_preprocessing` | Load EuroVerdict, PolyTruth, dezinformare-ro; clean Romanian diacritics; stratified 80/10/10 split |
| 2 | `02_discriminative_baseline_xlmr` | Fine-tune `xlm-roberta-base` and `google/rembert` for 3-way classification |
| 3 | `03_generative_lora_verdict` | LoRA fine-tune `Llama-3.1-8B-Instruct`; zero-shot and CoT baselines |
| 4 | `04_multihop_rag_agent` | Decompose → Retrieve → Synthesize → Verify RAG agent |
| 5 | `05_evaluation_and_ablation` | Full evaluation, comparative plots, LaTeX tables, error analysis |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run notebooks in order

```bash
jupyter notebook notebooks/
```

### 3. Optional: Scrape additional data from stopfals.md

```bash
python src/scraper.py --pages 20 --output data/raw/stopfals.jsonl
```

---

## Data Sources

| Dataset | Language | Size | Access |
|---------|----------|------|--------|
| [EuroVerdict](https://huggingface.co/datasets/Cartinoe5930/EuroVerdict) | Romanian subset | ~2k claims | HuggingFace |
| [PolyTruth](https://huggingface.co/datasets/Cartinoe5930/PolyTruth) | Romanian subset | ~1k claims | HuggingFace |
| [dezinformare-ro](https://huggingface.co/datasets/rares127/dezinformare-ro) | Romanian | varies | HuggingFace |
| [stopfals.md](https://stopfals.md) | Romanian | scraped | `src/scraper.py` |

---

## Models

| Model | Role | Source |
|-------|------|--------|
| `xlm-roberta-base` | Discriminative classifier | HuggingFace |
| `google/rembert` | Discriminative classifier | HuggingFace |
| `meta-llama/Llama-3.1-8B-Instruct` | Generative verdict + LoRA | HuggingFace (gated) |
| `intfloat/multilingual-e5-base` | Dense retrieval embeddings | HuggingFace |

> **Note**: Access to `meta-llama/Llama-3.1-8B-Instruct` requires accepting the Meta license at [huggingface.co](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and setting `HF_TOKEN` in your environment.

---

## Evaluation Metrics

- **Classification**: Accuracy, Macro-F1, per-class F1
- **NLG**: ROUGE-1, ROUGE-L, multilingual BERTScore, Cosine Similarity
- **Retrieval**: Evidence Recall, Question Recall

---

## Hardware Requirements

| Component | Minimum |
|-----------|---------|
| Discriminative fine-tuning | 16 GB GPU (e.g. A10G) |
| Llama LoRA fine-tuning | 40 GB GPU (e.g. A100) with 4-bit quantisation |
| Dense indexing | 8 GB GPU or CPU with ~10 min build time |

---

## Environment Variables

```bash
export HF_TOKEN=<your_huggingface_access_token>
```
