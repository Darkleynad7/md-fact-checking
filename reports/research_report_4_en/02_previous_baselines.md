# 2. Brief Summary of Pre-Agent Baselines

Before evaluating the autonomous pipeline, it is essential to contextualize the verification boundaries established during the isolated baseline evaluations.

## 2.1 The "Accuracy Illusion" of Discriminative Encoders
Classical text-classification Encoders (XLM-RoBERTa and RemBERT) were trained to classify claims as `False`, `True`, or `Partially True` based on cross-entropy loss optimizations (Notebook 02). 
While the Encoders achieved an extraordinary **98.8% Global Accuracy**, the metric was highly deceptive. Due to extreme Class Imbalance (the dataset featured ~98% binary True/False claims and only ~1.2% `Partially True` claims), the mathematical weights optimized towards a "Lazy Guesser" algorithm. The models achieved a strict **0.00** F1 Score on nuanced/partially true claims, proving that dense statistical embeddings cannot genuinely reason through complex political language without generative logic.

## 2.2 Generative LoRA on Gold Evidence
To introduce genuine logical synthesis, the foundational Decoder model `Llama-3.1-8B-Instruct` was fine-tuned via Low-Rank Adaptation (LoRA) using 4-bit memory quantization (Notebook 03).
When fed the *Gold Evidence* (a perfect, expert-sourced paragraph of counter-evidence), the LoRA network learned the exact bounds of the verification task.
* **Accuracy:** 85.80%
* **Stable False F1:** 0.8571
* **Stable True F1:** 0.8690

The Generative LoRA network proved it could reliably and honestly extract verdicts. The core question remained: Would these heavily specialized weights survive the transition from reading "Gold Evidence" to reading chaotic, algorithmically-retrieved chunks of text?
