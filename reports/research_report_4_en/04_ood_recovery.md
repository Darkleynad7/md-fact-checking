# 4. Out-of-Distribution (OOD) Recovery & Alignment

While the agent's Retrieval phase performed flawlessly (fetching correct truth context at a rate of 88.75%), the Verification node—powered by our LoRA weights—initially experienced a catastrophic collapse.

## 4.1 The Hallucination of LoRA Adapters
Internal diagnostics revealed that the system dropped to an aggregate Accuracy of ~13%, heavily outputting the `unknown` class. The root cause was identified as severe **Out-of-Distribution (OOD) Format Collapse**.

In Notebook 3, the LoRA Matrixes were minimized (Loss function configured correctly) by reading strict, clean strings of expert text strictly capped at 1,000 characters. 
In Notebook 4, the RAG chunks agglomerated into strings approaching 1,500 words (or ~9,000 characters), riddled with programmatic brackets such as `[1] Snippet... [2] Snippet...`. Further exacerbating the issue, researchers attempted to force the LoRA module to output `Chain-of-Thought` (CoT) reasoning before delivering a final verdict.
Because the specialized weights had never seen `Chain-of-Thought` or 9,000-character bracketed arrays during Backpropagation, the Attention Head mappings mathematically broke down, outputting complete gibberish and inducing cascading parsing failures.

## 4.2 The Surgical Alignment Fixes
To rescue the multi-hop sequence without re-training the intensive LoRA weights from scratch, a triad of structural "formatting disguises" were deployed into the Python architecture:

1. **Adapter Toggling (Zero-Shot Isolation):** Since the specialized LoRA weights cannot reason broadly enough to create generic questions, the HuggingFace `peft` interface was modified to execute `llm.disable_adapter()` strictly during the atomic Sub-Question generation step, relying entirely on the foundational zero-shot Llama parameter intelligence.
2. **Strict RAG Truncation:** The `aggregate_evidence` algorithm was patched to violently strip all programming identifiers (brackets) and rigidly slice the resulting concatenated string at exactly `1,000` characters. 
3. **Prompt Restoration:** The Chain-of-Thought commands were entirely deleted from the `VERIFY_PROMPT`.

To the downstream LoRA network, the incoming RAG-fetched data now visually and statistically mimicked the exact 1000-character, non-reasoned structural boundaries of its original Notebook 3 training environment.
