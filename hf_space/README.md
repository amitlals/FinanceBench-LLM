---
title: FinanceBench-LLM Financial QA
emoji: "\U0001F4CA"
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: true
license: mit
tags:
  - finance
  - nvidia
  - lora
  - llm
  - sec-filings
  - financial-qa
---

# FinanceBench-LLM: Domain-Adapted Financial QA

Built with **NVIDIA NIM**, **NeMo Customizer** (LoRA fine-tuning), and evaluated with **LLM-as-a-Judge** on the [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) dataset.

## Results

| Model | Exact Match | F1 Score | ELO Rating |
|-------|-------------|----------|------------|
| Base (Llama-3.1-8B) | 0.23 | 0.41 | 835 |
| + ICL (5-shot) | 0.34 | 0.56 | 1023 |
| + LoRA Fine-tuned | **0.52** | **0.71** | **1142** |

**+126% Exact Match improvement** through LoRA fine-tuning on financial QA data.

## Links

- [GitHub Repository](https://github.com/amitlals/FinanceBench-LLM)
- [NVIDIA DLI Course](https://www.nvidia.com/en-us/training/)
