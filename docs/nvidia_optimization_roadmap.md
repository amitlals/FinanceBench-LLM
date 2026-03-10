# NVIDIA Optimization Roadmap: FinanceBench-LLM

This document outlines the planned NVIDIA technology integrations for production-grade deployment of the FinanceBench-LLM financial QA system.

---

## 1. TensorRT-LLM: 2-4x Inference Speedup

### Overview

TensorRT-LLM compiles the LoRA adapter and base model into an optimized engine for maximum inference throughput on NVIDIA GPUs.

### Implementation Plan

```bash
# Step 1: Convert HF PEFT adapter to TensorRT-LLM format
python -m tensorrt_llm.commands.build \
    --model_dir meta-llama/Llama-3.1-8B-Instruct \
    --lora_dir ./exported_models/lora_adapter \
    --output_dir ./trt_engine \
    --dtype float16 \
    --max_batch_size 8 \
    --max_input_len 1024 \
    --max_output_len 512

# Step 2: Run inference with compiled engine
python -m tensorrt_llm.commands.run \
    --engine_dir ./trt_engine \
    --input "What was Apple's total revenue for FY2023?"
```

### Expected Performance Gains

| Metric | Base (HF Transformers) | TensorRT-LLM | Improvement |
|--------|------------------------|---------------|-------------|
| Latency (p50) | ~2.5s | ~0.6s | **4.2x** |
| Throughput | ~8 tok/s | ~32 tok/s | **4x** |
| GPU Memory | 16GB | 10GB | **37% less** |
| Batch inference (32) | ~80s | ~20s | **4x** |

### Prerequisites

- NVIDIA GPU with 16GB+ VRAM (A10G, A100, H100)
- `tensorrt-llm` >= 0.9.0
- CUDA 12.1+

---

## 2. NIM Multi-LoRA Serving

### Overview

NVIDIA NIM supports deploying multiple LoRA adapters behind a single model endpoint, enabling domain-specific routing without model duplication.

### Architecture

```
                    Single NIM Endpoint
                    (Llama-3.1-8B base)
                           |
              +------------+------------+
              |            |            |
         LoRA: Finance  LoRA: Insurance  LoRA: Compliance
         (FinanceBench)  (InsureBench)   (RegBench)
              |            |            |
         SEC filings    Policy docs    Regulatory text
```

### API Usage

```python
import requests

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Route to finance adapter
response = requests.post(NIM_URL, json={
    "model": "meta/llama-3.1-8b-instruct:financebench-lora",
    "messages": [
        {"role": "user", "content": "What was Apple's revenue?"}
    ],
})

# Route to insurance adapter (same endpoint, different adapter)
response = requests.post(NIM_URL, json={
    "model": "meta/llama-3.1-8b-instruct:insurance-lora",
    "messages": [
        {"role": "user", "content": "What does this policy cover?"}
    ],
})
```

### Benefits

- **Single GPU** serves multiple domain adapters
- **Hot-swapping**: Add/remove adapters without restarting
- **Cost efficient**: One base model in memory, lightweight adapter switching
- **A/B testing**: Route traffic between adapter versions

---

## 3. NeMo Guardrails for Financial Compliance

### Overview

NeMo Guardrails adds safety rails to prevent the financial QA model from generating harmful or non-compliant outputs.

### Planned Guardrail Configuration

```yaml
# config.yml for NeMo Guardrails
models:
  - type: main
    engine: nim
    model: meta/llama-3.1-8b-instruct

rails:
  input:
    flows:
      - check_financial_compliance

  output:
    flows:
      - block_investment_advice
      - flag_hallucinated_numbers
      - require_evidence_citation
      - add_disclaimer

prompts:
  - task: check_financial_compliance
    content: |
      Determine if the user's question is asking for investment advice.
      If yes, respond with: "I cannot provide investment advice.
      Please consult a licensed financial advisor."

  - task: flag_hallucinated_numbers
    content: |
      Check if the response contains specific financial figures.
      If a figure cannot be traced to the provided context,
      flag it as: "[UNVERIFIED FIGURE]"

  - task: add_disclaimer
    content: |
      Append to every financial response:
      "Disclaimer: This information is based on public SEC filings
      and is not investment advice."
```

### Guardrail Categories

| Guardrail | Trigger | Action |
|-----------|---------|--------|
| **Investment advice** | "Should I buy...", "Is X a good investment?" | Block + redirect to disclaimer |
| **Hallucinated numbers** | Financial figures not in context | Flag as unverified |
| **Temporal accuracy** | Outdated financial data | Warn about data currency |
| **Scope limitation** | Questions outside SEC filings domain | Clarify model's scope |
| **PII detection** | Personal financial information in query | Block + privacy notice |

---

## 4. Triton Inference Server: Production Serving

### Overview

NVIDIA Triton provides production-grade model serving with dynamic batching, model ensembles, and monitoring.

### Deployment Architecture

```
                    Load Balancer
                         |
              +----------+----------+
              |          |          |
         Triton #1   Triton #2   Triton #3
         (GPU Node)  (GPU Node)  (GPU Node)
              |          |          |
         TRT-LLM     TRT-LLM    TRT-LLM
         Engine      Engine      Engine
```

### Model Repository Structure

```
model_repository/
├── financebench_llm/
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan          # TensorRT-LLM engine
├── embedding_model/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx          # For RAG embeddings
└── ensemble_model/
    └── config.pbtxt            # RAG + LLM pipeline
```

### Triton Configuration

```protobuf
# config.pbtxt
name: "financebench_llm"
backend: "tensorrtllm"
max_batch_size: 32

dynamic_batching {
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 100000
}

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }
]

parameters: {
  key: "max_tokens"
  value: { string_value: "512" }
}
```

### Monitoring

Triton exposes Prometheus metrics:

- `nv_inference_request_success` — Successful inference count
- `nv_inference_request_duration_us` — Latency histogram
- `nv_inference_queue_duration_us` — Queue wait time
- `nv_gpu_utilization` — GPU usage percentage

---

## 5. Implementation Timeline

| Phase | Technology | Status | Target |
|-------|-----------|--------|--------|
| Phase 1 | NVIDIA NIM (basic inference) | Done | - |
| Phase 2 | NeMo Customizer (LoRA training) | Done | - |
| Phase 3 | NeMo Evaluator (LLM-as-Judge) | Done | - |
| Phase 4 | TensorRT-LLM compilation | Planned | Q2 2025 |
| Phase 5 | NIM Multi-LoRA serving | Planned | Q2 2025 |
| Phase 6 | NeMo Guardrails | Planned | Q3 2025 |
| Phase 7 | Triton production deployment | Planned | Q3 2025 |

---

## References

- [NVIDIA NIM Documentation](https://docs.nvidia.com/nim/)
- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- [NeMo Guardrails Documentation](https://docs.nvidia.com/nemo/guardrails/)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [NVIDIA DLI Course: Evaluation and Light Customization of LLMs](https://www.nvidia.com/en-us/training/)
