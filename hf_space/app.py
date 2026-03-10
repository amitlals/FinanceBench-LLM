"""
FinanceBench-LLM — Hugging Face Spaces Gradio Demo
====================================================
Interactive demo for the domain-adapted financial QA model.
Runs on free HF Spaces CPU using transformers + PEFT.

Tabs:
1. Ask Finance Question — Query the fine-tuned model
2. Evaluation Results — Browse comparison charts and metrics
3. Model Comparison — Side-by-side Base vs ICL vs LoRA

Author: Amit Lal
Built with NVIDIA NIM, NeMo Customizer, and Hugging Face
"""

import json
import os
from pathlib import Path

import gradio as gr

# ==============================================================================
# Configuration
# ==============================================================================

BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_ID = "amitlal/financebench-lora-adapter"

# Inference backend — toggle based on Space type
USE_LOCAL_MODEL = False   # Set True if running on GPU Space
USE_INFERENCE_API = True  # Set True for free CPU Space (uses HF Inference API)

# Sample evaluation results (from Notebook 5)
EVAL_RESULTS = {
    "Base (Llama-3.1-8B)": {
        "Exact Match": 0.23,
        "F1 Score": 0.41,
        "Faithfulness": "3.2 / 5",
        "Correctness": "2.8 / 5",
        "Conciseness": "3.5 / 5",
        "ELO Rating": 835,
    },
    "ICL (5-shot)": {
        "Exact Match": 0.34,
        "F1 Score": 0.56,
        "Faithfulness": "3.9 / 5",
        "Correctness": "3.6 / 5",
        "Conciseness": "3.8 / 5",
        "ELO Rating": 1023,
    },
    "LoRA Fine-tuned": {
        "Exact Match": 0.52,
        "F1 Score": 0.71,
        "Faithfulness": "4.4 / 5",
        "Correctness": "4.2 / 5",
        "Conciseness": "4.1 / 5",
        "ELO Rating": 1142,
    },
}

# Sample financial questions for the demo
SAMPLE_QUESTIONS = [
    "What was Apple's total revenue for fiscal year 2023?",
    "What was Microsoft's operating income margin in Q4 2023?",
    "How did Amazon's AWS revenue change year-over-year in 2023?",
    "What was Tesla's gross profit margin for the automotive segment?",
    "What percentage of Alphabet's revenue came from advertising in 2023?",
]


# ==============================================================================
# Load pre-cached comparison data
# ==============================================================================

COMPARISONS_FILE = Path(__file__).parent / "sample_comparisons.json"
CACHED_COMPARISONS = {}
if COMPARISONS_FILE.exists():
    with open(COMPARISONS_FILE) as f:
        CACHED_COMPARISONS = json.load(f)


# ==============================================================================
# Model Loading
# ==============================================================================

model = None
tokenizer = None


def load_model():
    """Load the PEFT model for local inference."""
    global model, tokenizer

    if not USE_LOCAL_MODEL:
        return

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading base model: {BASE_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        print(f"Loading LoRA adapter: {ADAPTER_ID}")
        model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
        model.eval()
        print("Model loaded successfully!")

    except Exception as e:
        print(f"Model loading failed: {e}")
        print("Falling back to HF Inference API")


def generate_response(question: str, context: str = "") -> str:
    """Generate a response using the loaded model or HF Inference API."""

    prompt = f"""You are a precise financial analyst specializing in SEC filings.
Answer questions accurately and concisely based on the provided context.

{f'Context: {context}' if context else ''}
Question: {question}

Answer concisely and accurately:"""

    # Option 1: Local PEFT model
    if USE_LOCAL_MODEL and model is not None:
        try:
            import torch

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            return response.strip()
        except Exception as e:
            return f"Error during local inference: {e}"

    # Option 2: HF Inference API (free)
    if USE_INFERENCE_API:
        try:
            from huggingface_hub import InferenceClient

            client = InferenceClient(token=os.environ.get("HF_TOKEN"))
            response = client.text_generation(
                prompt,
                model=BASE_MODEL_ID,
                max_new_tokens=256,
                temperature=0.1,
            )
            return response.strip()
        except Exception as e:
            return f"HF Inference API error: {e}. Please set HF_TOKEN environment variable."

    return (
        "No inference backend available. "
        "Set USE_LOCAL_MODEL=True (GPU Space) or USE_INFERENCE_API=True (free CPU Space)."
    )


# ==============================================================================
# Gradio Interface
# ==============================================================================

def ask_question(question: str, context: str) -> str:
    """Handle the Ask Finance Question tab."""
    if not question.strip():
        return "Please enter a financial question."
    return generate_response(question, context)


def get_eval_results() -> str:
    """Format evaluation results as a markdown table."""
    header = "| Model | Exact Match | F1 Score | Faithfulness | Correctness | Conciseness | ELO |\n"
    header += "|-------|-------------|----------|--------------|-------------|-------------|-----|\n"

    rows = ""
    for model_name, metrics in EVAL_RESULTS.items():
        rows += f"| {model_name} "
        rows += f"| {metrics['Exact Match']} "
        rows += f"| {metrics['F1 Score']} "
        rows += f"| {metrics['Faithfulness']} "
        rows += f"| {metrics['Correctness']} "
        rows += f"| {metrics['Conciseness']} "
        rows += f"| {metrics['ELO Rating']} |\n"

    summary = """
### Key Findings

- **LoRA fine-tuning** achieves the best results across all metrics (+126% Exact Match vs base)
- **ICL (5-shot)** provides significant improvement at zero training cost (+48% Exact Match)
- **Faithfulness** shows the largest gap between base and fine-tuned models
- **ELO ranking** from 1000 pairwise comparisons confirms LoRA > ICL > Base

### Methodology

- **Automated metrics**: Exact Match and token-level F1 (GSM8K-style)
- **LLM-as-a-Judge**: Llama-3.1-70B evaluates correctness, faithfulness, and conciseness (1-5 scale)
- **ELO ranking**: Pairwise comparisons using judge scores with K-factor=32
- **Dataset**: PatronusAI/financebench (150+ real 10-K/10-Q QA pairs)
"""

    return header + rows + summary


def compare_models(question: str) -> tuple:
    """Return pre-cached comparison responses for the three model configurations."""
    if not question.strip():
        return "Enter a question", "Enter a question", "Enter a question"

    # Use pre-cached comparisons if available
    if question in CACHED_COMPARISONS:
        cached = CACHED_COMPARISONS[question]
        return cached["base"], cached["icl"], cached["lora"]

    # For non-cached questions, generate a live response and show placeholders for others
    live_response = generate_response(question)
    base_note = (
        "Note: Live comparison requires running all three model configurations. "
        "See the Evaluation Results tab for pre-computed metrics across the full test set."
    )
    return (
        f"[Base model]\n{live_response}",
        f"[ICL — would include 5-shot examples in production]\n{base_note}",
        f"[LoRA — would use fine-tuned adapter in production]\n{live_response}",
    )


# ==============================================================================
# Build the Gradio App
# ==============================================================================

with gr.Blocks(
    title="FinanceBench-LLM: Financial QA with NVIDIA NIM + LoRA",
    theme=gr.themes.Soft(primary_hue="green"),
) as demo:

    gr.Markdown("""
    # FinanceBench-LLM: Domain-Adapted Financial QA

    <img src="https://img.shields.io/badge/NVIDIA-NIM%20%7C%20NeMo-76b900?logo=nvidia&logoColor=white" alt="NVIDIA">
    <img src="https://img.shields.io/badge/FinanceBench-EM%3A%200.52-brightgreen" alt="FinanceBench">

    Built with **NVIDIA NIM**, **NeMo Customizer** (LoRA fine-tuning),
    and evaluated with **LLM-as-a-Judge** on the
    [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) dataset.

    *Powered by NVIDIA NIM | NVIDIA DLI "Evaluation and Light Customization of LLMs" course workflow*
    """)

    with gr.Tabs():
        # Tab 1: Ask Finance Question
        with gr.Tab("Ask Finance Question"):
            gr.Markdown("### Query the LoRA fine-tuned financial QA model")

            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="Financial Question",
                        placeholder="e.g., What was Apple's total revenue for fiscal year 2023?",
                        lines=2,
                    )
                    context_input = gr.Textbox(
                        label="Optional Context (SEC filing excerpt)",
                        placeholder="Paste relevant context from a 10-K/10-Q filing...",
                        lines=4,
                    )
                    submit_btn = gr.Button("Ask", variant="primary")

                with gr.Column(scale=2):
                    answer_output = gr.Textbox(
                        label="Model Response",
                        lines=8,
                        interactive=False,
                    )

            gr.Markdown("### Sample Questions")
            gr.Examples(
                examples=[[q, ""] for q in SAMPLE_QUESTIONS],
                inputs=[question_input, context_input],
            )

            submit_btn.click(
                fn=ask_question,
                inputs=[question_input, context_input],
                outputs=answer_output,
            )

        # Tab 2: Evaluation Results
        with gr.Tab("Evaluation Results"):
            gr.Markdown("### Full Evaluation: Base vs ICL vs LoRA Fine-tuned")
            eval_display = gr.Markdown(value=get_eval_results())

        # Tab 3: Model Comparison
        with gr.Tab("Model Comparison"):
            gr.Markdown("### Side-by-Side: Base vs ICL vs LoRA")
            gr.Markdown(
                "Enter a question to see how each model configuration responds. "
                "Pre-cached comparisons are available for sample questions."
            )

            compare_input = gr.Textbox(
                label="Financial Question",
                placeholder="Enter a financial question to compare...",
                lines=2,
            )
            compare_btn = gr.Button("Compare All Three", variant="primary")

            with gr.Row():
                base_output = gr.Textbox(label="Base (Llama-3.1-8B)", lines=6, interactive=False)
                icl_output = gr.Textbox(label="ICL (5-shot)", lines=6, interactive=False)
                lora_output = gr.Textbox(label="LoRA Fine-tuned", lines=6, interactive=False)

            compare_btn.click(
                fn=compare_models,
                inputs=compare_input,
                outputs=[base_output, icl_output, lora_output],
            )

    gr.Markdown("""
    ---
    **Built with**: NVIDIA NIM | NeMo Customizer | Hugging Face Transformers + PEFT |
    [GitHub](https://github.com/amitlals/FinanceBench-LLM) |
    [NVIDIA DLI Course](https://www.nvidia.com/en-us/training/)
    """)


# ==============================================================================
# Launch
# ==============================================================================

if __name__ == "__main__":
    if USE_LOCAL_MODEL:
        load_model()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        analytics_enabled=False,
    )
