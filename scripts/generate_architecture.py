"""
Generate the architecture diagram as a PNG for the README.
Run: python scripts/generate_architecture.py
Output: assets/architecture.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ── Colors ────────────────────────────────────────────────────────────────
NVIDIA_GREEN = "#76b900"
NVIDIA_GREEN_LIGHT = "#e8f5e9"
DATA_BLUE = "#667eea"
DATA_BLUE_LIGHT = "#e8eaf6"
EVAL_ORANGE = "#f6ad55"
EVAL_ORANGE_LIGHT = "#fff3e0"
DEPLOY_RED = "#ff6f61"
DEPLOY_RED_LIGHT = "#fce4ec"
WHITE = "#ffffff"
DARK = "#1a1a2e"
GRAY_BG = "#f8f9fa"
BORDER = "#dee2e6"

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(WHITE)

def draw_box(ax, x, y, w, h, text, color, text_color="white", fontsize=9, bold=False):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor="#333333", linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        color=text_color, fontweight=weight, zorder=4,
        linespacing=1.4,
    )

def draw_group(ax, x, y, w, h, label, bg_color, border_color):
    group = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.2",
        facecolor=bg_color, edgecolor=border_color,
        linewidth=1.5, linestyle="-", zorder=1,
    )
    ax.add_patch(group)
    ax.text(
        x + 0.3, y + h - 0.3, label,
        ha="left", va="top", fontsize=10,
        color=border_color, fontweight="bold", zorder=2,
    )

def draw_arrow(ax, x1, y1, x2, y2, color="#555555"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color,
            lw=1.5, connectionstyle="arc3,rad=0.0",
        ),
        zorder=2,
    )

# ══════════════════════════════════════════════════════════════════════════
# Group backgrounds
# ══════════════════════════════════════════════════════════════════════════

# Data Layer
draw_group(ax, 0.3, 7.5, 4.4, 2.2, "DATA LAYER", DATA_BLUE, DATA_BLUE)

# NVIDIA NIM Inference
draw_group(ax, 5.2, 7.5, 5.3, 2.2, "NVIDIA NIM INFERENCE", NVIDIA_GREEN_LIGHT, NVIDIA_GREEN)

# NeMo Customization
draw_group(ax, 11.0, 7.5, 4.7, 2.2, "NeMo CUSTOMIZATION", NVIDIA_GREEN_LIGHT, NVIDIA_GREEN)

# Evaluation Pipeline
draw_group(ax, 0.3, 3.5, 10.2, 3.5, "EVALUATION PIPELINE", EVAL_ORANGE_LIGHT, EVAL_ORANGE)

# Export & Demo
draw_group(ax, 11.0, 3.5, 4.7, 3.5, "EXPORT & DEMO", DEPLOY_RED_LIGHT, DEPLOY_RED)

# ══════════════════════════════════════════════════════════════════════════
# Boxes
# ══════════════════════════════════════════════════════════════════════════

# Data Layer
draw_box(ax, 0.5, 8.0, 1.9, 1.2,
         "PatronusAI/\nfinancebench\n150+ QA pairs", DATA_BLUE, WHITE, 8, True)
draw_box(ax, 2.8, 8.2, 1.7, 0.8,
         "80/20\nTrain/Test Split", "#8c9eff", WHITE, 8)

# NIM Inference
draw_box(ax, 5.5, 8.2, 2.2, 0.8,
         "Llama-3.1-8B\nvia NIM", NVIDIA_GREEN, WHITE, 9, True)
draw_box(ax, 8.1, 8.2, 2.2, 0.8,
         "In-Context Learning\n5-Shot Examples", "#4caf50", WHITE, 8)

# NeMo Customization
draw_box(ax, 11.3, 8.5, 2.0, 0.9,
         "NeMo Customizer\nLoRA r=16, α=32", NVIDIA_GREEN, WHITE, 8, True)
draw_box(ax, 13.6, 8.5, 1.8, 0.9,
         "LoRA Adapter\nWeights", "#558b2f", WHITE, 8)

# Evaluation Pipeline
draw_box(ax, 0.8, 5.2, 2.2, 0.9,
         "NeMo Evaluator", EVAL_ORANGE, DARK, 9, True)
draw_box(ax, 3.5, 5.8, 2.8, 0.9,
         "LLM-as-a-Judge\nCorrectness · Faithfulness\n· Conciseness", "#ff9800", WHITE, 7.5)
draw_box(ax, 3.5, 4.4, 2.8, 0.9,
         "Exact Match · F1\nGSM8K-style Metrics", "#ffa726", DARK, 8)
draw_box(ax, 7.0, 4.8, 3.0, 1.2,
         "MLflow Tracking\nLoss Curves · Win Rates\nELO Rankings", "#ef6c00", WHITE, 8, True)

# Export & Demo
draw_box(ax, 11.4, 5.5, 2.0, 1.0,
         "Export to\nHF PEFT Format", "#e57373", WHITE, 8, True)
draw_box(ax, 13.7, 5.5, 1.8, 1.0,
         "HuggingFace\nSpaces Demo\n(Free Gradio)", DEPLOY_RED, WHITE, 7.5, True)
draw_box(ax, 11.4, 4.0, 4.1, 1.0,
         "NIM + Multi-LoRA Serving\n(Production Deployment)", "#b71c1c", WHITE, 8, True)

# ══════════════════════════════════════════════════════════════════════════
# Arrows
# ══════════════════════════════════════════════════════════════════════════

# Data → Split
draw_arrow(ax, 2.4, 8.6, 2.8, 8.6, DATA_BLUE)

# Split → NIM models
draw_arrow(ax, 4.5, 8.6, 5.5, 8.6, "#333")
draw_arrow(ax, 4.5, 8.6, 8.1, 8.6, "#333")

# Split → NeMo Customizer
draw_arrow(ax, 4.5, 8.9, 11.3, 8.9, NVIDIA_GREEN)

# Customizer → LoRA weights
draw_arrow(ax, 13.3, 8.95, 13.6, 8.95, NVIDIA_GREEN)

# NIM models → Evaluator
draw_arrow(ax, 6.6, 8.2, 1.9, 6.1, EVAL_ORANGE)
draw_arrow(ax, 9.2, 8.2, 2.5, 6.1, EVAL_ORANGE)

# Evaluator → Judge & Metrics
draw_arrow(ax, 3.0, 5.65, 3.5, 6.0, EVAL_ORANGE)
draw_arrow(ax, 3.0, 5.5, 3.5, 4.9, EVAL_ORANGE)

# Judge/Metrics → MLflow
draw_arrow(ax, 6.3, 6.0, 7.0, 5.6, "#ef6c00")
draw_arrow(ax, 6.3, 4.9, 7.0, 5.2, "#ef6c00")

# LoRA → Export
draw_arrow(ax, 14.5, 8.5, 12.4, 6.5, DEPLOY_RED)

# LoRA → Multi-LoRA
draw_arrow(ax, 14.5, 8.5, 13.5, 5.0, "#b71c1c")

# Export → HF Spaces
draw_arrow(ax, 13.4, 6.0, 13.7, 6.0, DEPLOY_RED)

# ══════════════════════════════════════════════════════════════════════════
# Title & branding
# ══════════════════════════════════════════════════════════════════════════

ax.text(
    8.0, 0.8,
    "FinanceBench-LLM  ·  NVIDIA NIM + NeMo LoRA + LLM-as-a-Judge  ·  End-to-End Pipeline",
    ha="center", va="center", fontsize=11,
    color="#555", style="italic",
)

# Legend
legend_items = [
    mpatches.Patch(color=DATA_BLUE, label="Data"),
    mpatches.Patch(color=NVIDIA_GREEN, label="NVIDIA"),
    mpatches.Patch(color=EVAL_ORANGE, label="Evaluation"),
    mpatches.Patch(color=DEPLOY_RED, label="Deployment"),
]
ax.legend(
    handles=legend_items, loc="lower left",
    fontsize=9, framealpha=0.9,
    edgecolor=BORDER, fancybox=True,
    ncol=4, bbox_to_anchor=(0.15, -0.02),
)

# ══════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════

output_dir = Path(__file__).parent.parent / "assets"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "architecture.png"

plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=WHITE, pad_inches=0.3)
plt.close()
print(f"Saved: {output_path}")
print(f"Size: {output_path.stat().st_size / 1024:.0f} KB")
