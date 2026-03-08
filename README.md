# Truth Vector

A mechanistic interpretability project that identifies and manipulates internal "truth representations" in large language models to understand and influence truthful behavior.

## Overview

This project investigates how transformer-based LLMs internally represent truthfulness. Using the [TruthfulQA](https://github.com/sylinrl/TruthfulQA) benchmark dataset and [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), we:

1. **Extract activations** across all transformer layers while the model answers true/false questions.
2. **Compute a truth direction** — the difference between mean activations on correct vs. incorrect answers.
3. **Intervene via hooks** to add, subtract, or randomize this direction and measure the effect on accuracy and refusal rates.

The model studied is **[Qwen 2.5 1.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)** (`Qwen/Qwen2.5-1.5B-Instruct`), a 28-layer instruction-tuned LLM.

## Research Questions

- Does a consistent "truth vector" exist in a model's residual stream?
- Which layers are most causally responsible for truthful behavior?
- Can we steer a model toward greater truthfulness by injecting this vector?
- Does suppressing the truth direction increase hallucination rates?

## Repository Structure

```
truth-vector/
├── truth_vector.ipynb   # Main analysis pipeline (activation extraction + hook experiments)
├── plots.ipynb          # Visualization of experiment results
├── TruthfulQA(1).csv    # TruthfulQA benchmark dataset (790 questions)
└── README.md
```

## Methodology

### 1. Dataset
The TruthfulQA dataset provides questions with both a **best correct answer** and a **best incorrect answer**. Each question is formatted into an instruction-tuned chat prompt and fed to the model.

### 2. Activation Extraction
Using TransformerLens hooks, residual stream activations (`resid_post`) are captured at every layer for each question. Activations are split into two groups:
- **True predictions** — where the model correctly identifies the truthful answer
- **False predictions** — where the model selects the incorrect answer

### 3. Truth Vector Computation
The truth vector is computed as:

```
truth_vector[layer] = mean(true_activations) − mean(false_activations)
```

This direction in activation space represents what separates truthful from untruthful model behavior.

### 4. Hook Experiments
Three intervention types are tested at each layer:

| Hook | Description |
|------|-------------|
| `lie_hook` | Adds a scaled truth vector (positive or negative) to residual stream activations |
| `truth_ablate` | Projects out the truth direction, removing it from the representation |
| `random_hook` | Adds random Gaussian noise as a control baseline |

Results are logged to `hook_results.csv` with fields including layer, hook type, strength, accuracy, and refusal score.

### 5. Visualization
`plots.ipynb` reads `hook_results.csv` and produces:
- Accuracy curves across layers for each intervention strength
- Refusal score distributions
- Comparison of hooked vs. baseline model performance

## Getting Started

### Option 1: Google Colab (Recommended)

Open `truth_vector.ipynb` directly in [Google Colab](https://colab.research.google.com/). The notebook automatically detects the Colab environment and installs all required dependencies.

### Option 2: Local Setup

**Requirements:** Python 3.8+, a CUDA-capable GPU is recommended (CPU inference is supported but slow).

**Install dependencies:**
```bash
pip install torch transformer_lens circuitsvis
pip install pandas plotly matplotlib einops fancy_einsum jaxtyping tqdm
```

**Run the notebooks:**
```bash
jupyter notebook
```

Open `truth_vector.ipynb` and run cells sequentially. The first run will download the Qwen 2.5 1.5B model (~3 GB).

After the main notebook completes and generates `hook_results.csv`, open `plots.ipynb` to visualize the results.

## Key Dependencies

| Package | Purpose |
|---------|---------|
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | Hook-based activation extraction and intervention |
| [PyTorch](https://pytorch.org/) | Neural network backend |
| [circuitsvis](https://github.com/alan-cooney/CircuitsVis) | Mechanistic interpretability visualizations |
| [Plotly](https://plotly.com/python/) | Interactive result plots |
| [pandas](https://pandas.pydata.org/) | Dataset handling and CSV I/O |
| [einops](https://github.com/arogozhnikov/einops) | Readable tensor manipulation |

## Output

Running `truth_vector.ipynb` produces a `hook_results.csv` file with the following schema:

| Column | Description |
|--------|-------------|
| `model_name` | Model identifier |
| `run_id` | Unique run identifier |
| `seed` | Random seed |
| `hook_type` | Type of intervention (`lie`, `ablate`, `random`, `none`) |
| `layer` | Transformer layer index (0–27) |
| `strength` | Scaling factor applied to the truth vector |
| `metric` | Metric name (`accuracy`, `refusal_score`) |
| `value` | Metric value |
| `batch_size` | Number of samples evaluated |
| `dataset_slice` | Slice of the dataset used |
| `notes` | Any additional notes |

## Background

This project is inspired by work in **mechanistic interpretability** and **representation engineering**, particularly:

- [Representation Engineering](https://arxiv.org/abs/2310.01405) — Zou et al., 2023
- [TruthfulQA](https://arxiv.org/abs/2109.07958) — Lin et al., 2021
- [Activation Addition / Steering Vectors](https://arxiv.org/abs/2308.10248) — Turner et al., 2023

## License

This project is open source. See the repository for details.
