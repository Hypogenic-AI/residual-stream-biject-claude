# The Residual Stream After Bijective Token Transformation

## Overview

This project investigates whether a pretrained transformer's residual stream can recover from a bijective token permutation (a shuffle of which token IDs represent which surface forms). We measure hidden state similarity between clean and ciphered inputs across layers, using cosine similarity, CKA, logit lens analysis, and permutation-adjusted comparisons.

## Key Findings

- **The model does NOT recover "modulo the shuffle."** Applying the cipher's inverse in embedding space makes similarity *worse*, not better — the cipher's effect is highly non-linear.
- **Apparent convergence is an architectural artifact.** In GPT-2, different clean texts already have >96% cosine similarity due to LayerNorm — high clean-vs-cipher similarity is meaningless.
- **Pythia-410M shows clearer separation:** Cipher representations stay ~30% above random but ~30% below clean, suggesting partial structural extraction without genuine decoding.
- **Logit lens confirms no cipher decoding.** The model never predicts the cipher-consistent next token (rank ~6,000–12,000 out of 50K).
- **The embedding lookup creates an irreversible non-linear distortion** that 12–24 layers of processing cannot undo without explicit in-context demonstrations.

## File Structure

```
├── REPORT.md              # Full research report with results and analysis
├── planning.md            # Research plan and methodology
├── literature_review.md   # Pre-gathered literature review
├── resources.md           # Resource catalog
├── src/
│   ├── bijective_cipher.py    # Bijective token permutation implementation
│   ├── run_experiments.py     # Main GPT-2 experiment
│   ├── run_pythia_experiment.py # Pythia-410M experiment
│   └── deeper_analysis.py    # Architecture convergence & probe analysis
├── results/
│   ├── experiment_results.json    # GPT-2 results
│   ├── pythia_results.json        # Pythia results
│   ├── deeper_analysis.json       # Architecture analysis results
│   └── plots/                     # All generated visualizations
├── papers/                # Downloaded research papers
├── datasets/              # Pre-downloaded datasets
└── code/                  # Cloned reference repositories
```

## Reproduction

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install torch transformer-lens numpy matplotlib scipy scikit-learn tqdm datasets

# Run experiments (requires NVIDIA GPU)
CUDA_VISIBLE_DEVICES=0 python src/run_experiments.py       # GPT-2 (~2 min)
CUDA_VISIBLE_DEVICES=1 python src/run_pythia_experiment.py  # Pythia (~1 min)
CUDA_VISIBLE_DEVICES=0 python src/deeper_analysis.py        # Analysis (~30 sec)
```

See [REPORT.md](REPORT.md) for full results and analysis.
