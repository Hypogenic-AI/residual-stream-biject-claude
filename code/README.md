# Cloned Repositories

## Repo 1: MLSAE (Multi-Layer Sparse Autoencoders)
- **URL**: https://github.com/tim-lawson/mlsae
- **Purpose**: Train and analyze sparse autoencoders across all transformer layers for residual stream analysis
- **Location**: code/mlsae/
- **Key files**: Training scripts, analysis notebooks, pre-trained model configs
- **Notes**: Pre-trained models available on HuggingFace. Primary tool for cross-layer residual stream feature analysis.

## Repo 2: Tuned Lens
- **URL**: https://github.com/AlignmentResearch/tuned-lens
- **Purpose**: Train and apply affine probes to decode intermediate hidden states into vocabulary predictions
- **Location**: code/tuned-lens/
- **Key files**: `tuned_lens/` package, training scripts, evaluation tools
- **Notes**: Install via `pip install tuned-lens`. Pre-trained lenses available for many models. Essential for tracking prediction evolution through layers.

## Repo 3: ACDC (Automatic Circuit Discovery)
- **URL**: https://github.com/ArthurConmy/Automatic-Circuit-Discovery
- **Purpose**: Automated identification of circuits responsible for specific model behaviors
- **Location**: code/acdc/
- **Key files**: ACDC algorithm implementation, benchmark circuits
- **Notes**: Requires TransformerLens. Can identify which model components are involved in cipher processing.

## Repo 4: ACDC++ / Edge Attribution Patching
- **URL**: https://github.com/Aaquib111/acdcpp
- **Purpose**: Faster alternative to ACDC using gradient-based attribution patching
- **Location**: code/acdcpp/
- **Key files**: EAP implementation, benchmarks
- **Notes**: Much faster than ACDC — requires only 2 forward passes + 1 backward pass.

## Repo 5: TransformerLens
- **URL**: https://github.com/neelnanda-io/TransformerLens
- **Purpose**: Library for mechanistic interpretability with hook-based activation access
- **Location**: code/TransformerLens/
- **Key files**: `transformer_lens/` package
- **Notes**: Install via `pip install transformer-lens`. Provides `HookedTransformer` with access to all intermediate activations. Supports GPT-2, Pythia, Llama, and many other models. **Critical dependency** for our experiments.

## Usage for Experiment Runner

The experiment runner should:
1. Install TransformerLens for activation access: `pip install transformer-lens`
2. Install tuned-lens for prediction probing: `pip install tuned-lens`
3. Use MLSAE pre-trained models from HuggingFace for feature analysis
4. Implement bijective cipher following ICL Ciphers methodology (Zipfian frequency matching)
5. Use ACDC/EAP for circuit identification if needed
