# Research Plan: The Residual Stream After Bijective Token Transformation

## Motivation & Novelty Assessment

### Why This Research Matters
Pretrained LLMs encode rich semantic knowledge in their embedding layers and residual streams. Understanding whether models can adapt their internal representations to systematic token remappings (bijective ciphers) reveals how deeply the model's "understanding" is tied to specific token identities versus distributional structure. This has implications for adversarial robustness, cipher-based jailbreaks, and fundamental understanding of how transformers process language.

### Gap in Existing Work
The ICL Ciphers paper (Fang et al., 2025) shows that models *can* perform tasks under bijective ciphers better than non-bijective ones, and basic logit lens probing shows the cipher is learned in deeper layers. However, no work has systematically measured:
1. How the residual stream **structure** (not just top-token predictions) changes under bijective transformation across sequence positions
2. Whether there is a convergence point where ciphered representations become linearly similar to clean ones (modulo the permutation)
3. How this varies across layers, context lengths, and model sizes

### Our Novel Contribution
We directly measure residual stream activations under bijective token permutation, testing whether the model's hidden states converge to a "permuted normal" state. Specifically, we test if there exists a linear transformation (related to the cipher's permutation matrix in embedding space) that maps ciphered hidden states back to clean ones, and how this mapping quality evolves across layers and token positions.

### Experiment Justification
- **Experiment 1 (Residual Stream Similarity)**: Measure cosine similarity between clean and ciphered hidden states at each layer and position — directly tests whether representations converge.
- **Experiment 2 (Logit Lens Analysis)**: Track which token the model "thinks about" at each layer — tests whether the model decodes the cipher progressively.
- **Experiment 3 (Permutation-Adjusted Similarity)**: Apply the cipher's permutation in embedding space to see if ciphered states are "normal modulo the shuffle" — the core question.
- **Experiment 4 (Context Length Effect)**: Vary how much ciphered text the model has seen to test the "recovery" hypothesis.

## Research Question
Given a bijective token permutation f: V→V applied to input text, does a pretrained transformer's residual stream eventually represent the permuted text in a way that is equivalent to normal processing modulo the permutation? Or does the model fail to recover because embeddings don't get properly remapped?

## Hypothesis Decomposition
- **H1**: Residual stream cosine similarity between clean and ciphered text will be low at early layers but increase at later layers (partial recovery).
- **H2**: When we account for the permutation (by applying the inverse permutation in embedding space), the similarity will be substantially higher — the model's state is "normal modulo the shuffle."
- **H3**: More context (longer sequences) will improve recovery, as the model has more signal to attune to the cipher.
- **H4**: Logit lens will show the model progressively shifting from predicting "wrong" (original) tokens to predicting cipher-consistent tokens at deeper layers.

## Proposed Methodology

### Approach
Use TransformerLens to extract residual stream activations from GPT-2 Small and Pythia-410M. Compare activations between:
- **Clean text**: Original text
- **Ciphered text**: Same text with bijective token permutation applied
- **Permutation-adjusted**: Ciphered hidden states transformed by the inverse permutation matrix in embedding space

### Models
- GPT-2 Small (12 layers, 768 dims, well-understood)
- Pythia-410M (24 layers, 1024 dims, has tuned lens/SAE support)

### Experimental Steps
1. Implement bijective token cipher with Zipfian frequency matching
2. Prepare text samples from The Pile / WikiText
3. Extract residual stream activations for clean and ciphered text
4. Compute layer-wise cosine similarity (clean vs ciphered)
5. Compute permutation-adjusted similarity (key metric)
6. Apply logit lens to track prediction evolution
7. Vary context length and measure recovery
8. Statistical analysis across multiple texts

### Baselines
- Clean text (upper bound on "normal" residual stream)
- Random token replacement (non-bijective, lower bound)
- Identity permutation (sanity check)

### Evaluation Metrics
- Cosine similarity between clean and ciphered hidden states per layer
- Permutation-adjusted cosine similarity (applying P^{-1} in embedding space)
- Logit lens top-1 accuracy (does model predict correct next token under cipher?)
- Rank of correct token at each layer via logit lens
- Centered Kernel Alignment (CKA) between clean and ciphered representations

### Statistical Analysis Plan
- Bootstrap confidence intervals (n=100+ text samples)
- Paired t-tests for layer-wise comparisons
- Effect sizes (Cohen's d) for similarity differences

## Expected Outcomes
- If H2 is supported: the residual stream IS normal modulo the shuffle — the model learns to remap internally
- If H2 is refuted: the embedding layer creates a persistent distortion that the model cannot overcome, even with context

## Timeline
- Phase 1 (Planning): 15 min ✓
- Phase 2 (Setup): 15 min
- Phase 3 (Implementation): 60 min
- Phase 4 (Experiments): 60 min
- Phase 5 (Analysis): 30 min
- Phase 6 (Documentation): 20 min

## Potential Challenges
- GPU memory for large models — mitigate with smaller batch sizes
- Permutation matrix in embedding space may not be well-defined if embeddings aren't orthogonal — use pseudo-inverse
- Context window limits — use sequences of ~512 tokens

## Success Criteria
- Clear evidence for or against whether the residual stream converges to "normal modulo permutation"
- Statistical significance (p < 0.05) on key comparisons
- Reproducible results across multiple text samples
