# The Residual Stream After Bijective Token Transformation

## 1. Executive Summary

We investigate whether a pretrained transformer's residual stream can adapt to a bijective token permutation — a shuffling of which token IDs represent which surface forms. Our experiments on GPT-2 Small and Pythia-410M reveal that **the model does not recover "modulo the shuffle."** While cosine similarity between clean and ciphered hidden states increases at later layers, this is largely an architectural artifact: different clean texts also show >96% cosine similarity with each other in GPT-2 due to LayerNorm convergence. The logit lens confirms the model never learns to predict cipher-consistent next tokens. Pythia-410M, with weaker architectural convergence, shows the cipher reduces similarity below clean but stays above random, suggesting partial structural extraction without genuine cipher decoding.

**Bottom line:** The embedding layer creates a persistent distortion that the transformer cannot undo through forward-pass processing alone. The model's residual stream does *not* converge to "normal modulo the shuffle."

## 2. Goal

**Hypothesis:** If a bijective function f: V → V is applied to shuffle token-surface form mappings and the transformed text is fed into a pretrained language model, the model's residual stream will eventually resemble its normal state modulo the shuffle, because the transformer can internally learn to remap the cipher.

**Why this matters:** Understanding how pretrained LLMs handle systematic input perturbations reveals (1) how deeply semantic understanding is tied to specific token identities vs. distributional structure, (2) the robustness of internal representations to adversarial token remapping, and (3) limits of in-context adaptation in frozen models.

## 3. Data Construction

### Dataset
We used 20 diverse English text samples (2-4 sentences each) covering topics including science, economics, literature, technology, and daily life. Texts were selected to cover a range of vocabulary and syntactic structures.

### Bijective Cipher Construction
- **Method:** Random bijective permutation f: V → V with Zipfian frequency matching (tokens shuffled within log-frequency bins to preserve distributional statistics)
- **Shuffle rates tested:** 0.3, 0.5, 0.7, 1.0 (fraction of vocabulary permuted)
- **Seed:** 42 for reproducibility
- **Verification:** Bijectivity confirmed via assertion (forward ∘ inverse = identity)

### Baselines
1. **Clean text:** Original, unmodified tokens (upper bound)
2. **Bijective cipher:** Tokens permuted by f at various rates
3. **Random tokens:** Non-bijective random replacement (lower bound)

## 4. Experiment Description

### Methodology

#### High-Level Approach
We extract residual stream activations at every layer of a pretrained transformer for three conditions (clean, ciphered, random) and compare them using:
1. **Cosine similarity** — direct comparison of hidden state directions
2. **Permutation-adjusted similarity** — transform ciphered states by the cipher's inverse in embedding space before comparing
3. **Logit lens** — decode intermediate hidden states to vocabulary predictions
4. **Inter-text similarity** — control for architectural convergence artifacts
5. **CKA** — representation similarity robust to rotation/scaling (Pythia only)

#### Why This Method?
The permutation-adjusted similarity is the key test: if the residual stream IS "normal modulo the shuffle," then undoing the shuffle in embedding space should dramatically increase similarity. If it doesn't, the model has not internally remapped the cipher.

### Implementation Details

#### Tools and Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.10.0+cu128 | Tensor operations, GPU |
| TransformerLens | 2.15.4 | Hooked transformer with activation cache |
| NumPy | (latest) | Numerical computation |
| SciPy | (latest) | Statistical tests |
| scikit-learn | (latest) | Linear probes |
| Matplotlib | (latest) | Visualization |

#### Models
| Model | Layers | d_model | Vocab | Purpose |
|-------|--------|---------|-------|---------|
| GPT-2 Small | 12 | 768 | 50257 | Primary analysis, well-understood |
| Pythia-410M | 24 | 1024 | 50304 | Cross-model validation, less LN convergence |

#### Hardware
- NVIDIA RTX A6000 (49 GB VRAM)
- CUDA 12.8

### Experimental Protocol
- **Random seed:** 42
- **Number of text samples:** 30 (GPT-2), 20 (Pythia)
- **Execution time:** 2 min (GPT-2), 1 min (Pythia), <1 min (deeper analysis)

### Permutation-Adjustment Matrix
To test the "normal modulo shuffle" hypothesis, we compute a transformation matrix T such that for ciphered hidden states h_cipher, T·h_cipher ≈ h_clean. T is obtained as the least-squares solution to E_perm · T ≈ E, where E is the embedding matrix and E_perm is the permuted embedding matrix. This only works if the cipher's effect propagates linearly through the network — a strong assumption we test empirically.

## 5. Results

### Experiment 1: Cosine Similarity Across Layers (GPT-2 Small)

| Condition | Shuffle Rate | Embedding Layer | Middle Layer (6) | Final Layer (12) |
|-----------|-------------|----------------|-----------------|-----------------|
| Raw (clean vs cipher) | 0.3 | 0.927 | 0.849 | 0.942 |
| Raw (clean vs cipher) | 0.5 | 0.873 | 0.756 | 0.912 |
| Raw (clean vs cipher) | 0.7 | 0.822 | 0.687 | 0.896 |
| Raw (clean vs cipher) | 1.0 | 0.743 | 0.579 | 0.873 |
| Adjusted | 1.0 | 0.256 | 0.068 | 0.417 |
| Random baseline | — | 0.637 | 0.526 | 0.858 |

**Key finding:** Raw cosine similarity INCREASES from middle to final layer, but the **random baseline also reaches 0.858** at the final layer. The permutation adjustment makes things WORSE, not better.

### Experiment 2: Architecture Convergence Control (GPT-2)

Cosine similarity between position-averaged hidden states of **different clean texts**:

| Layer | Inter-text Similarity |
|-------|----------------------|
| 0 | 0.963 |
| 3 | 0.996 |
| 6 | 0.994 |
| 9 | 0.986 |
| 12 | 0.963 |

**Key finding:** Different clean texts already have >96% cosine similarity at every layer. The high clean-vs-cipher similarity in GPT-2 is an **architectural artifact** of LayerNorm forcing all representations toward similar directions.

### Experiment 3: Pythia-410M Cross-Validation

| Condition | Shuffle Rate | Embedding | Final Layer | CKA (Final) |
|-----------|-------------|-----------|-------------|-------------|
| Raw | 0.3 | 0.766 | 0.688 | 0.603 |
| Raw | 0.5 | 0.581 | 0.565 | 0.509 |
| Raw | 1.0 | 0.146 | 0.391 | 0.300 |
| Random | — | — | 0.295 | — |

**Key finding:** Pythia shows clearer separation. The cipher result (0.391) is above random (0.295), suggesting the model extracts some distributional structure from ciphered text. But similarity DECREASES from embedding to final layer for low shuffle rates, contrary to the "recovery" hypothesis.

### Experiment 4: Logit Lens Analysis (GPT-2)

Mean rank of the "correct" next token at the final layer:

| Condition | Token Being Ranked | Mean Rank |
|-----------|--------------------|-----------|
| Clean | Correct next token | 166 |
| Cipher (r=0.3) | Cipher-correct token | 5,909 |
| Cipher (r=0.5) | Cipher-correct token | 7,787 |
| Cipher (r=1.0) | Cipher-correct token | 11,542 |
| Cipher (r=1.0) | Original token | 11,607 |

**Key finding:** The model does NOT decode the cipher. The "correct" next token under the cipher mapping has rank ~6,000–12,000 (out of 50,257), essentially random. The model also doesn't predict the original (un-ciphered) next token — both are equally lost.

### Experiment 5: Attention Pattern Analysis

Attention entropy (mean across heads and positions):

| Layer | Clean Entropy | Cipher Entropy |
|-------|--------------|----------------|
| 0 | 1.038 | 1.054 |
| 3 | 0.877 | 0.847 |
| 6 | 0.704 | 0.669 |
| 9 | 0.478 | 0.529 |
| 11 | 0.710 | 0.811 |

**Key finding:** Attention patterns are slightly more diffuse at later layers under cipher (0.811 vs 0.710 at layer 11), suggesting the model is less certain about which tokens to attend to, but the difference is small — the attention mechanism is not dramatically disrupted.

### Statistical Tests

For the comparison of raw vs. random similarity at later layers (6–12):

| Shuffle Rate | Raw Mean | Random Mean | t-statistic | p-value |
|-------------|----------|-------------|-------------|---------|
| 0.3 | 0.926 | 0.800 | 7.41 | 0.0003 |
| 0.5 | 0.898 | 0.800 | 7.02 | 0.0004 |
| 0.7 | 0.876 | 0.799 | 6.92 | 0.0004 |
| 1.0 | 0.847 | 0.800 | 7.48 | 0.0003 |

The cipher consistently produces higher similarity than random (p < 0.001), but the effect size is small relative to the architectural convergence baseline.

## 6. Result Analysis

### Key Findings

1. **The model does NOT recover "modulo the shuffle."** The permutation-adjustment matrix makes similarity *worse*, not better. The cipher's effect is not a simple linear transformation in hidden space.

2. **Apparent convergence is an architectural artifact.** In GPT-2, LayerNorm drives all inputs toward similar representations (inter-text cosine sim > 0.96). The high clean-vs-cipher similarity is not cipher decoding — it's a property of the architecture.

3. **Cipher stays above random, but below clean.** In Pythia-410M (less convergence), the cipher produces representations 30% above random but 30% below clean at the final layer (full shuffle). The model extracts some structure from ciphered text (distributional patterns survive the bijection) but doesn't decode the cipher.

4. **The logit lens confirms no cipher decoding.** At no layer does the model predict the cipher-consistent next token — the rank is essentially random for the ciphered condition.

5. **Attention patterns are only mildly affected.** The attention mechanism operates similarly on clean and ciphered text, with slightly higher entropy under cipher at later layers.

### Why the Model Doesn't Recover

The fundamental issue is that the embedding layer is a **lookup table**, not a learned transformation. When token ID 42 maps to embedding E[42] in clean text, and after the cipher it maps to E[f(42)] instead, the model receives completely wrong semantic embeddings. Unlike a learned linear transformation (which could be inverted by later layers), a random permutation of 50,000 embeddings creates a highly non-linear distortion that the 12–24 layers of processing cannot undo.

The model's later layers do produce *similar-looking* hidden states for any input (clean, cipher, or random), but this is because:
- LayerNorm constrains the norm and centering of representations
- Attention patterns are dominated by positional structure (nearby tokens)
- The residual stream accumulates a "default" pattern through the residual connections

This is superficial similarity, not genuine understanding.

### Surprises

The most surprising finding was that **permutation adjustment made things worse**. We expected that if the residual stream was "normal modulo the shuffle," the adjustment would significantly increase similarity. Instead, the least-squares mapping T introduced additional noise, confirming that the cipher's effect propagates non-linearly through attention and MLP layers.

Also surprising: the random baseline in GPT-2 reached 0.858 cosine similarity at the final layer, almost as high as the cipher (0.873). This was fully explained by the inter-text convergence analysis.

### Limitations

1. **No in-context demonstrations.** We tested zero-shot cipher processing. The ICL Ciphers paper (Fang et al., 2025) shows models CAN learn ciphers from demonstrations — our results apply to the no-demonstration setting.
2. **Short sequences.** Our texts are ~30 tokens. Longer sequences might provide more signal, though our context-length analysis showed no recovery trend within 512 tokens.
3. **Only autoregressive models.** Bidirectional models might behave differently.
4. **Position-averaged analysis.** Per-position analysis showed no clear recovery trend, but a more fine-grained analysis might reveal local patterns.
5. **Two models.** Results may vary with model size (the ICL Ciphers paper found larger models recover more).

## 7. Conclusions

### Summary
A pretrained transformer's residual stream does **not** converge to "normal modulo the shuffle" when fed bijectively-permuted tokens without demonstrations. The model's representations at later layers appear superficially similar across all inputs (clean, cipher, and random) due to architectural convergence (LayerNorm), but this is not genuine cipher decoding. The logit lens confirms the model never predicts the correct next token under the cipher mapping.

### Implications
- **For adversarial robustness:** Simple bijective token shuffling is sufficient to completely destroy a model's next-token prediction ability, despite the residual stream "looking normal" at a surface level.
- **For cipher jailbreaks:** The model needs explicit in-context demonstrations to learn a cipher mapping — it cannot induce the mapping from ciphered text alone.
- **For mechanistic interpretability:** Cosine similarity of residual streams can be misleading as a measure of "understanding" — architectural convergence (LayerNorm) creates high similarity between any two inputs.
- **For the original question:** "After a lot of text, will the residual stream look kind of normal, modulo the shuffle?" — It looks *superficially* normal at later layers, but this is not because the model recovered. The model never recovers because the embedding lookup creates a non-linear distortion that cannot be inverted by the forward pass.

### Confidence in Findings
**High confidence** in the main finding (no recovery modulo shuffle). The result is consistent across two models, four shuffle rates, and multiple analysis methods. The architecture-convergence control experiment provides a clear mechanistic explanation.

## 8. Next Steps

### Immediate Follow-ups
1. **Add in-context demonstrations:** Provide (ciphered → clean) token mappings in context and measure if the residual stream THEN converges modulo the shuffle (following ICL Ciphers methodology).
2. **Test larger models:** Llama-3.1-8B/70B may show different behavior due to greater capacity for in-context learning.
3. **Fine-grained position analysis:** Track per-token recovery as a function of how many times each token has appeared in context.

### Alternative Approaches
- Use SAEs (Sparse Autoencoders) to compare which features fire for clean vs. ciphered text
- Apply ACDC/EAP circuit discovery to identify what circuits activate under cipher
- Test with character-level or byte-level models where the "embedding lookup" step is simpler

### Open Questions
- How many in-context demonstrations are needed for genuine residual-stream recovery?
- Is there a model size threshold below which recovery is impossible and above which it emerges?
- Can a small amount of fine-tuning on ciphered text enable recovery in a way the forward pass alone cannot?

## References

1. Fang et al. (2025). "ICL CIPHERS: Quantifying Learning in In-Context Learning via Substitution Ciphers." arXiv:2504.19395.
2. Lawson et al. (2025). "Residual Stream Analysis with Multi-Layer SAEs." arXiv:2409.04185.
3. Belrose et al. (2023). "Eliciting Latent Predictions from Transformers with the Tuned Lens." arXiv:2303.08112.
4. Zhao et al. (2024). "Analysing the Residual Stream Under Knowledge Conflicts." arXiv:2410.16090.
5. Yuan et al. (2023). "GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher." arXiv:2308.06463.
6. Sinha et al. (2021). "Masked Language Modeling and the Distributional Hypothesis: Order Word Matters Pre-training for Little." arXiv:2104.06644.

## Appendix: Generated Plots

| Plot | File | Description |
|------|------|-------------|
| Cosine similarity across layers | `results/plots/cosine_similarity_layers.png` | Raw and adjusted similarity for all shuffle rates |
| Logit lens ranks | `results/plots/logit_lens_ranks.png` | Token rank at each layer for clean and cipher |
| Position-dependent similarity | `results/plots/position_dependent_similarity.png` | Per-token-position similarity |
| Summary comparison | `results/plots/summary_comparison.png` | Bar chart of final-layer similarities |
| Context length recovery | `results/plots/context_length_recovery.png` | Does more context help? |
| Deeper analysis | `results/plots/deeper_analysis.png` | Norms, inter-text sim, probes, attention |
| Pythia analysis | `results/plots/pythia_analysis.png` | Pythia-410M raw, adjusted, and CKA |
