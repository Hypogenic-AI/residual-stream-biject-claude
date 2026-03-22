# Literature Review: The Residual Stream After Bijective Token Transformation

## Research Area Overview

This literature review covers the intersection of three research areas relevant to studying how pretrained language models handle bijective token-surface form remappings and whether their residual stream adapts:

1. **Residual stream analysis and mechanistic interpretability** — understanding how transformers process and refine predictions layer-by-layer through the residual stream
2. **Token permutation, word order, and cipher robustness** — how LMs respond when input tokens are shuffled, permuted, or encoded via substitution ciphers
3. **In-context learning of new mappings** — whether models can learn arbitrary bijective mappings from demonstrations alone

## Key Papers

### 1. ICL CIPHERS: Quantifying "Learning" in In-Context Learning via Substitution Ciphers
- **Authors**: Zhouxiang Fang, Aayush Mishra, Muhan Gao, Anqi Liu, Daniel Khashabi (Johns Hopkins)
- **Year**: 2025
- **Source**: arXiv:2504.19395
- **Key Contribution**: Introduces a rigorous framework to distinguish task learning from task retrieval in ICL using bijective vs. non-bijective substitution ciphers
- **Methodology**: Applies random bijective token-level substitution ciphers to NLP task inputs and compares model performance against a non-bijective (random noise) baseline. The gap measures genuine "task learning." Uses Zipfian frequency-matched shuffling and priority sampling of demonstrations.
- **Datasets Used**: SST-2, Amazon Reviews, HellaSwag, WinoGrande
- **Models**: Llama-3.1-8B/70B, Qwen2.5-7B, OLMo-7B, Gemma-2-9b
- **Results**: Bijective ciphers consistently outperform non-bijective across all models (e.g., +4.8% on SST-2, +7.6% on Amazon for Llama-3.1-8B). The gap widens with more demonstrations (from +4.7% at 5-shot to +10.1% at 50-shot). Peak gap at moderate shuffle rates (r ∈ 0.4–0.6).
- **Residual Stream Finding**: Using Logit Lens probing, the authors show that in deeper layers under bijective ciphers, the model's residual stream increasingly assigns higher probability to the substituted token over the original — evidence the model internally learns the cipher mapping layer-by-layer. No such trend under non-bijective ciphers.
- **Relevance**: **Directly addresses our research hypothesis.** Demonstrates that bijective token transformations are learnable in-context, and the residual stream encodes the cipher.

### 2. Residual Stream Analysis with Multi-Layer SAEs
- **Authors**: Tim Lawson, Lucy Farnik, Conor Houghton, Laurence Aitchison (Bristol)
- **Year**: 2025 (ICLR)
- **Source**: arXiv:2409.04185
- **Key Contribution**: Introduces Multi-Layer SAE (MLSAE) — a single sparse autoencoder trained across all transformer layers to study cross-layer information flow in the residual stream
- **Methodology**: Trains TopK sparse autoencoders on residual stream activations from all layers simultaneously. Analyzes the distribution of latent activations over layers using variance decomposition. Compares with and without tuned-lens basis transformations.
- **Datasets Used**: The Pile (10M tokens for evaluation)
- **Models**: Pythia suite (70M–2.8B), GPT-2 Small, Gemma 2 2B, Llama 3.2 3B
- **Results**: Individual MLSAE latents fire at a single layer for a given token (low within-token variance) but are active at different layers across tokens (high aggregate variance — ~100x larger). Larger models show more cross-layer activity. Tuned-lens normalization doesn't change qualitative findings.
- **Code Available**: Yes — https://github.com/tim-lawson/mlsae
- **Relevance**: Provides the methodology (MLSAE) to study how a bijective transformation affects feature representations across all layers simultaneously.

### 3. Eliciting Latent Predictions from Transformers with the Tuned Lens
- **Authors**: Nora Belrose, Igor Ostrovsky, Lev McKinney et al. (EleutherAI)
- **Year**: 2023
- **Source**: arXiv:2303.08112
- **Key Contribution**: The tuned lens — trained affine probes per layer that decode residual stream states into vocabulary distributions, more reliably than the logit lens
- **Methodology**: Trains per-layer affine transformations to project hidden states into the final layer's representation space. Uses distillation loss. Introduces Causal Basis Extraction (CBE) to find directions most influential on predictions.
- **Datasets Used**: The Pile (validation split)
- **Models**: Pythia suite, GPT-Neo, GPT-NeoX-20B, BLOOM, OPT, LLaMA
- **Results**: Tuned lens predictions have lower perplexity and less bias than logit lens. CBE features causally affect model outputs. Hidden state covariance drifts smoothly across layers (representational drift). Detects malicious inputs from prediction trajectories.
- **Code Available**: Yes — https://github.com/AlignmentResearch/tuned-lens
- **Relevance**: Essential tool for probing how residual stream predictions evolve under bijective transformation — can we see the model "decode" the cipher through the tuned lens?

### 4. Analysing the Residual Stream Under Knowledge Conflicts
- **Authors**: Yu Zhao, Xiaotang Du, Giwon Hong et al. (Edinburgh, CUHK, UCL)
- **Year**: 2024
- **Source**: arXiv:2410.16090
- **Key Contribution**: Shows that knowledge conflicts (parametric vs. contextual) are detectable through residual stream probing, with distinct activation patterns
- **Methodology**: Logistic regression probes on residual stream activations at each layer. Analyzes distributional shape (kurtosis, Hoyer measure, Gini index) of activations.
- **Datasets Used**: NQSwap, Macnoise, ConflictQA
- **Results**: Conflict detected at mid-layers (~13 of 32 in Llama3-8B, ~90% accuracy). Knowledge source selection distinguishable at later layers (~17–20) via skewness differences. The combined residual stream (not MLP or attention alone) carries the signal.
- **Relevance**: Demonstrates that competing information sources create detectable patterns in the residual stream — analogous to our setting where the model must reconcile its trained token semantics with the bijective remapping.

### 5. Towards Automated Circuit Discovery for Mechanistic Interpretability
- **Authors**: Arthur Conmy, Augustine Mavor-Parker, Aengus Lynch et al.
- **Year**: 2023
- **Source**: arXiv:2304.14997
- **Key Contribution**: ACDC algorithm for automatically finding circuits responsible for specific model behaviors via activation patching
- **Methodology**: Recursive edge-level importance scoring using activation patching. Identifies subgraphs in transformer computational graphs.
- **Results**: Recovered 5/5 component types in GPT-2 Small's Greater-Than circuit, selecting 68 of 32,000 edges.
- **Code Available**: Yes — https://github.com/ArthurConmy/Automatic-Circuit-Discovery
- **Relevance**: Methodology for identifying which circuit components are responsible for cipher decoding behavior.

### 6. Attribution Patching Outperforms Automated Circuit Discovery
- **Authors**: Aaquib Syed, Can Rager, Arthur Conmy
- **Year**: 2023 (NeurIPS ATTRIB Workshop)
- **Source**: arXiv:2310.10348
- **Key Contribution**: Edge Attribution Patching (EAP) — faster alternative to ACDC using gradient-based approximation
- **Code Available**: Yes — https://github.com/Aaquib111/acdcpp
- **Relevance**: More efficient tool for circuit discovery that could identify which components handle cipher decoding.

### 7. GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher
- **Authors**: Youliang Yuan, Wenxiang Jiao et al.
- **Year**: 2023
- **Source**: arXiv:2308.06463
- **Key Contribution**: Shows GPT-4 can understand and respond to prompts encoded with various ciphers (Caesar, ASCII, etc.)
- **Methodology**: Tests multiple cipher types on safety-aligned LLMs. Evaluates cipher understanding without explicit in-context examples.
- **Results**: GPT-4 successfully interprets Caesar cipher, Morse code, ASCII encoding. Demonstrates LLMs have latent cipher-processing capabilities from pre-training.
- **Relevance**: Shows that large LLMs already have some cipher-processing ability, suggesting the residual stream may have pathways for handling systematic token transformations.

### 8. Masked Language Modeling and the Distributional Hypothesis: Order Word Matters Pre-training for Little
- **Authors**: Koustuv Sinha, Robin Jia, Dieuwke Hupkes, Joelle Pineau, Adina Williams, Douwe Kiela (Facebook AI Research)
- **Year**: 2021
- **Source**: arXiv:2104.06644
- **Key Contribution**: MLMs pre-trained on randomly shuffled text still achieve high accuracy on many downstream tasks
- **Methodology**: Pre-trains RoBERTa on BookWiki with various word order permutations (n-gram shuffles from n=1 to n=4). Tests on GLUE, PAWS, and syntactic probes.
- **Results**: Models pre-trained on shuffled text retain ~95-98% of natural-order performance on most GLUE tasks. Only CoLA (acceptability) and PAWS (paraphrase detection) show major degradation. Parametric syntactic probes still succeed on shuffled models, suggesting probes may be unreliable.
- **Relevance**: Demonstrates that distributional co-occurrence statistics (not syntax) drive much of MLM pre-training success. A bijective token remapping preserves co-occurrence structure, suggesting models may adapt more readily than expected.

### 9. Word Order Does Matter (And Shuffled Language Models Know It)
- **Authors**: Vinit Ravishankar, Mostafa Abdou, Artur Kulmizev et al.
- **Year**: 2022
- **Source**: arXiv:2203.10995
- **Key Contribution**: Challenges Sinha et al. by showing shuffled LMs *do* encode some word order information
- **Methodology**: Analyzes internal representations of shuffled-text-trained models using probes for syntactic dependency and word order.
- **Results**: Even models trained on shuffled text learn some positional information through the transformer architecture's inductive biases.
- **Relevance**: Shows the transformer architecture itself provides implicit structural cues even when surface order is destroyed — relevant to understanding what happens in the residual stream under bijective remapping.

### 10. Transformer Language Models without Positional Encodings Still Learn Positional Information
- **Authors**: Adi Haviv, Ori Ram, Ofir Press, Peter Izsak, Omer Levy
- **Year**: 2022
- **Source**: arXiv:2203.16634
- **Key Contribution**: Transformers without positional encodings still learn positional information from causal attention masks
- **Results**: Models without positional encodings achieve competitive perplexity and learn position from the autoregressive mask alone.
- **Relevance**: Shows positional information emerges from architecture, not just explicit encodings — the residual stream adapts to structural constraints regardless of explicit position signals.

### 11. UnNatural Language Inference
- **Authors**: Koustuv Sinha, Prasanna Parthasarathi, Joelle Pineau, Adina Williams
- **Year**: 2021
- **Source**: arXiv:2101.00010
- **Key Contribution**: NLI models are surprisingly robust to unnatural word orders in both premises and hypotheses
- **Relevance**: Further evidence that NLU models rely on distributional cues over syntactic structure.

### 12. Decipherment-Aware Multilingual Learning in Jointly Trained Language Models
- **Authors**: Grandee Lee et al. (Singapore University of Technology)
- **Year**: 2024
- **Source**: arXiv:2406.07231
- **Key Contribution**: Training LMs to jointly model cipher/decipherment alongside normal text improves multilingual transfer
- **Relevance**: Shows that explicit cipher-awareness can be trained into language models, suggesting the residual stream can represent both original and transformed token mappings.

### 13. CipherBank: Exploring the Boundary of LLM Reasoning Capabilities through Cryptography Challenges
- **Authors**: Yu Li et al.
- **Year**: 2025
- **Source**: arXiv:2504.19093
- **Key Contribution**: Systematic benchmark of LLM reasoning on cryptographic tasks including substitution ciphers
- **Results**: LLMs struggle with complex ciphers but show capability with simple substitutions.
- **Relevance**: Establishes baseline expectations for LLM cipher-processing ability.

### 14. Mechanistic Interpretability for AI Safety — A Review
- **Authors**: Leonard Bereska, Efstratios Gavves (University of Amsterdam)
- **Year**: 2024
- **Source**: arXiv:2404.14082
- **Key Contribution**: Comprehensive review of mechanistic interpretability methods
- **Relevance**: Survey of tools and techniques applicable to our analysis (activation patching, probing, SAEs, etc.).

### 15. VISIT: Visualizing and Interpreting the Semantic Information Flow of Transformers
- **Authors**: Shahar Katz, Yonatan Belinkov (Technion)
- **Year**: 2023
- **Source**: arXiv:2305.13417
- **Key Contribution**: Interactive visualization tool for information flow through transformer layers via vocabulary projection
- **Relevance**: Methodology for visualizing how semantic content transforms through the residual stream.

### 16. Emergent Linear Representations in World Models of Self-Supervised Sequence Models
- **Authors**: Neel Nanda, Andrew Lee, Martin Wattenberg
- **Year**: 2023
- **Source**: arXiv:2309.00941
- **Key Contribution**: Evidence of linear board-state representations in Othello-playing models
- **Relevance**: Demonstrates that transformers encode structured world models linearly in the residual stream — a bijective transformation might preserve or disrupt such linear structure.

### 17. Forbidden Facts: An Investigation of Competing Objectives in Llama-2
- **Authors**: Tony Wang, Miles Wang, Kai Horton et al.
- **Year**: 2023
- **Source**: arXiv:2312.08793
- **Key Contribution**: Analyzes how competing objectives (helpfulness vs. safety) manifest in internal representations
- **Relevance**: Methodology for studying how conflicting information sources are represented in the residual stream.

### 18. You can remove GPT2's LayerNorm by fine-tuning
- **Authors**: Stefan Heimersheim (Apollo Research)
- **Year**: 2024
- **Source**: arXiv:2409.13710
- **Key Contribution**: Shows GPT-2's LayerNorm can be removed via fine-tuning, revealing that normalization is not essential for learned representations
- **Relevance**: Understanding what role LayerNorm plays in the residual stream basis — important for analyzing whether bijective transformations interact with normalization layers.

## Common Methodologies

1. **Logit Lens / Tuned Lens probing**: Used in ICL Ciphers, Tuned Lens paper — projects intermediate hidden states to vocabulary space to see evolving predictions. Key tool for our research.
2. **Sparse Autoencoders (SAEs)**: Used in MLSAE paper — decomposes residual stream into interpretable features. Enables cross-layer analysis.
3. **Activation/Attribution Patching**: Used in ACDC, EAP — identifies causal role of specific components. Can identify cipher-decoding circuits.
4. **Linear probing**: Used in Knowledge Conflicts, word order papers — trains classifiers on hidden states. Tests what information the residual stream encodes.
5. **Controlled permutation experiments**: Used in word order papers — systematically varies input structure while measuring model behavior.

## Standard Baselines

- **Logit Lens** (nostalgebraist, 2020): Direct unembedding of hidden states (baseline for tuned lens)
- **Random baseline**: Non-bijective cipher (ICL Ciphers) or random word order (Sinha et al.)
- **Clean / natural order**: Original text without transformation
- **Frequency-matched controls**: Zipfian frequency matching for cipher tokens (ICL Ciphers)

## Evaluation Metrics

- **Task accuracy**: Primary metric for downstream performance under transformation
- **Bijective–Non-bijective gap**: Measures genuine learning vs. retrieval (ICL Ciphers)
- **Perplexity of intermediate predictions**: Tuned lens / logit lens output perplexity per layer
- **Logit rank difference**: Rank of original vs. substituted token at intermediate layers
- **Probing accuracy**: Linear probe accuracy for detecting specific information in hidden states
- **Activation distribution statistics**: Kurtosis, Hoyer measure, Gini index of residual stream
- **Cosine similarity**: Between hidden states at different layers or under different conditions
- **FVU (Fraction of Variance Unexplained)**: SAE reconstruction quality

## Gaps and Opportunities

1. **No systematic study of residual stream under bijective token transformation**: ICL Ciphers provides behavioral evidence and basic Logit Lens probing, but does not deeply analyze how the residual stream structure changes. Our research fills this gap.
2. **Bijective vs. random permutation on representations**: Word order studies focus on downstream task performance, not on internal representation changes under bijective mappings.
3. **Cross-layer cipher decoding analysis**: The MLSAE approach has not been applied to cipher/transformed inputs to study how features evolve differently.
4. **Tuned lens under transformation**: No work has applied the tuned lens to compare prediction trajectories between normal and bijectively-transformed inputs.
5. **Circuit identification for cipher processing**: ACDC/EAP have not been applied to identify which circuits are responsible for cipher decoding.

## Recommendations for Our Experiment

Based on this literature review:

- **Recommended datasets**: SST-2 (primary, well-studied), Amazon Reviews (secondary, larger), HellaSwag and WinoGrande (for generalization testing)
- **Recommended models**: Pythia suite (various sizes, well-understood, MLSAE pretrained), Llama-3.1-8B (used in ICL Ciphers), GPT-2 Small (simplest for circuit analysis)
- **Recommended baselines**: (1) Clean text, (2) Bijective cipher, (3) Non-bijective random, (4) Varying shuffle rates
- **Recommended analysis tools**: Tuned Lens (layer-by-layer prediction tracking), MLSAE (cross-layer feature analysis), Logit Lens (quick baseline), TransformerLens (hook-based activation access)
- **Recommended metrics**: Logit rank difference (as in ICL Ciphers), tuned lens perplexity trajectory, cosine similarity between clean and ciphered hidden states, SAE feature overlap analysis
- **Methodological considerations**: Use Zipfian frequency matching for cipher tokens; test at multiple shuffle rates (0.3, 0.5, 0.7); ensure demonstrations contain cipher mappings for test tokens; compare both pre-trained and instruction-tuned models
