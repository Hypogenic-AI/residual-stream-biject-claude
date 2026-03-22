# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "The Residual Stream After Bijective Token Transformation." Resources include papers on residual stream analysis, cipher/substitution learning, word order perturbation, and mechanistic interpretability tools.

## Papers
Total papers downloaded: 19

| # | Title | Authors | Year | File | Category |
|---|-------|---------|------|------|----------|
| 1 | ICL CIPHERS: Quantifying Learning via Substitution Ciphers | Fang et al. | 2025 | papers/icl_ciphers.pdf | Cipher/ICL |
| 2 | Residual Stream Analysis with Multi-Layer SAEs | Lawson et al. | 2025 | papers/residual_stream_multi_layer_sae.pdf | Residual Stream |
| 3 | Eliciting Latent Predictions with the Tuned Lens | Belrose et al. | 2023 | papers/tuned_lens.pdf | Residual Stream |
| 4 | Analysing Residual Stream Under Knowledge Conflicts | Zhao et al. | 2024 | papers/residual_stream_knowledge_conflicts.pdf | Residual Stream |
| 5 | Emergent Linear Representations in World Models | Nanda et al. | 2023 | papers/emergent_linear_representations.pdf | Residual Stream |
| 6 | GPT-4 Is Too Smart: Stealthy Chat via Cipher | Yuan et al. | 2023 | papers/gpt4_cipher.pdf | Cipher |
| 7 | CipherBank: LLM Reasoning via Cryptography | Li et al. | 2025 | papers/cipherbank.pdf | Cipher |
| 8 | Decipherment-Aware Multilingual Learning | Lee et al. | 2024 | papers/decipherment_multilingual.pdf | Cipher |
| 9 | Masked LM: Order Word Matters for Little | Sinha et al. | 2021 | papers/word_order_matters_pretraining.pdf | Word Order |
| 10 | Word Order Does Matter (Shuffled LMs Know It) | Ravishankar et al. | 2022 | papers/shuffled_lm_know_order.pdf | Word Order |
| 11 | Transformer LMs without Positional Encodings | Haviv et al. | 2022 | papers/transformer_no_pos_enc.pdf | Word Order |
| 12 | Out of Order: Sequential Word Order Importance | Pham et al. | 2020 | papers/bert_without_word_ordering.pdf | Word Order |
| 13 | UnNatural Language Inference | Sinha et al. | 2021 | papers/unnatural_language_inference.pdf | Word Order |
| 14 | Towards Automated Circuit Discovery (ACDC) | Conmy et al. | 2023 | papers/automated_circuit_discovery.pdf | Mech Interp |
| 15 | Attribution Patching Outperforms ACDC | Syed et al. | 2023 | papers/exploring_residual_stream.pdf | Mech Interp |
| 16 | Mechanistic Interpretability for AI Safety — A Review | Bereska & Gavves | 2024 | papers/mech_interp_review.pdf | Mech Interp |
| 17 | VISIT: Visualizing Semantic Information Flow | Katz & Belinkov | 2023 | papers/visit_semantic_flow.pdf | Mech Interp |
| 18 | Removing GPT2's LayerNorm by Fine-tuning | Heimersheim | 2024 | papers/remove_layernorm.pdf | Embedding |
| 19 | Forbidden Facts: Competing Objectives in Llama-2 | Wang et al. | 2023 | papers/forbidden_facts.pdf | Mech Interp |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 4 (+ 1 streaming reference)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| SST-2 | HuggingFace stanfordnlp/sst2 | 67K train, 872 val, 1.8K test | Sentiment | datasets/sst2/ | Primary benchmark |
| Amazon Reviews (sample) | HuggingFace amazon_polarity | 1K test sample | Sentiment | datasets/amazon_reviews_sample/ | Secondary benchmark |
| HellaSwag | HuggingFace Rowan/hellaswag | 500 validation | Completion | datasets/hellaswag/ | Generalization test |
| WinoGrande | HuggingFace allenai/winogrande | 500 validation | Pronoun resolution | datasets/winogrande/ | Generalization test |
| The Pile | HuggingFace monology/pile-uncopyrighted | ~800GB (stream) | LM / SAE training | N/A (streaming) | For Pythia/SAE work |

See datasets/README.md for download instructions.

## Code Repositories
Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| MLSAE | github.com/tim-lawson/mlsae | Cross-layer SAE analysis | code/mlsae/ | Pre-trained models on HF |
| Tuned Lens | github.com/AlignmentResearch/tuned-lens | Layer-by-layer prediction probing | code/tuned-lens/ | pip install tuned-lens |
| ACDC | github.com/ArthurConmy/Automatic-Circuit-Discovery | Circuit discovery | code/acdc/ | Requires TransformerLens |
| ACDC++ | github.com/Aaquib111/acdcpp | Fast circuit discovery | code/acdcpp/ | Edge Attribution Patching |
| TransformerLens | github.com/neelnanda-io/TransformerLens | Mechanistic interp library | code/TransformerLens/ | pip install transformer-lens |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Used paper-finder service with Semantic Scholar API for relevance-ranked results
- Three search queries: (1) residual stream transformer interpretability, (2) cipher text language model robustness token permutation, (3) language model input perturbation embedding space token shuffling
- Supplemented with arxiv library searches for specific papers by title
- Cross-referenced datasets and code from papers' methods sections

### Selection Criteria
Papers selected based on:
1. Direct relevance to bijective token transformation (cipher papers)
2. Methodology for residual stream analysis (probing, SAEs, circuit discovery)
3. Understanding of token permutation effects on LMs (word order studies)
4. Foundational mechanistic interpretability tools and surveys

### Challenges Encountered
- ArXiv PDF downloads sometimes returned different papers than expected for some IDs; resolved by searching by title via arxiv library
- Semantic Scholar API rate limiting required staggered requests
- Some key papers (ICL Ciphers) were very recent (2025), confirming this is an active research frontier

### Gaps and Workarounds
- No existing work applies MLSAE or tuned lens specifically to bijectively-transformed inputs — this is our research contribution
- The ICL Ciphers paper provides behavioral evidence but limited residual stream analysis — our work deepens this
- No circuit-level analysis of cipher decoding exists — ACDC/EAP tools are available but unapplied to this question

## Recommendations for Experiment Design

1. **Primary dataset(s)**: SST-2 (binary sentiment, used in ICL Ciphers, fast evaluation), Amazon Reviews (larger, more challenging)
2. **Primary models**: Pythia-410M or Pythia-1.4B (MLSAE pre-trained models available), GPT-2 Small (simplest for circuit analysis), Llama-3.1-8B (if compute allows, for comparison with ICL Ciphers)
3. **Baseline methods**: (a) Clean text, (b) Bijective cipher at shuffle rates 0.3/0.5/0.7, (c) Non-bijective random cipher (ICL Ciphers control), (d) Varying number of ICL demonstrations
4. **Evaluation metrics**: Task accuracy gap (bijective vs. non-bijective), logit rank difference at each layer, tuned lens perplexity trajectory, cosine similarity of clean vs. ciphered hidden states, MLSAE feature activation overlap
5. **Code to adapt/reuse**: TransformerLens for activation hooks, tuned-lens for prediction probing, MLSAE for feature analysis. Implement bijective cipher following ICL Ciphers' Zipfian frequency matching approach.
