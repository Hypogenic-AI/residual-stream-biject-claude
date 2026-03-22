"""
Main experiment: Residual stream analysis under bijective token transformation.

Runs GPT-2 Small on clean vs. ciphered text and compares residual stream
activations at each layer to test whether the model "recovers" modulo the permutation.
"""

import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
from bijective_cipher import (
    apply_cipher,
    create_bijective_cipher,
    get_permutation_matrix_in_embedding_space,
)

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


def load_model(model_name="gpt2"):
    """Load model via TransformerLens."""
    import transformer_lens
    model = transformer_lens.HookedTransformer.from_pretrained(model_name, device=DEVICE)
    model.eval()
    return model


def get_text_samples(n_samples=50, max_length=256):
    """Get text samples for analysis."""
    # Use a mix of coherent text
    texts = [
        "The quick brown fox jumps over the lazy dog. This is a simple sentence that any language model should handle well.",
        "In the year 2024, artificial intelligence continued to advance at a remarkable pace. Researchers developed new methods for understanding how neural networks process information.",
        "The stock market experienced significant volatility throughout the quarter. Investors were concerned about rising interest rates and their impact on corporate earnings.",
        "A group of scientists discovered a new species of deep-sea fish near the Mariana Trench. The creature was bioluminescent and had adapted to extreme pressure conditions.",
        "The recipe calls for two cups of flour, one cup of sugar, and three eggs. Mix the dry ingredients first, then slowly add the wet ingredients while stirring.",
        "Shakespeare wrote many of his most famous plays during the late sixteenth and early seventeenth centuries. His works continue to be performed and studied worldwide.",
        "Machine learning models learn patterns from data through iterative optimization of their parameters. The loss function guides the model toward better predictions over time.",
        "The city council voted to approve the new public transportation plan. The project would add three new bus routes and extend the subway system by five miles.",
        "Climate change is causing sea levels to rise at an accelerating rate. Coastal communities around the world are developing adaptation strategies to protect their infrastructure.",
        "The professor explained the concept of quantum entanglement to the class. When two particles are entangled, measuring one instantly affects the state of the other.",
        "During the Renaissance, European art underwent a dramatic transformation. Artists like Leonardo da Vinci and Michelangelo pioneered new techniques in painting and sculpture.",
        "The software engineering team implemented a new microservices architecture. Each service handles a specific domain function and communicates through well-defined APIs.",
        "Photosynthesis is the process by which plants convert sunlight into chemical energy. This process is essential for life on Earth as it produces oxygen and organic compounds.",
        "The detective carefully examined the crime scene for clues. Fingerprints on the window suggested that the intruder had entered through the second-floor balcony.",
        "Economic theory suggests that markets tend toward equilibrium when left to operate freely. However, real-world markets often exhibit inefficiencies due to information asymmetry.",
        "The ancient Egyptians built the pyramids using sophisticated engineering techniques. These massive structures have survived for thousands of years and continue to fascinate researchers.",
        "Neural networks consist of layers of interconnected nodes that process information. Each connection has a weight that is adjusted during training to minimize prediction errors.",
        "The orchestra performed Beethoven's Ninth Symphony to a packed concert hall. The final movement, with its famous Ode to Joy, received a standing ovation from the audience.",
        "Advances in genomics have made it possible to sequence an entire human genome in just a few hours. This technology has revolutionized personalized medicine and disease research.",
        "The hiking trail wound through dense forest before emerging at a spectacular mountain overlook. From the summit, hikers could see three different states on a clear day.",
    ]

    # Repeat and vary to get enough samples
    while len(texts) < n_samples:
        texts.extend(texts[:n_samples - len(texts)])

    return texts[:n_samples]


def compute_token_frequencies(model, texts):
    """Estimate token frequencies from sample texts."""
    freq = np.zeros(model.cfg.d_vocab, dtype=np.float64)
    for text in texts:
        tokens = model.to_tokens(text, prepend_bos=True)[0]
        for t in tokens.cpu().numpy():
            freq[t] += 1
    return freq


def extract_residual_streams(model, tokens):
    """Extract residual stream activations at every layer."""
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, remove_batch_dim=False)

    # Get residual stream at each layer (after each transformer block)
    n_layers = model.cfg.n_layers
    residuals = []

    # Layer 0: after embedding (hook_embed + hook_pos_embed = resid_pre layer 0)
    residuals.append(cache["blocks.0.hook_resid_pre"][0].cpu())  # (seq_len, d_model)

    # After each layer
    for layer in range(n_layers):
        residuals.append(cache[f"blocks.{layer}.hook_resid_post"][0].cpu())

    return residuals  # list of (seq_len, d_model) tensors


def cosine_similarity_per_position(a, b):
    """Compute cosine similarity between two activation tensors at each position."""
    # a, b: (seq_len, d_model)
    a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-10)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-10)
    return (a_norm * b_norm).sum(dim=-1)  # (seq_len,)


def logit_lens_predictions(model, residuals, top_k=5):
    """Apply logit lens: project residual stream to vocabulary space."""
    unembed = model.W_U.cpu().float()  # (d_model, vocab_size)
    unembed_bias = model.b_U.cpu().float() if model.b_U is not None else None

    results = []
    for layer_residual in residuals:
        logits = layer_residual.float() @ unembed  # (seq_len, vocab_size)
        if unembed_bias is not None:
            logits = logits + unembed_bias
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_tokens = probs.topk(top_k, dim=-1)
        results.append({
            "top_tokens": top_tokens,  # (seq_len, top_k)
            "top_probs": top_probs,
            "logits": logits,
        })
    return results


def run_experiment_single_text(
    model, text, forward_map, inverse_map, T_matrix, cipher_name="cipher"
):
    """Run experiment on a single text: compare clean vs ciphered residual streams."""
    # Tokenize
    clean_tokens = model.to_tokens(text, prepend_bos=True)  # (1, seq_len)
    seq_len = clean_tokens.shape[1]

    # Apply cipher
    ciphered_tokens = apply_cipher(clean_tokens, forward_map.to(clean_tokens.device))

    # Also create a non-bijective (random) baseline
    random_tokens = torch.randint(0, model.cfg.d_vocab, clean_tokens.shape, device=clean_tokens.device)

    # Extract residual streams
    clean_residuals = extract_residual_streams(model, clean_tokens)
    cipher_residuals = extract_residual_streams(model, ciphered_tokens)
    random_residuals = extract_residual_streams(model, random_tokens)

    n_layers = len(clean_residuals)  # n_layers + 1 (embedding + each block)

    # 1. Raw cosine similarity per layer
    raw_cos_sim = []
    random_cos_sim = []
    for layer in range(n_layers):
        cs = cosine_similarity_per_position(clean_residuals[layer], cipher_residuals[layer])
        raw_cos_sim.append(cs.mean().item())
        cs_rand = cosine_similarity_per_position(clean_residuals[layer], random_residuals[layer])
        random_cos_sim.append(cs_rand.mean().item())

    # 2. Permutation-adjusted similarity (T_matrix precomputed)
    T = T_matrix

    adjusted_cos_sim = []
    for layer in range(n_layers):
        # Transform ciphered residuals by T to "undo" the permutation
        adjusted = cipher_residuals[layer].float() @ T.float()
        cs = cosine_similarity_per_position(clean_residuals[layer].float(), adjusted)
        adjusted_cos_sim.append(cs.mean().item())

    # 3. Logit lens analysis
    clean_logits = logit_lens_predictions(model, clean_residuals)
    cipher_logits = logit_lens_predictions(model, cipher_residuals)

    # For each layer, check if cipher logit lens predicts the ciphered next token
    # The "correct" next token under cipher is forward_map[original_next_token]
    clean_next_tokens = clean_tokens[0, 1:]  # (seq_len-1,)
    cipher_next_tokens = forward_map[clean_next_tokens.cpu()]  # what the model "should" predict if it decoded the cipher

    logit_lens_clean_rank = []
    logit_lens_cipher_correct_rank = []
    logit_lens_cipher_original_rank = []

    for layer in range(n_layers):
        # Clean: rank of correct next token
        clean_layer_logits = clean_logits[layer]["logits"][:-1]  # (seq_len-1, vocab)
        clean_ranks = (clean_layer_logits > clean_layer_logits.gather(1, clean_next_tokens.cpu().unsqueeze(1))).sum(dim=1).float()
        logit_lens_clean_rank.append(clean_ranks.mean().item())

        # Cipher: rank of the ciphered-correct next token
        cipher_layer_logits = cipher_logits[layer]["logits"][:-1]
        # What rank does the model assign to the "correct under cipher" token?
        cipher_correct_ranks = (cipher_layer_logits > cipher_layer_logits.gather(1, cipher_next_tokens.unsqueeze(1))).sum(dim=1).float()
        logit_lens_cipher_correct_rank.append(cipher_correct_ranks.mean().item())

        # Cipher: rank of the original (clean) next token
        cipher_original_ranks = (cipher_layer_logits > cipher_layer_logits.gather(1, clean_next_tokens.cpu().unsqueeze(1))).sum(dim=1).float()
        logit_lens_cipher_original_rank.append(cipher_original_ranks.mean().item())

    # 4. Position-dependent analysis: does similarity increase with more context?
    position_cos_sims = {}
    for layer in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
        cs = cosine_similarity_per_position(clean_residuals[layer], cipher_residuals[layer])
        position_cos_sims[f"layer_{layer}"] = cs.tolist()

    return {
        "seq_len": seq_len,
        "raw_cos_sim": raw_cos_sim,
        "random_cos_sim": random_cos_sim,
        "adjusted_cos_sim": adjusted_cos_sim,
        "logit_lens_clean_rank": logit_lens_clean_rank,
        "logit_lens_cipher_correct_rank": logit_lens_cipher_correct_rank,
        "logit_lens_cipher_original_rank": logit_lens_cipher_original_rank,
        "position_cos_sims": position_cos_sims,
    }


def run_all_experiments(model_name="gpt2"):
    """Run full experiment suite."""
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")
    model = load_model(model_name)

    texts = get_text_samples(n_samples=30)
    token_frequencies = compute_token_frequencies(model, texts)

    shuffle_rates = [0.3, 0.5, 0.7, 1.0]
    all_results = {}

    for shuffle_rate in shuffle_rates:
        print(f"\n--- Shuffle rate: {shuffle_rate} ---")
        forward_map, inverse_map = create_bijective_cipher(
            vocab_size=model.cfg.d_vocab,
            seed=SEED,
            shuffle_rate=shuffle_rate,
            frequency_matched=True,
            token_frequencies=token_frequencies,
        )

        # Count how many tokens are actually shuffled
        n_shuffled = (forward_map != torch.arange(model.cfg.d_vocab)).sum().item()
        print(f"Tokens shuffled: {n_shuffled}/{model.cfg.d_vocab} ({100*n_shuffled/model.cfg.d_vocab:.1f}%)")

        # Precompute the permutation-adjustment matrix (expensive lstsq)
        E = model.W_E.cpu().float()
        T_matrix = get_permutation_matrix_in_embedding_space(E, forward_map.cpu())
        print(f"Permutation adjustment matrix computed: {T_matrix.shape}")

        results_for_rate = []
        for i, text in enumerate(tqdm(texts, desc=f"Processing texts (r={shuffle_rate})")):
            result = run_experiment_single_text(
                model, text, forward_map, inverse_map, T_matrix,
                cipher_name=f"bijective_r{shuffle_rate}"
            )
            results_for_rate.append(result)

        all_results[f"shuffle_{shuffle_rate}"] = results_for_rate

    return all_results, model


def aggregate_results(all_results):
    """Aggregate results across texts for each shuffle rate."""
    aggregated = {}

    for key, results_list in all_results.items():
        n_layers = len(results_list[0]["raw_cos_sim"])

        raw_sims = np.array([r["raw_cos_sim"] for r in results_list])
        random_sims = np.array([r["random_cos_sim"] for r in results_list])
        adjusted_sims = np.array([r["adjusted_cos_sim"] for r in results_list])
        clean_ranks = np.array([r["logit_lens_clean_rank"] for r in results_list])
        cipher_correct_ranks = np.array([r["logit_lens_cipher_correct_rank"] for r in results_list])
        cipher_original_ranks = np.array([r["logit_lens_cipher_original_rank"] for r in results_list])

        aggregated[key] = {
            "n_layers": n_layers,
            "raw_cos_sim_mean": raw_sims.mean(axis=0).tolist(),
            "raw_cos_sim_std": raw_sims.std(axis=0).tolist(),
            "random_cos_sim_mean": random_sims.mean(axis=0).tolist(),
            "random_cos_sim_std": random_sims.std(axis=0).tolist(),
            "adjusted_cos_sim_mean": adjusted_sims.mean(axis=0).tolist(),
            "adjusted_cos_sim_std": adjusted_sims.std(axis=0).tolist(),
            "logit_clean_rank_mean": clean_ranks.mean(axis=0).tolist(),
            "logit_clean_rank_std": clean_ranks.std(axis=0).tolist(),
            "logit_cipher_correct_rank_mean": cipher_correct_ranks.mean(axis=0).tolist(),
            "logit_cipher_correct_rank_std": cipher_correct_ranks.std(axis=0).tolist(),
            "logit_cipher_original_rank_mean": cipher_original_ranks.mean(axis=0).tolist(),
            "logit_cipher_original_rank_std": cipher_original_ranks.std(axis=0).tolist(),
        }

        # Position-dependent analysis (average across texts for key layers)
        position_data = {}
        for layer_key in results_list[0]["position_cos_sims"]:
            all_pos = [r["position_cos_sims"][layer_key] for r in results_list]
            # Pad to same length
            max_len = max(len(p) for p in all_pos)
            padded = [p + [np.nan] * (max_len - len(p)) for p in all_pos]
            arr = np.array(padded)
            position_data[layer_key] = {
                "mean": np.nanmean(arr, axis=0).tolist(),
                "std": np.nanstd(arr, axis=0).tolist(),
            }
        aggregated[key]["position_cos_sims"] = position_data

    return aggregated


def create_plots(aggregated, model_name="gpt2"):
    """Generate all plots."""
    fig_dir = PLOTS_DIR

    # Plot 1: Cosine similarity across layers for each shuffle rate
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for key in sorted(aggregated.keys()):
        data = aggregated[key]
        layers = np.arange(data["n_layers"])
        rate = key.split("_")[1]

        # Raw similarity
        mean = np.array(data["raw_cos_sim_mean"])
        std = np.array(data["raw_cos_sim_std"])
        axes[0].plot(layers, mean, label=f"r={rate}", marker="o", markersize=3)
        axes[0].fill_between(layers, mean - std, mean + std, alpha=0.15)

        # Adjusted similarity
        mean_adj = np.array(data["adjusted_cos_sim_mean"])
        std_adj = np.array(data["adjusted_cos_sim_std"])
        axes[1].plot(layers, mean_adj, label=f"r={rate}", marker="o", markersize=3)
        axes[1].fill_between(layers, mean_adj - std_adj, mean_adj + std_adj, alpha=0.15)

    # Random baseline on raw plot
    random_mean = np.array(list(aggregated.values())[0]["random_cos_sim_mean"])
    axes[0].plot(layers, random_mean, label="random (non-bijective)", color="gray", linestyle="--")

    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_title(f"Raw: Clean vs. Ciphered Residual Stream ({model_name})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_title(f"Permutation-Adjusted: Clean vs. Ciphered ({model_name})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "cosine_similarity_layers.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: cosine_similarity_layers.png")

    # Plot 2: Logit lens rank analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for key in sorted(aggregated.keys()):
        data = aggregated[key]
        layers = np.arange(data["n_layers"])
        rate = key.split("_")[1]

        axes[0].plot(layers, data["logit_clean_rank_mean"], label=f"clean", marker="o", markersize=3)
        axes[1].plot(layers, data["logit_cipher_correct_rank_mean"], label=f"r={rate}", marker="o", markersize=3)
        axes[2].plot(layers, data["logit_cipher_original_rank_mean"], label=f"r={rate}", marker="o", markersize=3)

    axes[0].set_title("Clean: Rank of Correct Next Token")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Mean Rank (lower=better)")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Cipher: Rank of Cipher-Correct Token")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Mean Rank (lower=better)")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_title("Cipher: Rank of Original (Clean) Token")
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Mean Rank (lower=better)")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "logit_lens_ranks.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: logit_lens_ranks.png")

    # Plot 3: Position-dependent similarity (does model recover with more context?)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, key in enumerate(sorted(aggregated.keys())):
        data = aggregated[key]
        rate = key.split("_")[1]
        ax = axes[idx]

        for layer_key in sorted(data["position_cos_sims"].keys()):
            pos_data = data["position_cos_sims"][layer_key]
            positions = np.arange(len(pos_data["mean"]))
            ax.plot(positions, pos_data["mean"], label=layer_key.replace("_", " "), alpha=0.8)

        ax.set_xlabel("Token Position")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"Position-wise Similarity (shuffle rate={rate})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "position_dependent_similarity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: position_dependent_similarity.png")

    # Plot 4: Comparison summary - raw vs adjusted vs random
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.2
    x = np.arange(len(aggregated))
    labels = []

    raw_final = []
    adj_final = []
    rand_final = []
    raw_mid = []
    adj_mid = []

    for key in sorted(aggregated.keys()):
        data = aggregated[key]
        n = data["n_layers"]
        labels.append(f"r={key.split('_')[1]}")
        raw_final.append(data["raw_cos_sim_mean"][-1])
        adj_final.append(data["adjusted_cos_sim_mean"][-1])
        rand_final.append(data["random_cos_sim_mean"][-1])
        raw_mid.append(data["raw_cos_sim_mean"][n // 2])
        adj_mid.append(data["adjusted_cos_sim_mean"][n // 2])

    ax.bar(x - width, raw_final, width, label="Raw (final layer)", color="steelblue")
    ax.bar(x, adj_final, width, label="Adjusted (final layer)", color="coral")
    ax.bar(x + width, rand_final, width, label="Random (final layer)", color="gray")

    ax.set_xlabel("Shuffle Rate")
    ax.set_ylabel("Cosine Similarity with Clean")
    ax.set_title("Final Layer: Clean vs. Ciphered Residual Stream Similarity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(fig_dir / "summary_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: summary_comparison.png")


def run_context_length_experiment(model, forward_map, texts):
    """Test whether more context helps the model recover."""
    print("\n--- Context Length Experiment ---")

    # Use a longer text by concatenating
    long_text = " ".join(texts[:10])
    clean_tokens = model.to_tokens(long_text, prepend_bos=True)
    max_len = min(clean_tokens.shape[1], 512)
    clean_tokens = clean_tokens[:, :max_len]

    ciphered_tokens = apply_cipher(clean_tokens, forward_map.to(clean_tokens.device))

    # Extract residual streams
    with torch.no_grad():
        _, clean_cache = model.run_with_cache(clean_tokens, remove_batch_dim=False)
        _, cipher_cache = model.run_with_cache(ciphered_tokens, remove_batch_dim=False)

    n_layers = model.cfg.n_layers
    results = {"layers": list(range(n_layers + 1))}

    # For each layer, compute position-wise cosine similarity
    for layer_idx in range(n_layers + 1):
        if layer_idx == 0:
            clean_h = clean_cache["blocks.0.hook_resid_pre"][0].cpu()
            cipher_h = cipher_cache["blocks.0.hook_resid_pre"][0].cpu()
        else:
            clean_h = clean_cache[f"blocks.{layer_idx-1}.hook_resid_post"][0].cpu()
            cipher_h = cipher_cache[f"blocks.{layer_idx-1}.hook_resid_post"][0].cpu()

        cos_sim = cosine_similarity_per_position(clean_h, cipher_h)
        results[f"layer_{layer_idx}"] = cos_sim.tolist()

    return results


def run_statistical_tests(aggregated):
    """Run statistical tests on key comparisons."""
    print("\n--- Statistical Tests ---")
    stats_results = {}

    for key in sorted(aggregated.keys()):
        data = aggregated[key]
        n_layers = data["n_layers"]
        raw = np.array(data["raw_cos_sim_mean"])
        adj = np.array(data["adjusted_cos_sim_mean"])
        rand = np.array(data["random_cos_sim_mean"])

        # Test: is adjusted similarity higher than raw at later layers?
        later_layers = slice(n_layers // 2, n_layers)
        t_stat, p_val = stats.ttest_rel(adj[later_layers], raw[later_layers])
        effect_size = (adj[later_layers] - raw[later_layers]).mean() / (adj[later_layers] - raw[later_layers]).std()

        # Test: is raw similarity higher than random?
        t_stat2, p_val2 = stats.ttest_rel(raw[later_layers], rand[later_layers])

        # Test: does similarity increase from first to last layer?
        improvement = raw[-1] - raw[0]

        stats_results[key] = {
            "adjusted_vs_raw_later_layers": {
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "effect_size_d": float(effect_size),
                "adjusted_mean": float(adj[later_layers].mean()),
                "raw_mean": float(raw[later_layers].mean()),
            },
            "raw_vs_random_later_layers": {
                "t_stat": float(t_stat2),
                "p_value": float(p_val2),
                "raw_mean": float(raw[later_layers].mean()),
                "random_mean": float(rand[later_layers].mean()),
            },
            "raw_improvement_first_to_last": float(improvement),
            "raw_final_layer": float(raw[-1]),
            "adjusted_final_layer": float(adj[-1]),
            "random_final_layer": float(rand[-1]),
        }

        print(f"\n{key}:")
        print(f"  Raw sim (final layer): {raw[-1]:.4f}")
        print(f"  Adjusted sim (final layer): {adj[-1]:.4f}")
        print(f"  Random sim (final layer): {rand[-1]:.4f}")
        print(f"  Adjusted vs Raw (later layers): t={t_stat:.3f}, p={p_val:.4f}, d={effect_size:.3f}")
        print(f"  Raw vs Random (later layers): t={t_stat2:.3f}, p={p_val2:.4f}")
        print(f"  Improvement first→last: {improvement:+.4f}")

    return stats_results


def main():
    start_time = time.time()

    print("=" * 60)
    print("EXPERIMENT: Residual Stream Under Bijective Token Transformation")
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print("=" * 60)

    # Environment info
    import sys
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Run main experiments on GPT-2 Small
    all_results, model = run_all_experiments("gpt2")

    # Aggregate
    aggregated = aggregate_results(all_results)

    # Statistical tests
    stats_results = run_statistical_tests(aggregated)

    # Context length experiment
    texts = get_text_samples(n_samples=10)
    token_frequencies = compute_token_frequencies(model, texts)
    forward_map, inverse_map = create_bijective_cipher(
        vocab_size=model.cfg.d_vocab, seed=SEED, shuffle_rate=1.0,
        frequency_matched=True, token_frequencies=token_frequencies,
    )
    context_results = run_context_length_experiment(model, forward_map, texts)

    # Create plots
    create_plots(aggregated, model_name="gpt2")

    # Plot context length results
    fig, ax = plt.subplots(figsize=(12, 6))
    for layer_key in [f"layer_{i}" for i in [0, 3, 6, 9, 12]]:
        if layer_key in context_results:
            data = context_results[layer_key]
            # Smooth with rolling average
            window = 5
            smoothed = np.convolve(data, np.ones(window)/window, mode="valid")
            ax.plot(np.arange(len(smoothed)), smoothed, label=layer_key.replace("_", " "), alpha=0.8)

    ax.set_xlabel("Token Position in Sequence")
    ax.set_ylabel("Cosine Similarity (clean vs. ciphered)")
    ax.set_title("Does the Model Recover with More Context? (GPT-2 Small, full shuffle)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "context_length_recovery.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: context_length_recovery.png")

    # Save all results
    save_data = {
        "model": "gpt2",
        "seed": SEED,
        "device": DEVICE,
        "aggregated": aggregated,
        "statistics": stats_results,
        "context_length_results": context_results,
        "execution_time_seconds": time.time() - start_time,
    }

    with open(RESULTS_DIR / "experiment_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'experiment_results.json'}")

    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return save_data


if __name__ == "__main__":
    results = main()
