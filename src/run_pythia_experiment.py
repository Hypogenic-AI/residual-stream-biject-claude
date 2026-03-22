"""
Pythia-410M experiment: Same analysis as GPT-2 but on a larger model.
Tests whether model size affects recovery from bijective cipher.
Also includes CKA (Centered Kernel Alignment) analysis.
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

sys.path.insert(0, str(Path(__file__).parent))
from bijective_cipher import (
    apply_cipher,
    create_bijective_cipher,
    get_permutation_matrix_in_embedding_space,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


def load_model(model_name="pythia-410m"):
    import transformer_lens
    model = transformer_lens.HookedTransformer.from_pretrained(model_name, device=DEVICE)
    model.eval()
    return model


def get_text_samples(n_samples=30):
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
    while len(texts) < n_samples:
        texts.extend(texts[:n_samples - len(texts)])
    return texts[:n_samples]


def cosine_similarity_per_position(a, b):
    a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-10)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-10)
    return (a_norm * b_norm).sum(dim=-1)


def compute_cka(X, Y):
    """Compute linear CKA between two representation matrices.
    X, Y: (n_samples, features) tensors
    """
    X = X.float()
    Y = Y.float()
    # Center
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Linear CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    XtX = X.T @ X
    YtY = Y.T @ Y
    YtX = Y.T @ X

    cka = (YtX.norm() ** 2) / (XtX.norm() * YtY.norm() + 1e-10)
    return cka.item()


def run_analysis(model, texts, shuffle_rate=1.0):
    """Run full analysis for a given shuffle rate."""
    def compute_token_freq(model, texts):
        freq = np.zeros(model.cfg.d_vocab, dtype=np.float64)
        for text in texts:
            tokens = model.to_tokens(text, prepend_bos=True)[0]
            for t in tokens.cpu().numpy():
                freq[t] += 1
        return freq

    token_freq = compute_token_freq(model, texts)
    forward_map, inverse_map = create_bijective_cipher(
        vocab_size=model.cfg.d_vocab, seed=SEED, shuffle_rate=shuffle_rate,
        frequency_matched=True, token_frequencies=token_freq,
    )

    n_shuffled = (forward_map != torch.arange(model.cfg.d_vocab)).sum().item()
    print(f"  Tokens shuffled: {n_shuffled}/{model.cfg.d_vocab} ({100*n_shuffled/model.cfg.d_vocab:.1f}%)")

    n_layers = model.cfg.n_layers
    all_raw_sims = []
    all_adj_sims = []
    all_rand_sims = []
    all_cka = []

    E = model.W_E.cpu().float()
    T = get_permutation_matrix_in_embedding_space(E, forward_map.cpu())

    for text in tqdm(texts, desc=f"  Processing (r={shuffle_rate})"):
        clean_tokens = model.to_tokens(text, prepend_bos=True)
        ciphered_tokens = apply_cipher(clean_tokens, forward_map.to(clean_tokens.device))
        random_tokens = torch.randint(0, model.cfg.d_vocab, clean_tokens.shape, device=clean_tokens.device)

        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_tokens, remove_batch_dim=False)
            _, cipher_cache = model.run_with_cache(ciphered_tokens, remove_batch_dim=False)
            _, random_cache = model.run_with_cache(random_tokens, remove_batch_dim=False)

        raw_sims = []
        adj_sims = []
        rand_sims = []
        cka_vals = []

        for layer in range(n_layers + 1):
            if layer == 0:
                key = "blocks.0.hook_resid_pre"
            else:
                key = f"blocks.{layer-1}.hook_resid_post"

            clean_h = clean_cache[key][0].cpu()
            cipher_h = cipher_cache[key][0].cpu()
            random_h = random_cache[key][0].cpu()

            # Cosine similarity
            raw_sims.append(cosine_similarity_per_position(clean_h, cipher_h).mean().item())
            rand_sims.append(cosine_similarity_per_position(clean_h, random_h).mean().item())

            # Adjusted
            adjusted = cipher_h.float() @ T.float()
            adj_sims.append(cosine_similarity_per_position(clean_h.float(), adjusted).mean().item())

            # CKA
            cka_vals.append(compute_cka(clean_h, cipher_h))

        all_raw_sims.append(raw_sims)
        all_adj_sims.append(adj_sims)
        all_rand_sims.append(rand_sims)
        all_cka.append(cka_vals)

    return {
        "raw_cos_sim": np.array(all_raw_sims),
        "adjusted_cos_sim": np.array(all_adj_sims),
        "random_cos_sim": np.array(all_rand_sims),
        "cka": np.array(all_cka),
        "n_layers": n_layers + 1,
        "n_shuffled": n_shuffled,
    }


def main():
    start_time = time.time()
    print("=" * 60)
    print("EXPERIMENT: Pythia-410M Residual Stream Analysis")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    model = load_model("pythia-410m")
    texts = get_text_samples(20)

    results = {}
    for rate in [0.3, 0.5, 1.0]:
        print(f"\n--- Shuffle rate: {rate} ---")
        results[f"shuffle_{rate}"] = run_analysis(model, texts, shuffle_rate=rate)

    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for key in sorted(results.keys()):
        data = results[key]
        layers = np.arange(data["n_layers"])
        rate = key.split("_")[1]

        mean_raw = data["raw_cos_sim"].mean(axis=0)
        mean_adj = data["adjusted_cos_sim"].mean(axis=0)
        mean_cka = data["cka"].mean(axis=0)

        axes[0].plot(layers, mean_raw, label=f"r={rate}", marker="o", markersize=2)
        axes[1].plot(layers, mean_adj, label=f"r={rate}", marker="o", markersize=2)
        axes[2].plot(layers, mean_cka, label=f"r={rate}", marker="o", markersize=2)

    mean_rand = results["shuffle_1.0"]["random_cos_sim"].mean(axis=0)
    axes[0].plot(layers, mean_rand, label="random", color="gray", linestyle="--")

    axes[0].set_title("Pythia-410M: Raw Cosine Similarity")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Pythia-410M: Permutation-Adjusted Similarity")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Pythia-410M: CKA (Centered Kernel Alignment)")
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("CKA")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pythia_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: pythia_analysis.png")

    # Save results
    save_data = {}
    for key in results:
        data = results[key]
        save_data[key] = {
            "raw_cos_sim_mean": data["raw_cos_sim"].mean(axis=0).tolist(),
            "raw_cos_sim_std": data["raw_cos_sim"].std(axis=0).tolist(),
            "adjusted_cos_sim_mean": data["adjusted_cos_sim"].mean(axis=0).tolist(),
            "adjusted_cos_sim_std": data["adjusted_cos_sim"].std(axis=0).tolist(),
            "random_cos_sim_mean": data["random_cos_sim"].mean(axis=0).tolist(),
            "cka_mean": data["cka"].mean(axis=0).tolist(),
            "cka_std": data["cka"].std(axis=0).tolist(),
            "n_layers": data["n_layers"],
            "n_shuffled": data["n_shuffled"],
        }

    with open(RESULTS_DIR / "pythia_results.json", "w") as f:
        json.dump(save_data, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
