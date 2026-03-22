"""
Deeper analysis: Why does raw cosine similarity increase at later layers?
Is it genuine cipher decoding or an architectural artifact (LayerNorm convergence)?

Key analyses:
1. Norm analysis: Do hidden states converge in norm at later layers?
2. Direction analysis: Do all inputs converge to similar directions?
3. Probe analysis: Can we linearly predict the cipher mapping from hidden states?
4. Attention pattern analysis: Does the model attend differently under cipher?
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from bijective_cipher import apply_cipher, create_bijective_cipher

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


def get_texts():
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models learn patterns from data through optimization.",
        "The stock market experienced significant volatility this quarter.",
        "Scientists discovered a new species near the Mariana Trench.",
        "Shakespeare wrote many famous plays during the sixteenth century.",
        "Climate change is causing sea levels to rise at an accelerating rate.",
        "The professor explained quantum entanglement to the class.",
        "Neural networks consist of layers of interconnected processing nodes.",
        "Economic theory suggests markets tend toward equilibrium over time.",
        "The ancient Egyptians built pyramids using sophisticated engineering.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "The detective examined the crime scene for fingerprints and clues.",
        "Advances in genomics enable rapid sequencing of entire human genomes.",
        "The orchestra performed Beethoven's Ninth Symphony to a packed hall.",
        "The hiking trail emerged at a spectacular mountain overlook.",
    ]


def analyze_norm_convergence(model, texts, forward_map):
    """Check if hidden state norms converge across layers."""
    import transformer_lens

    n_layers = model.cfg.n_layers

    clean_norms = [[] for _ in range(n_layers + 1)]
    cipher_norms = [[] for _ in range(n_layers + 1)]
    random_norms = [[] for _ in range(n_layers + 1)]

    for text in tqdm(texts, desc="Norm analysis"):
        clean_tokens = model.to_tokens(text, prepend_bos=True)
        cipher_tokens = apply_cipher(clean_tokens, forward_map.to(clean_tokens.device))
        random_tokens = torch.randint(0, model.cfg.d_vocab, clean_tokens.shape, device=clean_tokens.device)

        with torch.no_grad():
            _, cc = model.run_with_cache(clean_tokens, remove_batch_dim=False)
            _, ci = model.run_with_cache(cipher_tokens, remove_batch_dim=False)
            _, cr = model.run_with_cache(random_tokens, remove_batch_dim=False)

        for layer in range(n_layers + 1):
            key = "blocks.0.hook_resid_pre" if layer == 0 else f"blocks.{layer-1}.hook_resid_post"
            clean_norms[layer].append(cc[key][0].norm(dim=-1).mean().cpu().item())
            cipher_norms[layer].append(ci[key][0].norm(dim=-1).mean().cpu().item())
            random_norms[layer].append(cr[key][0].norm(dim=-1).mean().cpu().item())

    return {
        "clean": [np.mean(n) for n in clean_norms],
        "cipher": [np.mean(n) for n in cipher_norms],
        "random": [np.mean(n) for n in random_norms],
    }


def analyze_pairwise_similarity(model, texts, forward_map):
    """Check if DIFFERENT clean texts also converge at later layers.
    If clean_text_A and clean_text_B have high cosine similarity at later layers,
    that's an architecture artifact, not cipher decoding."""

    n_layers = model.cfg.n_layers
    n_texts = len(texts)

    # Get residual streams for all clean texts
    all_residuals = []
    for text in tqdm(texts, desc="Pairwise similarity"):
        tokens = model.to_tokens(text, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, remove_batch_dim=False)

        residuals = []
        for layer in range(n_layers + 1):
            key = "blocks.0.hook_resid_pre" if layer == 0 else f"blocks.{layer-1}.hook_resid_post"
            # Take mean across positions to get a single vector per text per layer
            residuals.append(cache[key][0].mean(dim=0).cpu())
        all_residuals.append(residuals)

    # Compute pairwise cosine similarity for different texts at each layer
    inter_text_sim = []
    for layer in range(n_layers + 1):
        vecs = torch.stack([r[layer] for r in all_residuals])  # (n_texts, d_model)
        vecs_norm = vecs / (vecs.norm(dim=-1, keepdim=True) + 1e-10)
        sim_matrix = vecs_norm @ vecs_norm.T  # (n_texts, n_texts)
        # Take off-diagonal mean
        mask = ~torch.eye(n_texts, dtype=torch.bool)
        inter_text_sim.append(sim_matrix[mask].mean().item())

    return inter_text_sim


def analyze_cipher_probe(model, texts, forward_map, inverse_map):
    """Train a linear probe to see if the residual stream encodes the cipher.
    If it does, we can predict the original token from the ciphered hidden state."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    n_layers = model.cfg.n_layers

    # Collect (hidden_state, original_token_id) pairs
    probe_accuracies = []

    for layer in range(0, n_layers + 1, max(1, n_layers // 6)):
        X_train = []
        y_train = []

        for text in texts[:10]:  # Use subset for training
            clean_tokens = model.to_tokens(text, prepend_bos=True)
            cipher_tokens = apply_cipher(clean_tokens, forward_map.to(clean_tokens.device))

            with torch.no_grad():
                _, cipher_cache = model.run_with_cache(cipher_tokens, remove_batch_dim=False)

            key = "blocks.0.hook_resid_pre" if layer == 0 else f"blocks.{layer-1}.hook_resid_post"
            hidden = cipher_cache[key][0].cpu().numpy()  # (seq_len, d_model)
            original_ids = clean_tokens[0].cpu().numpy()  # (seq_len,)

            X_train.append(hidden)
            y_train.append(original_ids)

        X = np.concatenate(X_train, axis=0)
        y = np.concatenate(y_train, axis=0)

        # Limit classes to most common tokens for tractability
        unique_tokens, counts = np.unique(y, return_counts=True)
        top_tokens = unique_tokens[counts >= 3]

        if len(top_tokens) < 5:
            probe_accuracies.append({"layer": layer, "accuracy": 0.0, "n_classes": 0})
            continue

        mask = np.isin(y, top_tokens)
        X_filtered = X[mask]
        y_filtered = y[mask]

        if len(X_filtered) < 20:
            probe_accuracies.append({"layer": layer, "accuracy": 0.0, "n_classes": 0})
            continue

        # Simple probe
        try:
            clf = LogisticRegression(max_iter=500, C=1.0, random_state=SEED, solver='lbfgs')
            # Use cross-val style: train on 80%, test on 20%
            n = len(X_filtered)
            split = int(0.8 * n)
            clf.fit(X_filtered[:split], y_filtered[:split])
            y_pred = clf.predict(X_filtered[split:])
            acc = accuracy_score(y_filtered[split:], y_pred)
        except Exception:
            acc = 0.0

        probe_accuracies.append({
            "layer": layer,
            "accuracy": acc,
            "n_classes": len(top_tokens),
            "n_samples": len(X_filtered),
        })
        print(f"  Layer {layer}: probe accuracy = {acc:.3f} ({len(top_tokens)} classes, {len(X_filtered)} samples)")

    return probe_accuracies


def analyze_attention_entropy(model, texts, forward_map):
    """Compare attention pattern entropy between clean and ciphered inputs."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    clean_entropies = [[] for _ in range(n_layers)]
    cipher_entropies = [[] for _ in range(n_layers)]

    for text in tqdm(texts[:10], desc="Attention entropy"):
        clean_tokens = model.to_tokens(text, prepend_bos=True)
        cipher_tokens = apply_cipher(clean_tokens, forward_map.to(clean_tokens.device))

        with torch.no_grad():
            _, cc = model.run_with_cache(clean_tokens, remove_batch_dim=False)
            _, ci = model.run_with_cache(cipher_tokens, remove_batch_dim=False)

        for layer in range(n_layers):
            # Attention patterns: (batch, head, q_pos, k_pos)
            clean_attn = cc[f"blocks.{layer}.attn.hook_pattern"][0]  # (head, q, k)
            cipher_attn = ci[f"blocks.{layer}.attn.hook_pattern"][0]

            # Entropy per head per query position
            clean_ent = -(clean_attn * (clean_attn + 1e-10).log()).sum(dim=-1).mean().cpu().item()
            cipher_ent = -(cipher_attn * (cipher_attn + 1e-10).log()).sum(dim=-1).mean().cpu().item()

            clean_entropies[layer].append(clean_ent)
            cipher_entropies[layer].append(cipher_ent)

    return {
        "clean_entropy": [np.mean(e) for e in clean_entropies],
        "cipher_entropy": [np.mean(e) for e in cipher_entropies],
    }


def main():
    import transformer_lens

    print("=" * 60)
    print("DEEPER ANALYSIS: Architecture artifact vs. cipher decoding")
    print("=" * 60)

    model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()
    texts = get_texts()

    # Create cipher
    token_freq = np.zeros(model.cfg.d_vocab)
    for text in texts:
        for t in model.to_tokens(text, prepend_bos=True)[0].cpu().numpy():
            token_freq[t] += 1

    forward_map, inverse_map = create_bijective_cipher(
        vocab_size=model.cfg.d_vocab, seed=SEED, shuffle_rate=1.0,
        frequency_matched=True, token_frequencies=token_freq,
    )

    # 1. Norm convergence
    print("\n--- 1. Norm Convergence Analysis ---")
    norms = analyze_norm_convergence(model, texts, forward_map)

    # 2. Pairwise similarity (architecture convergence)
    print("\n--- 2. Pairwise Similarity (Architecture Convergence) ---")
    inter_text_sim = analyze_pairwise_similarity(model, texts, forward_map)
    for i in [0, 6, 12]:
        print(f"  Layer {i}: inter-text cosine sim = {inter_text_sim[i]:.4f}")

    # 3. Cipher probe
    print("\n--- 3. Cipher Probe Analysis ---")
    probe_results = analyze_cipher_probe(model, texts, forward_map, inverse_map)

    # 4. Attention entropy
    print("\n--- 4. Attention Entropy Analysis ---")
    attn_results = analyze_attention_entropy(model, texts, forward_map)

    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Norm convergence
    layers = np.arange(len(norms["clean"]))
    axes[0, 0].plot(layers, norms["clean"], label="Clean", marker="o", markersize=3)
    axes[0, 0].plot(layers, norms["cipher"], label="Cipher", marker="s", markersize=3)
    axes[0, 0].plot(layers, norms["random"], label="Random", marker="^", markersize=3, color="gray")
    axes[0, 0].set_title("Hidden State Norms Across Layers")
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].set_ylabel("Mean L2 Norm")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Inter-text similarity
    axes[0, 1].plot(layers, inter_text_sim, marker="o", markersize=3, color="purple")
    axes[0, 1].set_title("Mean Cosine Similarity Between Different Clean Texts")
    axes[0, 1].set_xlabel("Layer")
    axes[0, 1].set_ylabel("Cosine Similarity")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Plot 3: Probe accuracy
    probe_layers = [p["layer"] for p in probe_results if p["accuracy"] > 0]
    probe_accs = [p["accuracy"] for p in probe_results if p["accuracy"] > 0]
    if probe_layers:
        axes[1, 0].bar(probe_layers, probe_accs, width=1.5, color="teal")
    axes[1, 0].set_title("Linear Probe: Can We Decode Original Tokens?")
    axes[1, 0].set_xlabel("Layer")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Attention entropy
    attn_layers = np.arange(len(attn_results["clean_entropy"]))
    axes[1, 1].plot(attn_layers, attn_results["clean_entropy"], label="Clean", marker="o", markersize=3)
    axes[1, 1].plot(attn_layers, attn_results["cipher_entropy"], label="Cipher", marker="s", markersize=3)
    axes[1, 1].set_title("Attention Pattern Entropy")
    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].set_ylabel("Mean Entropy (nats)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "deeper_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: deeper_analysis.png")

    # Save results
    results = {
        "norms": norms,
        "inter_text_similarity": inter_text_sim,
        "probe_results": probe_results,
        "attention_entropy": attn_results,
    }
    with open(RESULTS_DIR / "deeper_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: deeper_analysis.json")


if __name__ == "__main__":
    main()
