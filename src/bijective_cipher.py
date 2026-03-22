"""
Bijective token cipher implementation.
Creates a permutation f: V -> V that shuffles token IDs while preserving
frequency distribution (Zipfian matching).
"""

import torch
import numpy as np
from typing import Optional


def create_bijective_cipher(
    vocab_size: int,
    seed: int = 42,
    shuffle_rate: float = 1.0,
    frequency_matched: bool = True,
    token_frequencies: Optional[np.ndarray] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a bijective token permutation f: V -> V.

    Args:
        vocab_size: Size of vocabulary
        seed: Random seed for reproducibility
        shuffle_rate: Fraction of tokens to shuffle (0=identity, 1=full shuffle)
        frequency_matched: If True, shuffle within frequency bins to preserve Zipfian distribution
        token_frequencies: Token frequency counts for frequency matching

    Returns:
        (forward_map, inverse_map): Tensors mapping token_id -> permuted_id and back
    """
    rng = np.random.RandomState(seed)
    forward_map = np.arange(vocab_size)

    if shuffle_rate == 0:
        return torch.tensor(forward_map), torch.tensor(forward_map)

    if frequency_matched and token_frequencies is not None:
        # Bin tokens by log-frequency to preserve Zipfian distribution
        log_freq = np.log1p(token_frequencies)
        n_bins = 50
        bin_edges = np.percentile(log_freq[log_freq > 0], np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(log_freq, bin_edges)

        for bin_idx in range(n_bins + 2):
            tokens_in_bin = np.where(bin_indices == bin_idx)[0]
            if len(tokens_in_bin) < 2:
                continue
            n_to_shuffle = max(2, int(len(tokens_in_bin) * shuffle_rate))
            selected = rng.choice(tokens_in_bin, size=min(n_to_shuffle, len(tokens_in_bin)), replace=False)
            shuffled = selected.copy()
            rng.shuffle(shuffled)
            # Ensure it's a derangement (no fixed points) for selected tokens
            for attempt in range(100):
                if not np.any(selected == shuffled):
                    break
                rng.shuffle(shuffled)
            forward_map[selected] = shuffled
    else:
        # Simple random permutation of a fraction of tokens
        n_to_shuffle = max(2, int(vocab_size * shuffle_rate))
        selected = rng.choice(vocab_size, size=n_to_shuffle, replace=False)
        shuffled = selected.copy()
        rng.shuffle(shuffled)
        for attempt in range(100):
            if not np.any(selected == shuffled):
                break
            rng.shuffle(shuffled)
        forward_map[selected] = shuffled

    # Build inverse map
    inverse_map = np.zeros(vocab_size, dtype=np.int64)
    inverse_map[forward_map] = np.arange(vocab_size)

    # Verify bijectivity
    assert len(set(forward_map)) == vocab_size, "Forward map is not bijective!"
    assert np.all(inverse_map[forward_map] == np.arange(vocab_size)), "Inverse is wrong!"

    return torch.tensor(forward_map, dtype=torch.long), torch.tensor(inverse_map, dtype=torch.long)


def apply_cipher(token_ids: torch.Tensor, cipher_map: torch.Tensor) -> torch.Tensor:
    """Apply bijective cipher to token IDs."""
    return cipher_map[token_ids]


def get_permutation_matrix_in_embedding_space(
    embedding_matrix: torch.Tensor,  # (vocab_size, d_model)
    forward_map: torch.Tensor,       # (vocab_size,)
) -> torch.Tensor:
    """
    Compute the permutation's effect in embedding space.

    If E is the embedding matrix, the cipher permutes rows: E_cipher = P @ E
    where P is the permutation matrix. The effect on a hidden state h that
    was produced from ciphered input can be "undone" by applying E^{-1} P^{-1} E
    (approximately, via pseudo-inverse).

    Returns the transformation matrix that maps ciphered embeddings back to clean space.
    """
    vocab_size, d_model = embedding_matrix.shape

    # The permutation matrix P such that E[forward_map] = P @ E
    # P[i, forward_map[i]] = 1 for all i
    # So P^{-1}[forward_map[i], i] = 1, i.e. P^{-1} = P^T (since P is a permutation)

    # In embedding space, the cipher maps embedding[t] -> embedding[f(t)]
    # The "inverse" in embedding space: E_pinv @ P^{-1} @ E
    # But this only works at the embedding layer. For deeper layers, the
    # relationship is more complex.

    E = embedding_matrix.float()
    # Permuted embedding: E_perm[i] = E[forward_map[i]]
    # We want T such that T @ E_perm[i] ≈ E[i]
    # i.e., T @ E[f(i)] ≈ E[i] for all i

    # This is: T ≈ E @ (E[forward_map])^+  (pseudo-inverse)
    E_perm = E[forward_map]  # (vocab, d_model)

    # T = E @ pinv(E_perm) -- maps from permuted embedding space to clean
    T = E @ torch.linalg.pinv(E_perm)  # (d_model, d_model) -- wait, shapes wrong

    # Actually: We want T (d_model x d_model) such that T @ E_perm^T ≈ E^T
    # So T ≈ E^T @ pinv(E_perm^T) = E^T @ (E_perm^T)^+
    # But simpler: T = E^T @ pinv(E_perm^T)
    # Or: for each embedding dim, solve T @ e_perm_i = e_clean_i

    # Least squares: T = E^T @ pinv(E_perm^T)
    # E^T is (d_model, vocab), E_perm^T is (d_model, vocab)
    # pinv(E_perm^T) is (vocab, d_model)
    # So T is (d_model, d_model)
    T = torch.linalg.lstsq(E_perm, E).solution  # (d_model, d_model)
    # E_perm @ T ≈ E  =>  for hidden state h from ciphered input, T @ h ≈ clean h

    return T
