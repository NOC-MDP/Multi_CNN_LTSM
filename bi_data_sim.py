from bi_data_struct import Recording
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data generator
# ──────────────────────────────────────────────────────────────────────────────

def make_synthetic_recordings(
    n_null: int = 40,
    n_positive: int = 20,
    num_params: int = 4,
    depth: int = 32,
    time_len: int = 512,
    seed: int = 0,
) -> list[Recording]:
    """
    Generates synthetic recordings where:
      • null    : Ornstein-Uhlenbeck-like stable fluctuations
      • positive: variance increases post-bifurcation (critical slowing down),
                  bifurcation occurs at a random timestep in [20%, 80%] of T
    """
    rng = np.random.default_rng(seed)
    recordings = []

    def ou_process(T, D, P, rng, theta=0.1, sigma=0.3):
        x = np.zeros((P, D, T))
        for t in range(1, T):
            x[:, :, t] = (
                x[:, :, t-1]
                - theta * x[:, :, t-1]
                + sigma * rng.standard_normal((P, D))
            )
        return x

    for i in range(n_null):
        data = ou_process(time_len, depth, num_params, rng)
        recordings.append(Recording(data, None, f"null_{i:04d}"))

    for i in range(n_positive):
        bf_t = int(rng.uniform(0.2 * time_len, 0.8 * time_len))
        data = ou_process(time_len, depth, num_params, rng)

        # Critical slowing down: growing variance after bifurcation
        for t in range(bf_t, time_len):
            scale = 1.0 + 3.0 * (t - bf_t) / (time_len - bf_t)
            data[:, :, t] *= scale
            data[:, :, t] += 0.5 * rng.standard_normal((num_params, depth))

        recordings.append(Recording(data, bf_t, f"pos_{i:04d}"))

    return recordings

