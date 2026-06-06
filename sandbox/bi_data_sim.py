import numpy as np
from bi_data_struct import Recording

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
                x[:, :, t - 1]
                - theta * x[:, :, t - 1]
                + sigma * rng.standard_normal((P, D))
            )
        return x

    for i in range(n_null):
        data = ou_process(time_len, depth, num_params, rng)
        recordings.append(Recording(data, None, f"null_{i:04d}"))

    for i in range(n_positive):
        # Anchor the window centre, keeping it safely inside [20%, 80%]
        center_bf_t = int(rng.uniform(0.2 * time_len, 0.8 * time_len))
        half_window = int(0.05 * time_len)  # ±5% → 10% total spread

        # One bifurcation point per parameter, clipped to valid range
        bf_ts = np.clip(
            (
                center_bf_t
                + rng.integers(-half_window, half_window + 1, size=num_params)
            ),
            1,
            time_len - 2,
        )

        data = ou_process(time_len, depth, num_params, rng)

        # Critical slowing down applied per-parameter from its own bf_t
        for p, bf_t_p in enumerate(bf_ts):
            t_range = np.arange(bf_t_p, time_len)
            scales = 1.0 + 3.0 * (t_range - bf_t_p) / (time_len - bf_t_p)  # (T-bf,)
            data[p, :, bf_t_p:] *= scales[np.newaxis, :]  # broadcast over depth
            data[p, :, bf_t_p:] += 0.5 * rng.standard_normal((depth, len(t_range)))

        recordings.append(Recording(
                            data              = data,
                            bifurcation_t     = center_bf_t,
                            recording_id      = f"pos_{i:04d}",
                            param_bifurcation_ts = bf_ts,       # list[int], length P, already computed above
                        ))

    return recordings
