from __future__ import annotations

from typing import Any

import numpy as np

from planning.kinematics import wrap_np_pi as _wrap_np_pi


def _local_neighbors(x: np.ndarray, idx: int, k: int, periodic_joint: bool) -> np.ndarray:
    k = int(max(2, min(k, len(x) - 1)))
    center = x[idx]
    if periodic_joint:
        diff = _wrap_np_pi(x - center.reshape(1, -1))
        d2 = np.sum(diff * diff, axis=1)
        order = np.argsort(d2)
        return order[1 : k + 1].astype(np.int64)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(x.astype(np.float64))
        _, nbr = tree.query(center.astype(np.float64), k=k + 1)
        nbr = np.asarray(nbr, dtype=np.int64).reshape(-1)
        nbr = nbr[nbr != idx]
        if len(nbr) >= k:
            return nbr[:k]
    except Exception:
        pass
    d2 = np.sum((x - center.reshape(1, -1)) ** 2, axis=1)
    order = np.argsort(d2)
    return order[1 : k + 1].astype(np.int64)


def estimate_codim_local_pca(
    x: np.ndarray,
    *,
    sample_ratio: float = 0.2,
    k_neighbors: int = 0,
    periodic_joint: bool = False,
    seed: int = 0,
    const_axis_std_ratio: float = 0.0,
) -> dict[str, Any]:
    n, d = x.shape

    std = np.std(x.astype(np.float64), axis=0)
    std_ref = max(float(np.max(std)), 1e-12)
    std_tol = max(1e-10, float(const_axis_std_ratio) * std_ref)
    keep_mask = std > std_tol
    n_const = int(np.sum(~keep_mask))
    x_eff = x[:, keep_mask] if np.any(keep_mask) else x
    d_eff = int(x_eff.shape[1])
    n_sample = int(max(16, min(n, round(float(sample_ratio) * n))))
    if int(k_neighbors) > 0:
        k = int(k_neighbors)
    else:
        k = int(np.clip(round(np.sqrt(n)), max(3, d_eff + 2), min(96, n - 1)))

    if d_eff < 2:
        est = int(np.clip(max(1, n_const), 1, max(1, d - 1)))
        return {
            "n": int(n),
            "d": int(d),
            "sample_ratio": float(sample_ratio),
            "n_sample": int(n_sample),
            "k_neighbors": int(max(2, min(k, max(2, n - 1)))),
            "estimated_codim": est,
            "mode_count": int(n_sample),
            "mode_fraction": 1.0,
            "codim_histogram": {str(est): int(n_sample)},
            "n_const_axes": n_const,
            "d_eff": d_eff,
            "std_tol": float(std_tol),
        }

    rng = np.random.default_rng(int(seed))
    sample_idx = rng.choice(n, size=n_sample, replace=False)
    codims = np.zeros((n_sample,), dtype=np.int64)
    eps = 1e-10
    for i, idx in enumerate(sample_idx):
        nbr_idx = _local_neighbors(x_eff, int(idx), k=k, periodic_joint=periodic_joint)
        center = x_eff[int(idx)]
        neigh = x_eff[nbr_idx]
        if periodic_joint:
            neigh = center.reshape(1, -1) + _wrap_np_pi(neigh - center.reshape(1, -1))
        mu = np.mean(neigh, axis=0, keepdims=True)
        xc = neigh - mu
        cov = (xc.T @ xc) / max(1, len(neigh))
        evals = np.linalg.eigvalsh(cov.astype(np.float64))
        evals = np.maximum(evals, eps)
        log_ratio = np.log(evals[1:] + eps) - np.log(evals[:-1] + eps)
        codim_i = int(np.argmax(log_ratio) + 1)
        codim_eff = int(np.clip(codim_i, 1, d_eff - 1))
        codims[i] = int(np.clip(codim_eff + n_const, 1, d - 1))

    counts = np.bincount(codims, minlength=d + 1)
    est = int(np.argmax(counts[1:]) + 1)
    mode_count = int(counts[est])
    hist = {str(i): int(counts[i]) for i in range(1, d) if counts[i] > 0}
    return {
        "n": int(n),
        "d": int(d),
        "sample_ratio": float(sample_ratio),
        "n_sample": int(n_sample),
        "k_neighbors": int(k),
        "estimated_codim": est,
        "mode_count": mode_count,
        "mode_fraction": float(mode_count / max(1, n_sample)),
        "codim_histogram": hist,
        "n_const_axes": n_const,
        "d_eff": d_eff,
        "std_tol": float(std_tol),
    }
