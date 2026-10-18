"""Small, leakage-aware IASL feasibility utilities.

The pilot deliberately uses numeric, pickle-free EEG exports.  It is an
analysis control, not a production alignment model: subspaces and groups are
the primary unit and individual latent identities are only diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class PilotConfig:
    subjects: tuple[str, ...] = ("sub01", "sub02", "sub03", "sub04", "sub05")
    n_images: int = 50
    n_repetitions: int = 4
    poststimulus: tuple[float, float] = (0.0, 1.0)
    pca_dim: int = 128
    sae_width_multipliers: tuple[int, ...] = (2, 4)
    seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 50
    batch_size: int = 128
    intervention_magnitudes: tuple[float, ...] = (0.25, 0.5, 1.0)


def load_numeric_subject(path: str | Path, n_images: int, n_repetitions: int,
                         times: np.ndarray) -> tuple[np.ndarray, dict]:
    """Load a pickle-free array and return (images, repetitions, channels, time)."""
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 4 or arr.shape[0] < n_images or arr.shape[1] < n_repetitions:
        raise ValueError(f"unexpected EEG shape {arr.shape} for {path}")
    if not np.isfinite(arr[:n_images, :n_repetitions]).all():
        raise ValueError(f"non-finite EEG in {path}")
    t = np.asarray(times, dtype=float)
    mask = (t >= 0) & (t <= 1.0 + 1e-9)
    if mask.sum() < 2:
        raise ValueError("time vector has no usable poststimulus samples")
    x = np.asarray(arr[:n_images, :n_repetitions, :, mask], dtype=np.float32)
    return x, {"source": str(path), "shape": list(arr.shape),
               "channels": int(arr.shape[2]), "samples": int(mask.sum()),
               "time_start": float(t[mask][0]), "time_end": float(t[mask][-1]),
               "time_samples": int(mask.sum()), "poststimulus_complete": bool(t[mask][-1] >= 0.999)}


def repeated_reliability(x: np.ndarray) -> dict:
    """Split repetitions into two halves and correlate image-by-feature means."""
    a = x[:, ::2].mean(axis=1).reshape(x.shape[0], -1)
    b = x[:, 1::2].mean(axis=1).reshape(x.shape[0], -1)
    aa, bb = a - a.mean(0), b - b.mean(0)
    num = np.sum(aa * bb, axis=1)
    den = np.sqrt(np.sum(aa * aa, axis=1) * np.sum(bb * bb, axis=1))
    r = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return {"median_image_split_half_r": float(np.median(r)),
            "mean_image_split_half_r": float(np.mean(r)),
            "fraction_positive": float(np.mean(r > 0)),
            "n_images": int(x.shape[0])}


def flatten_trials(x: np.ndarray) -> np.ndarray:
    """Per-trial standardized flattening; scaling parameters are not fit here."""
    z = x.reshape(x.shape[0] * x.shape[1], -1).astype(np.float32)
    z -= z.mean(axis=1, keepdims=True)
    sd = z.std(axis=1, keepdims=True)
    return z / np.maximum(sd, 1e-6)


def pca_fit_transform(x: np.ndarray, n_components: int = 128):
    """Compact SVD PCA, returning transformed data and fitted parameters."""
    mean = x.mean(axis=0)
    xc = x - mean
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(int(n_components), vt.shape[0])
    comp = vt[:k]
    return xc @ comp.T, {"mean": mean, "components": comp, "singular_values": s[:k]}


def tiny_temporal_encoder(x: np.ndarray, output_dim: int = 128, seed: int = 0) -> np.ndarray:
    """A bounded depthwise temporal-convolution representation control.

    The weights are seeded and frozen for the Phase-0 architecture control;
    they are not a pretrained EEG model and are not used to claim learned
    generalization. Input is (trials, channels, time).
    """
    import torch
    torch.manual_seed(seed); torch.set_num_threads(1)
    inp = torch.tensor(x, dtype=torch.float32)
    channels = x.shape[1]
    depth = torch.nn.Conv1d(channels, channels, kernel_size=5, padding=2,
                            groups=channels, bias=False)
    point = torch.nn.Conv1d(channels, 32, kernel_size=1, bias=True)
    with torch.no_grad():
        y = torch.relu(point(torch.relu(depth(inp))))
        pooled = torch.cat([y.mean(-1), y.std(-1)], dim=1).numpy()
    if pooled.shape[1] > output_dim:
        pooled = pooled[:, :output_dim]
    elif pooled.shape[1] < output_dim:
        pooled = np.pad(pooled, ((0, 0), (0, output_dim - pooled.shape[1])))
    return pooled.astype(np.float32)


def _torch_sae(x: np.ndarray, width: int, seed: int, epochs: int = 50):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    inp = torch.tensor(x, dtype=torch.float32, device=device)
    hidden = x.shape[1]
    enc = torch.nn.Linear(hidden, width, bias=True)
    dec = torch.nn.Linear(width, hidden, bias=True)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=2e-3)
    best, best_state = float("inf"), None
    for _ in range(max(1, int(epochs))):
        opt.zero_grad()
        z = torch.relu(enc(inp))
        rec = dec(z)
        loss = ((rec - inp) ** 2).mean() + 1e-3 * z.abs().mean()
        loss.backward(); opt.step()
        val = float(loss.detach())
        if val < best:
            best = val; best_state = (enc.state_dict(), dec.state_dict())
    if best_state:
        enc.load_state_dict(best_state[0]); dec.load_state_dict(best_state[1])
    with torch.no_grad():
        z = torch.relu(enc(inp)).cpu().numpy()
        rec = dec(torch.tensor(z, dtype=torch.float32)).cpu().numpy()
    # Store one decoder direction per latent (latent, hidden), which is the
    # convention used by matching and intervention code below.
    decoder = dec.weight.detach().cpu().numpy().T
    evr = 1 - np.sum((x - rec) ** 2) / max(np.sum((x - x.mean(0)) ** 2), 1e-12)
    active = z > 1e-6
    return {"codes": z, "reconstruction": rec, "decoder": decoder,
            "loss": float(np.mean((x-rec)**2)), "explained_variance": float(evr),
            "dead_latent_fraction": float(np.mean(active.mean(0) == 0)),
            "activation_rate": float(active.mean()), "seed": seed, "width": width}


def train_sae_replicates(x: np.ndarray, width_multipliers=(2, 4), seeds=(0, 1, 2), epochs=50):
    out = []
    for mult in width_multipliers:
        width = max(2, int(x.shape[1] * mult))
        for seed in seeds:
            result = _torch_sae(x, width, seed, epochs)
            out.append(result)
    return out


def cosine_matching(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    return a @ b.T


def matching_diagnostics(replicates: list[dict], rng_seed: int = 0) -> dict:
    """Report seed matching and low-rank subspace stability."""
    if not replicates:
        return {}
    rng = np.random.default_rng(rng_seed)
    rows = []
    for width in sorted({r["width"] for r in replicates}):
        rs = [r for r in replicates if r["width"] == width]
        sims, overlaps, ckas, angles = [], [], [], []
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                sim = cosine_matching(rs[i]["decoder"], rs[j]["decoder"])
                best = sim.max(axis=1)
                sims.append(float(np.mean(best)))
                overlaps.append(float(np.mean(best > 0.8)))
                # CKA on code activations, invariant to orthogonal rotations.
                a, b = rs[i]["codes"], rs[j]["codes"]
                k1, k2 = a @ a.T, b @ b.T
                ckas.append(float(np.sum(k1*k2) / max(np.linalg.norm(k1)*np.linalg.norm(k2), 1e-12)))
                q1 = np.linalg.qr(rs[i]["decoder"].T)[0]
                q2 = np.linalg.qr(rs[j]["decoder"].T)[0]
                sv = np.linalg.svd(q1.T @ q2, compute_uv=False)
                angles.append(float(np.degrees(np.arccos(np.clip(np.min(sv), -1, 1)))))
        # Null uses independent random decoder directions with identical dimensions.
        null = []
        for _ in range(100):
            a = rng.normal(size=rs[0]["decoder"].shape)
            b = rng.normal(size=rs[0]["decoder"].shape)
            null.append(float(np.mean(cosine_matching(a, b).max(axis=1))))
        # A subspace null permutes sample rows independently before CKA.
        cka_null = []
        for _ in range(100):
            a, b = rs[0]["codes"], rs[1]["codes"]
            bp = b[rng.permutation(len(b))]
            k1, k2 = a @ a.T, bp @ bp.T
            cka_null.append(float(np.sum(k1*k2) / max(np.linalg.norm(k1)*np.linalg.norm(k2), 1e-12)))
        rows.append({"width": width, "n_replicates": len(rs),
                     "feature_cosine_max_mean": float(np.mean(sims)),
                     "feature_overlap_gt_0_8": float(np.mean(overlaps)),
                     "cka_mean": float(np.mean(ckas)),
                     "principal_angle_deg_mean": float(np.mean(angles)),
                     "feature_null_mean": float(np.mean(null)),
                     "feature_null_p95": float(np.percentile(null, 95)),
                     "cka_null_mean": float(np.mean(cka_null)),
                     "cka_null_p95": float(np.percentile(cka_null, 95))})
    return {"widths": rows}


def intervention_audit(rep: dict, magnitudes=(0.25, 0.5, 1.0), n_groups=10, seed=0):
    """Ablate decoder directions and report raw and energy-normalized damage."""
    z, dec, rec = rep["codes"], rep["decoder"], rep["reconstruction"]
    order = np.argsort(np.mean(z, axis=0))[::-1][:max(1, min(n_groups, z.shape[1]))]
    baseline = np.mean((rec - rec.mean(0)) ** 2)
    rows = []
    for j in order:
        latent_energy = float(np.mean(z[:, j] ** 2))
        direction_energy = float(np.mean((z[:, j:j+1] * dec[j:j+1]) ** 2))
        for alpha in magnitudes:
            z2 = z.copy(); z2[:, j] *= (1 - alpha)
            r2 = z2 @ dec
            task_damage = float(np.mean((r2 - rec) ** 2))
            reconstruction_damage = float(np.mean((r2 - rec) ** 2) / max(baseline, 1e-12))
            raw = task_damage + reconstruction_damage
            norm = raw / max(direction_energy * alpha * alpha, 1e-12)
            rows.append({"latent": int(j), "magnitude": float(alpha),
                         "task_damage": task_damage, "reconstruction_damage": reconstruction_damage,
                         "removed_latent_energy": latent_energy * alpha * alpha,
                         "removed_eeg_energy": direction_energy * alpha * alpha,
                         "C_raw": raw, "C_norm": norm})
    return rows


def intervention_permutation_null(rep: dict, magnitudes=(0.25, 0.5, 1.0), n_perm=100, seed=0):
    """Energy-matched null: ablate random latent directions at each magnitude."""
    rng = np.random.default_rng(seed)
    z, dec, rec = rep["codes"], rep["decoder"], rep["reconstruction"]
    baseline = np.mean((rec - rec.mean(0)) ** 2)
    values = []
    for _ in range(n_perm):
        j = int(rng.integers(dec.shape[0])); alpha = float(rng.choice(magnitudes))
        z2 = z.copy(); z2[:, j] *= (1 - alpha); r2 = z2 @ dec
        raw = float(np.mean((r2-rec)**2) + np.mean((r2-rec)**2)/max(baseline,1e-12))
        e = float(np.mean((z[:,j:j+1]*dec[j:j+1])**2) * alpha*alpha)
        values.append(raw/max(e,1e-12))
    return {"n_permutations": n_perm, "median": float(np.median(values)),
            "p95": float(np.percentile(values,95)), "values": values}


def jsonable(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)): return obj.item()
    if isinstance(obj, dict): return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [jsonable(v) for v in obj]
    return obj


def write_json(path: str | Path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(jsonable(obj), indent=2) + "\n")
