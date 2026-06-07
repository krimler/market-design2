#!/usr/bin/env python3
"""Protocol C: the ternary {accept, reject, abstain} ablation.

Extends extract.py: reuses its cached linear victims, per-cell seeds, 20-seed CI
aggregation, and surrogate/fidelity machinery. The victim returns the truthful
label except within distance delta of its boundary, where it abstains. Two
queriers are compared: an adaptive coordinate-wise binary search that uses the
abstain as a localization witness, and a random querier that feeds abstains to
its surrogate as near-boundary constraints. Writes out/ternary_c_results.json,
consumed by plot_ternary_ablation.py.

Assumes a linear victim, so signed distance is (w.x + b)/||w|| in closed form,
matching extract.py and gaming_experiment.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import extract
from extract import (
    CACHE,
    DATASETS,
    DATASET_LABEL,
    L_VALUES,
    N_SEEDS,
    OUT,
    aggregate,
    load_checkpoint,
    load_dataset,
    log,
    save_checkpoint,
    victim_predict,
    _eval_error,
    _sample_indices,
    _seed_for,
)

DELTA_Q = 0.05                                  # abstain band ~ q-quantile of |dist|
DELTA_Q_SWEEP = [0.02, 0.05, 0.10, 0.20, 0.40]
ABSTAIN_W = 0.25                                # weight of an abstain constraint vs a label

COLOR_C = "#9467bd"
MARK_C = "D"
COLOR_C_RANDOM = "#8c564b"
MARK_C_RANDOM = "v"

TEXT_TASK = "trec-num"
DELTA_SWEEP_DATASET = "acs-income"


def signed_distance(w: np.ndarray, b: float, X: np.ndarray) -> np.ndarray:
    return (X @ w + b) / float(np.linalg.norm(w))


def delta_for(payload: dict, q: float) -> float:
    w, b = payload["victim_w"], payload["victim_b"]
    dists = np.abs(signed_distance(w, b, payload["X_test"]))
    return float(np.quantile(dists, q))


def ternary_signal(w, b, X, delta):
    s = signed_distance(w, b, X)
    return (s > 0).astype(int), np.abs(s) < delta


KMAX = 40  # bisection cap per probe line


def run_adaptive_C(payload: dict, L: int, seed: int, delta: float) -> float:
    """Coordinate-wise binary search: open probe lines (rejected<->accepted
    segments) until the budget is spent, bisecting each on the accept/reject
    threshold and retiring it on an abstain. Fit the surrogate to the queried
    near-boundary responses."""
    w, b = payload["victim_w"], payload["victim_b"]
    X_pool = payload["X_test"]
    rng = np.random.default_rng(_seed_for("ternaryC_adaptive", seed))

    labels_pool = victim_predict(w, b, X_pool)
    pos_idx = np.where(labels_pool == 1)[0]
    neg_idx = np.where(labels_pool == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return 1.0 - float((payload["victim_test_pred"] == 0).mean())

    QX, QY = [], []
    used = 0
    while used < L:
        a_lo = X_pool[neg_idx[rng.integers(len(neg_idx))]].copy()
        a_hi = X_pool[pos_idx[rng.integers(len(pos_idx))]].copy()
        QX.append(a_lo); QY.append(0)
        QX.append(a_hi); QY.append(1)
        used += 2                                   # endpoint labels count as queries
        for _ in range(KMAX):
            if used >= L:
                break
            mid = 0.5 * (a_lo + a_hi)
            s = float(signed_distance(w, b, mid[None, :])[0])
            used += 1
            if abs(s) < delta:                      # abstain witness retires the line
                QX.append(mid); QY.append(0)
                QX.append(mid); QY.append(1)
                break
            if s > 0:
                a_hi = mid; QX.append(mid); QY.append(1)
            else:
                a_lo = mid; QX.append(mid); QY.append(0)

    X_train = np.vstack(QX)
    y_train = np.asarray(QY, dtype=int)
    if len(np.unique(y_train)) < 2:
        return 1.0 - float((payload["victim_test_pred"] == 0).mean())
    return _eval_error(payload, X_train, y_train, seed)


def run_random_C(payload: dict, L: int, seed: int, delta: float) -> float:
    """Random ternary queries; abstains are fed to the surrogate as both-label
    boundary constraints (weight ABSTAIN_W), non-abstains as labels."""
    w, b = payload["victim_w"], payload["victim_b"]
    X_pool = payload["X_test"]
    rng = np.random.default_rng(_seed_for("ternaryC_random", seed))
    idx = np.unique(_sample_indices(rng, X_pool.shape[0], L))  # distinct queries only
    X_q = X_pool[idx]
    labels, abstain = ternary_signal(w, b, X_q, delta)

    keep = ~abstain
    X_lab, y_lab = X_q[keep], labels[keep]
    X_ab = X_q[abstain]
    if X_lab.shape[0] == 0 or len(np.unique(y_lab)) < 2:
        return 1.0 - float((payload["victim_test_pred"] == 0).mean())
    if X_ab.shape[0] > 0:
        X_train = np.vstack([X_lab, X_ab, X_ab])
        y_train = np.concatenate([y_lab,
                                  np.zeros(X_ab.shape[0], int),
                                  np.ones(X_ab.shape[0], int)])
        sw = np.concatenate([np.ones(X_lab.shape[0]),
                             np.full(2 * X_ab.shape[0], ABSTAIN_W)])
    else:
        X_train, y_train, sw = X_lab, y_lab, None
    sur = LogisticRegression(C=extract.LR_C, max_iter=extract.LR_MAX_ITER,
                             random_state=seed)
    sur.fit(X_train, y_train, sample_weight=sw)
    pred = sur.predict(payload["X_test"])
    return 1.0 - float((pred == payload["victim_test_pred"]).mean())


def _cells(state_key: str, runner, payloads, L_list, delta_map, save_state):
    state = save_state
    store = state.setdefault(state_key, {})
    last_save = time.time()
    new = 0
    for d in payloads:
        delta = delta_map[d]
        for L in L_list:
            for s in range(N_SEEDS):
                key = f"{d}|{L}|{s}"
                if key in store:
                    continue
                store[key] = float(runner(payloads[d], L, s, delta))
                new += 1
                if new >= 25 or time.time() - last_save > 30:
                    save_checkpoint(state); last_save = time.time(); new = 0
    save_checkpoint(state)


def _agg_curve(store, d, L_list):
    means, cis = [], []
    for L in L_list:
        vals = [store[f"{d}|{L}|{s}"] for s in range(N_SEEDS)]
        m, c = aggregate(vals)
        means.append(m); cis.append(c)
    return np.array(means), np.array(cis)


def _fit_exponential(L_list, means, lo=1.5e-3, hi=0.30):
    """Fit log(1-F) = c - beta L on the decay window 1-F in (lo, hi]."""
    pts = [(L, np.log(m)) for L, m in zip(L_list, means) if lo < m <= hi]
    if len(pts) < 2:
        return float("nan"), float("nan"), len(pts)
    from scipy import stats as spstats
    xs, ys = zip(*pts)
    res = spstats.linregress(xs, ys)
    return -float(res.slope), float(res.rvalue) ** 2, len(pts)


DATASETS_C = [DELTA_SWEEP_DATASET, TEXT_TASK]  # figure datasets; --all sweeps every dataset


def run_AB(payload, p, L, seed):
    return extract.run_extraction(payload, p, L, seed)


def main(datasets=None):
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    state = load_checkpoint()
    ds_list = datasets or DATASETS_C

    payloads = {d: load_dataset(d) for d in ds_list}
    delta_map = {d: delta_for(payloads[d], DELTA_Q) for d in ds_list}
    log("Protocol C delta per dataset (q=%.2f): %s"
        % (DELTA_Q, {d: round(delta_map[d], 5) for d in ds_list}))

    log("Protocol C adaptive sweep ...")
    _cells("ternaryC_adaptive", run_adaptive_C, payloads, L_VALUES, delta_map, state)
    log("Protocol C random sweep ...")
    _cells("ternaryC_random", run_random_C, payloads, L_VALUES, delta_map, state)
    log("Protocol A reference sweep ...")
    _cells("ternaryC_A", lambda pl, L, s, _d: run_AB(pl, "A", L, s),
           payloads, L_VALUES, delta_map, state)
    log("Protocol B reference sweep ...")
    _cells("ternaryC_B", lambda pl, L, s, _d: run_AB(pl, "B", L, s),
           payloads, L_VALUES, delta_map, state)

    log("Protocol C delta-sensitivity sweep on %s ..." % DELTA_SWEEP_DATASET)
    dsd = DELTA_SWEEP_DATASET
    dsweep = state.setdefault("ternaryC_dsweep", {})
    for q in DELTA_Q_SWEEP:
        delta_q = delta_for(payloads[dsd], q)
        for L in L_VALUES:
            for s in range(N_SEEDS):
                key = f"{q:.4f}|{L}|{s}"
                if key in dsweep:
                    continue
                dsweep[key] = float(run_adaptive_C(payloads[dsd], L, s, delta_q))
    save_checkpoint(state)

    analyze(state, payloads, delta_map, ds_list)


def analyze(state, payloads, delta_map, ds_list):
    adC, rdC = state["ternaryC_adaptive"], state["ternaryC_random"]
    Aref, Bref = state.get("ternaryC_A", {}), state.get("ternaryC_B", {})

    results = {
        "L_values": L_VALUES, "n_seeds": N_SEEDS,
        "delta_q": DELTA_Q, "abstain_w": ABSTAIN_W,
        "delta": {d: delta_map[d] for d in ds_list},
        "labels": {d: DATASET_LABEL[d] for d in ds_list},
        "dims": {d: payloads[d]["n_features"] for d in ds_list},
        "text_task": TEXT_TASK, "delta_sweep_dataset": DELTA_SWEEP_DATASET,
        "adaptive": {}, "random": {}, "protocolA": {}, "protocolB": {},
        "fits": {}, "delta_sweep": {}, "rate_vs_dim": {},
    }

    for d in ds_list:
        ma, ca = _agg_curve(adC, d, L_VALUES)
        mr, cr = _agg_curve(rdC, d, L_VALUES)
        results["adaptive"][d] = {"mean": ma.tolist(), "ci": ca.tolist()}
        results["random"][d] = {"mean": mr.tolist(), "ci": cr.tolist()}
        if Aref:
            results["protocolA"][d] = _agg_curve(Aref, d, L_VALUES)[0].tolist()
            results["protocolB"][d] = _agg_curve(Bref, d, L_VALUES)[0].tolist()
        beta, r2, n = _fit_exponential(L_VALUES, ma)
        results["fits"][d] = {"beta": beta, "r2": r2, "n_points": n}

    dsweep, dsd = state.get("ternaryC_dsweep", {}), DELTA_SWEEP_DATASET
    for q in DELTA_Q_SWEEP:
        means = [aggregate([dsweep[f"{q:.4f}|{L}|{s}"] for s in range(N_SEEDS)])[0]
                 for L in L_VALUES]
        beta, r2, _ = _fit_exponential(L_VALUES, np.array(means))
        results["delta_sweep"][f"{q:.4f}"] = {
            "delta": delta_for(payloads[dsd], q), "q": q,
            "beta": beta, "r2": r2, "mean": means}

    for d in ds_list:
        ma = np.array(results["adaptive"][d]["mean"])
        L_star = next((L for L, m in zip(L_VALUES, ma) if m <= 0.02), None)
        results["rate_vs_dim"][d] = {"dim": payloads[d]["n_features"],
                                     "L_star": L_star}

    (OUT / "ternary_c_results.json").write_text(
        json.dumps(results, indent=2, default=float))
    log("wrote ternary_c_results.json")

    print("\n=== PROTOCOL C ADAPTIVE FITS (1-F = e^{-beta L}, window 1-F in (1.5e-3, 0.3]) ===")
    for d in ds_list:
        f = results["fits"][d]
        print(f"  {DATASET_LABEL[d]:40s} d={payloads[d]['n_features']:3d}  "
              f"beta={f['beta']:.3e}  R^2={f['r2']:.3f}  (n={f['n_points']})")
    print("\n=== ORDERING (1-F by L: A / random-C / B / adaptive-C) ===")
    for d in ds_list:
        if not Aref:
            continue
        print(f"  {DATASET_LABEL[d]} (d={payloads[d]['n_features']}, delta={delta_map[d]:.4f}):")
        for i, L in enumerate(L_VALUES):
            a = results["protocolA"][d][i]; b = results["protocolB"][d][i]
            c = results["random"][d]["mean"][i]; ad = results["adaptive"][d]["mean"][i]
            flag = "  A>=Crand>=B" if (a + 2e-3 >= c >= b - 2e-3) else "  (tie/floor)"
            print(f"    L={L:5d}  A={a:.4f}  Crand={c:.4f}  B={b:.4f}  Cadapt={ad:.4f}{flag}")
    print("\n=== DELTA SENSITIVITY (adaptive beta vs delta) on %s ===" % dsd)
    for q in DELTA_Q_SWEEP:
        e = results["delta_sweep"][f"{q:.4f}"]
        print(f"  q={q:.2f}  delta={e['delta']:.4f}  beta={e['beta']:.3e}  R^2={e['r2']:.3f}")
    print("\n=== RATE VS DIMENSION (adaptive queries to 1-F<=0.02) ===")
    for d in ds_list:
        rv = results["rate_vs_dim"][d]
        print(f"  {DATASET_LABEL[d]:40s} d={rv['dim']:3d}  L*={rv['L_star']}")
    return results


if __name__ == "__main__":
    import sys
    main(DATASETS if "--all" in sys.argv else None)
