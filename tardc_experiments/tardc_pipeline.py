#!/usr/bin/env python3
"""Reproducible reduced-order TARDC experiment pipeline.

The electrical parameters and event sequence for System A are taken from
Li et al. (IEEE/CAA JAS, 2016).  The original PSCAD implementation is not
available; consequently this code uses an explicitly documented aggregate
frequency/reactive-voltage model and must not be described as an EMT replica.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


CONTROLLERS = ["B0", "B1", "B2", "B3", "B4", "B5", "P"]
SCENARIOS = [f"S{i}" for i in range(8)]


@dataclass(frozen=True)
class Config:
    dt: float = 0.02
    control_period: float = 0.2
    duration: float = 20.0
    f0: float = 50.0
    v0: float = 400.0
    freq_inertia: float = 400.0
    freq_damping: float = 28.0
    voltage_inertia: float = 5.0
    voltage_damping: float = 4.0
    power_tau: float = 0.10
    reactive_tau: float = 0.25
    ref_ramp_kw_s: float = 35.0
    qref_ramp_kvar_s: float = 30.0
    secondary_gain_p: float = 0.75
    secondary_gain_q: float = 0.70
    h_min: int = 1
    h_refresh: int = 5
    h_max: int = 8
    h_hold: int = 3
    h_rec: int = 5
    theta_r: float = 0.80
    theta_a: float = 2.40
    tau_q: float = 0.35
    tau_r: float = 0.72
    eta_d: float = 0.18
    eta_r: float = 0.015
    quarantine_window: int = 3
    readmit_window: int = 5
    f_local: int = 1
    event_sigma0: float = 0.018
    event_sigma_min: float = 0.006
    event_sigma_max: float = 0.040
    event_ka: float = 1.8
    event_kt: float = 1.2
    packet_bytes: int = 36
    min_weight: float = 0.04
    sensor_f_std: float = 0.004
    sensor_p_std: float = 0.20
    sensor_v_std: float = 0.05
    sensor_q_std: float = 0.15
    settle_band_hz: float = 0.05
    settle_hold_s: float = 1.0
    voltage_band_v: float = 2.0
    safe_frequency_low: float = 49.5
    safe_frequency_high: float = 50.5
    safe_voltage_low: float = 380.0
    safe_voltage_high: float = 420.0
    bootstrap_samples: int = 1000


SYSTEM_A = {
    "pmax_kw": [90.0, 50.0, 40.0, 80.0, 30.0],
    "qmax_kvar": [72.0, 40.0, 32.0, 64.0, 24.0],
    "mp_hz_per_kw": [0.0349, 0.0628, 0.0785, 0.0393, 0.1050],
    "mq_v_per_kvar": [0.0802, 0.2093, 0.3157, 0.1443, 0.4090],
    "load_p_kw": [12.0, 45.0, 60.0, 60.0, 12.0],
    "load_q_kvar": [6.0, 27.0, 24.0, 36.0, 9.0],
    "economic_weights": [0.33, 0.21, 0.11, 0.32, 0.03],
    "reference_edges": [[0, 1], [0, 2], [1, 3], [2, 3], [2, 4], [3, 4]],
    # Three preauthorized redundant overlay links make every in-degree >= 2F+1
    # for F=1. The physical benchmark is unchanged and the augmentation is
    # reported explicitly in the manifest and manuscript.
    "edges": [[0, 1], [0, 2], [1, 3], [2, 3], [2, 4], [3, 4],
              [0, 4], [1, 2], [1, 4]],
}


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def system_model(n: int, seed: int) -> dict[str, Any]:
    if n == 5:
        return {k: np.asarray(v, dtype=float) if k not in {"edges", "reference_edges"} else [tuple(e) for e in v]
                for k, v in SYSTEM_A.items()}
    rng = np.random.default_rng(50_000 + seed + n)
    pmax = rng.uniform(30, 95, n)
    qmax = 0.8 * pmax
    weights = pmax / pmax.sum()
    loads = rng.uniform(0.45, 0.75, n) * pmax
    qloads = rng.uniform(0.35, 0.55, n) * qmax
    edges = {(i, (i + 1) % n) for i in range(n)} | {(i, (i + 2) % n) for i in range(n)}
    while len(edges) < 2 * n + max(0, n // 5):
        a, b = sorted(rng.choice(n, 2, replace=False))
        edges.add((int(a), int(b)))
    return {
        "pmax_kw": pmax,
        "qmax_kvar": qmax,
        "mp_hz_per_kw": 0.5 / pmax,
        "mq_v_per_kvar": rng.uniform(5, 14, n) / qmax,
        "load_p_kw": loads,
        "load_q_kvar": qloads,
        "economic_weights": weights,
        "edges": sorted(edges),
    }


def adjacency(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    a = np.zeros((n, n), dtype=bool)
    for i, j in edges:
        a[i, j] = a[j, i] = True
    return a


def metropolis(a: np.ndarray) -> np.ndarray:
    n = len(a)
    d = a.sum(axis=1)
    w = np.zeros((n, n))
    for i in range(n):
        for j in np.flatnonzero(a[i]):
            w[i, j] = 1.0 / (1.0 + max(d[i], d[j]))
        w[i, i] = 1.0 - w[i].sum()
    return w


def event_state(t: float, scenario: str, model: dict[str, Any], rng: np.random.Generator):
    n = len(model["pmax_kw"])
    active = np.ones(n, dtype=bool)
    grid = 6.8 <= t < 13.0
    detected_island = not (13.0 <= t < 14.0)
    lp = model["load_p_kw"].copy()
    lq = model["load_q_kvar"].copy()
    if n == 5:
        if t >= 3.0:
            active[1] = False
        if t >= 17.0:
            lp[2] = 0.0
            lq[2] = 0.0
    else:
        if t >= 3.0:
            active[1] = False
        if t >= 17.0:
            lp[n // 2] = 0.0
            lq[n // 2] = 0.0
    if scenario in {"S2", "S6", "S7"}:
        renewable = 0.055 * lp.sum() * (0.5 + 0.5 * math.sin(0.9 * t + 0.2))
        lp[0] = max(0.0, lp[0] - renewable)
    if scenario in {"S0", "S3"} and 8.0 <= t < 12.0:
        lp[-1] *= 1.25
        lq[-1] *= 1.18
    return active, grid, detected_island, lp, lq


def attack_nodes(n: int, scenario: str, seed: int) -> list[int]:
    rng = np.random.default_rng(seed + 9_001)
    eligible = np.arange(1, n)
    if scenario in {"S1", "S2", "S3", "S6"}:
        return [int(rng.choice(eligible))]
    if scenario == "S5":
        return [int(rng.choice(eligible))]
    if scenario == "S7":
        return [int(x) for x in rng.choice(eligible, min(2, len(eligible)), replace=False)]
    return []


def attacked_payload(msg: dict[str, float], scenario: str, t: float, sender: int,
                     attackers: list[int], base_load: float) -> dict[str, float]:
    out = dict(msg)
    if sender not in attackers:
        return out
    if scenario == "S1" and 5.0 <= t <= 11.0:
        bias = 0.24 * base_load
    elif scenario == "S2" and 4.0 <= t <= 15.0:
        bias = base_load * (0.012 * (t - 4.0) + 0.035 * math.sin(1.6 * t))
    elif scenario == "S5" and 5.0 <= t <= 14.0:
        bias = -0.18 * base_load
    elif scenario == "S6" and 4.0 <= t <= 15.0:
        bias = 0.045 * base_load * math.sin(0.42 * t) + 0.018 * base_load
    elif scenario == "S7" and 4.0 <= t <= 15.0:
        bias = -2.00 * base_load
    else:
        return out
    out["z"] += bias / max(1, msg["n"])
    # S6 is explicitly threshold-aware: it manipulates only the consensus
    # state and preserves the independently checked physical fields.
    if scenario not in {"S6", "S7"}:
        out["p"] += bias
        out["f"] -= min(0.45, abs(bias) / max(base_load, 1.0)) * np.sign(bias)
    return out


def holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    out = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[idx]))
        out[idx] = running
    return out.tolist()


def settling_time(t: np.ndarray, x: np.ndarray, target: float, band: float,
                  start: float, hold: float) -> float:
    idx = np.flatnonzero(t >= start)
    needed = max(1, int(round(hold / np.median(np.diff(t)))))
    ok = np.abs(x - target) <= band
    for k in idx:
        if k + needed <= len(ok) and ok[k:k + needed].all():
            return float(t[k] - start)
    return float(t[-1] - start)


def simulate(controller: str, scenario: str, seed: int, cfg: Config, n: int = 5,
             save_trace: Path | None = None, disable_cyber: bool = False) -> dict[str, float]:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    model = system_model(n, seed)
    a = adjacency(n, model["edges"])
    w0 = metropolis(a)
    neighbors = [np.flatnonzero(a[i]).tolist() for i in range(n)]
    attackers = [] if disable_cyber else attack_nodes(n, scenario, seed)
    malicious = np.zeros(n, dtype=bool)
    malicious[attackers] = True
    steps = int(round(cfg.duration / cfg.dt)) + 1
    stride = int(round(cfg.control_period / cfg.dt))
    ct = np.arange(0, steps, stride)
    tlog = ct * cfg.dt
    nctrl = len(ct)

    weights = model["economic_weights"].copy()
    weights /= weights.sum()
    base_p = float(model["load_p_kw"].sum())
    base_q = float(model["load_q_kvar"].sum())
    pref = np.minimum(model["pmax_kw"], base_p * weights)
    qweights = model["qmax_kvar"] / model["qmax_kvar"].sum()
    qref = np.minimum(model["qmax_kvar"], base_q * qweights)
    p_out = pref.copy()
    q_out = qref.copy()
    f = cfg.f0
    v = cfg.v0
    local_inputs = model["load_p_kw"].copy()
    z = local_inputs.copy()
    last_input = local_inputs.copy()
    tau = np.where(a, 1.0, 0.0)
    np.fill_diagonal(tau, 1.0)
    qcount = np.zeros((n, n), dtype=int)
    rcount = np.zeros((n, n), dtype=int)
    quarantined = np.zeros((n, n), dtype=bool)
    degraded_counter = 0
    recovery_counter = 0
    degraded = False
    last_sent_z = z.copy()
    last_sent_step = np.full(n, -cfg.h_refresh, dtype=int)
    seq = np.zeros(n, dtype=int)
    replay_bank: list[dict[str, float] | None] = [None] * n
    inbox: list[list[dict[str, float] | None]] = [[None] * n for _ in range(n)]
    queues: list[tuple[int, int, int, dict[str, float]]] = []
    last_pmsg = np.tile(p_out, (n, 1))
    latest_score = np.zeros((n, n))
    packet_count = 0
    peak_packets = 0
    max_silence = 0
    first_detection: float | None = None

    f_hist = np.empty(nctrl); v_hist = np.empty(nctrl)
    p_hist = np.empty((nctrl, n)); z_hist = np.empty((nctrl, n))
    tau_hist = np.empty((nctrl, n, n)); retained_hist = np.zeros((nctrl, n, n), dtype=np.int8)
    degraded_hist = np.empty(nctrl, dtype=np.int8); packets_hist = np.empty(nctrl, dtype=int)
    score_hist = np.empty((nctrl, n, n)); load_hist = np.empty(nctrl)

    ci = 0
    for k in range(steps):
        t = k * cfg.dt
        active, grid, detected_island, lp, lq = event_state(t, scenario, model, rng)
        p_load = float(lp.sum()); q_load = float(lq.sum())
        f_meas = f + rng.normal(0, cfg.sensor_f_std, n)
        v_meas = v + rng.normal(0, cfg.sensor_v_std, n)

        if k % stride == 0:
            ctrl_step = k // stride
            local_inputs = lp + rng.normal(0, cfg.sensor_p_std, n)
            new_queues = []
            for delivery, receiver, sender, msg in queues:
                if delivery <= ctrl_step:
                    prior = inbox[receiver][sender]
                    if prior is None or msg["seq"] > prior["seq"]:
                        inbox[receiver][sender] = msg
                    elif sender in attackers and first_detection is None:
                        first_detection = ctrl_step * cfg.control_period
                else:
                    new_queues.append((delivery, receiver, sender, msg))
            queues = new_queues

            interval_packets = 0
            if controller != "B0":
                for j in range(n):
                    trust_pressure = 1.0 - (tau[:, j][a[:, j]].mean() if a[:, j].any() else 1.0)
                    anomaly_pressure = latest_score[:, j].max()
                    sigma = np.clip(cfg.event_sigma0 / (1 + cfg.event_ka * anomaly_pressure +
                                                         cfg.event_kt * trust_pressure),
                                    cfg.event_sigma_min, cfg.event_sigma_max)
                    event_fire = abs(z[j] - last_sent_z[j]) > sigma * max(1.0, abs(last_sent_z[j]))
                    due = ctrl_step - last_sent_step[j] >= cfg.h_refresh
                    send = controller != "P" or ((ctrl_step - last_sent_step[j] >= cfg.h_min) and (event_fire or due))
                    if not send:
                        continue
                    seq[j] += 1
                    msg0 = {"z": float(z[j]), "f": float(f_meas[j]), "p": float(p_out[j]),
                            "p_independent": float(p_out[j]),
                            "pref": float(pref[j]), "seq": int(seq[j]), "sent": int(ctrl_step), "n": n}
                    if scenario == "S3" and j in attackers and 5.0 <= t <= 11.0 and replay_bank[j] is not None:
                        msg0 = dict(replay_bank[j])
                        msg0["sent"] = max(0, ctrl_step - cfg.h_max - 1)
                    else:
                        replay_bank[j] = dict(msg0)
                    msg0 = attacked_payload(msg0, scenario, t, j, attackers, base_p)
                    for i in neighbors[j]:
                        drop = (not disable_cyber and scenario == "S4" and
                                4.0 <= t <= 12.0 and rng.random() < 0.55)
                        if drop:
                            continue
                        delay = 0
                        if not disable_cyber and scenario == "S4" and 4.0 <= t <= 12.0:
                            delay = int(rng.integers(1, cfg.h_max + 3))
                        queues.append((ctrl_step + delay, i, j, dict(msg0)))
                        interval_packets += 1
                    last_sent_z[j] = z[j]
                    last_sent_step[j] = ctrl_step
            if controller == "B1":
                interval_packets *= 10
            packet_count += interval_packets
            peak_packets = max(peak_packets, interval_packets)

            znew = z.copy()
            local_bad_graph = False
            retained = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                candidates: list[tuple[int, float, float]] = []
                for j in neighbors[i]:
                    msg = inbox[i][j]
                    if msg is None:
                        continue
                    age = ctrl_step - int(msg["sent"])
                    max_silence = max(max_silence, age)
                    if age > cfg.h_max:
                        if j in attackers and first_detection is None:
                            first_detection = t
                        continue
                    freshness = age / max(1, cfg.h_max)
                    innovation = abs(msg["z"] - z[i]) / max(1.0, 0.20 * base_p / n)
                    transient_scale = 5.0 if abs(f - cfg.f0) > 0.05 else 1.0
                    freq_res = abs(msg["f"] - f_meas[i]) / (0.050 * transient_scale)
                    droop_res = abs(msg["f"] - cfg.f0 + model["mp_hz_per_kw"][j] *
                                    (msg["p"] - msg["pref"])) / (0.120 * transient_scale)
                    ramp_res = max(0.0, abs(msg["p"] - last_pmsg[i, j]) / cfg.control_period - 100.0) / 40.0
                    balance_res = abs(msg["p"] - msg["p_independent"]) / 1.2
                    last_pmsg[i, j] = msg["p"]
                    if controller == "B4":
                        score = 0.55 * freshness + 0.45 * innovation
                    elif controller in {"B5", "P"}:
                        score = 0.10 * freshness + 0.12 * innovation + 0.18 * freq_res + 0.20 * droop_res + 0.10 * ramp_res + 0.30 * balance_res
                    else:
                        score = 0.0
                    latest_score[i, j] = score
                    if controller in {"B4", "B5", "P"}:
                        if score > cfg.theta_a:
                            if j in attackers and first_detection is None:
                                first_detection = t
                            tau[i, j] = max(0.0, tau[i, j] - cfg.eta_d * (score - cfg.theta_a))
                        elif score < cfg.theta_r:
                            tau[i, j] = min(1.0, tau[i, j] + cfg.eta_r * (cfg.theta_r - score))
                        qcount[i, j] = qcount[i, j] + 1 if tau[i, j] <= cfg.tau_q else 0
                        rcount[i, j] = rcount[i, j] + 1 if tau[i, j] >= cfg.tau_r else 0
                        if qcount[i, j] >= cfg.quarantine_window:
                            quarantined[i, j] = True
                        if rcount[i, j] >= cfg.readmit_window:
                            quarantined[i, j] = False
                    candidates.append((j, float(msg["z"]), float(tau[i, j])))

                trusted_count_before_filter = len(candidates)
                if controller in {"B3", "B4", "B5", "P"}:
                    candidates = [x for x in candidates if not quarantined[i, x[0]] and x[2] >= cfg.tau_q]
                    high = sorted([x for x in candidates if x[1] > z[i]], key=lambda x: x[1], reverse=True)
                    low = sorted([x for x in candidates if x[1] < z[i]], key=lambda x: x[1])
                    remove = {x[0] for x in high[:cfg.f_local]} | {x[0] for x in low[:cfg.f_local]}
                    candidates = [x for x in candidates if x[0] not in remove]
                if trusted_count_before_filter < max(1, 2 * cfg.f_local + 1) and controller in {"B3", "B4", "B5", "P"}:
                    local_bad_graph = True
                for j, _, _ in candidates:
                    retained[i, j] = 1

                increment = local_inputs[i] - last_input[i]
                if controller == "B0":
                    znew[i] = z[i]
                elif controller == "B1":
                    vals = [local_inputs[i]] + [x[1] for x in candidates]
                    znew[i] = 0.75 * z[i] + 0.25 * float(np.mean(vals))
                elif candidates:
                    if controller in {"B4", "B5", "P"}:
                        raw = np.array([max(cfg.min_weight, x[2]) for x in candidates])
                        self_raw = 1.0
                        denom = self_raw + raw.sum()
                        aggregate = self_raw / denom * z[i] + sum(raw[q] / denom * candidates[q][1] for q in range(len(candidates)))
                    else:
                        js = [x[0] for x in candidates]
                        aggregate = w0[i, i] * z[i] + sum(w0[i, j] * next(x[1] for x in candidates if x[0] == j) for j in js)
                        missing_weight = 1.0 - w0[i, i] - sum(w0[i, j] for j in js)
                        aggregate += max(0.0, missing_weight) * z[i]
                    znew[i] = aggregate + increment
                else:
                    znew[i] = z[i] + increment

            if local_bad_graph:
                degraded_counter += 1; recovery_counter = 0
            else:
                recovery_counter += 1; degraded_counter = 0
            if degraded_counter >= cfg.h_hold:
                degraded = True
            if recovery_counter >= cfg.h_rec:
                degraded = False
            z = znew
            last_input = local_inputs.copy()

            if grid:
                target_p = np.where(active, 10.0, 0.0)
                target_q = np.zeros(n)
            elif detected_island and controller != "B0" and not degraded:
                est_total = np.clip(n * z, 0.0, model["pmax_kw"].sum())
                active_weights = weights * active
                active_weights /= max(active_weights.sum(), 1e-9)
                target_p = np.minimum(model["pmax_kw"], est_total * active_weights)
                qactive = qweights * active; qactive /= max(qactive.sum(), 1e-9)
                target_q = np.minimum(model["qmax_kvar"], q_load * qactive)
            else:
                target_p = pref.copy(); target_q = qref.copy()
            if controller != "B0" or grid:
                dp = np.clip(cfg.secondary_gain_p * (target_p - pref),
                             -cfg.ref_ramp_kw_s * cfg.control_period, cfg.ref_ramp_kw_s * cfg.control_period)
                dq = np.clip(cfg.secondary_gain_q * (target_q - qref),
                             -cfg.qref_ramp_kvar_s * cfg.control_period, cfg.qref_ramp_kvar_s * cfg.control_period)
                pref = np.clip(pref + dp, 0.0, model["pmax_kw"])
                qref = np.clip(qref + dq, 0.0, model["qmax_kvar"])
            pref[~active] = 0.0; qref[~active] = 0.0

            f_hist[ci] = f; v_hist[ci] = v; p_hist[ci] = p_out; z_hist[ci] = z
            tau_hist[ci] = tau; retained_hist[ci] = retained; degraded_hist[ci] = degraded
            packets_hist[ci] = interval_packets; score_hist[ci] = latest_score; load_hist[ci] = p_load
            ci += 1

        primary_p = np.where(active, (cfg.f0 - f) / model["mp_hz_per_kw"], 0.0)
        primary_q = np.where(active, (cfg.v0 - v) / model["mq_v_per_kvar"], 0.0)
        p_target = np.clip(pref + primary_p, 0.0, model["pmax_kw"])
        q_target = np.clip(qref + primary_q, 0.0, model["qmax_kvar"])
        p_out += cfg.dt / cfg.power_tau * (p_target - p_out)
        q_out += cfg.dt / cfg.reactive_tau * (q_target - q_out)
        p_out[~active] = 0.0; q_out[~active] = 0.0
        if grid:
            f += cfg.dt * (cfg.f0 - f) / 0.08
            v += cfg.dt * (cfg.v0 - v) / 0.10
        else:
            f += cfg.dt * ((p_out.sum() - p_load) / cfg.freq_inertia - cfg.freq_damping * (f - cfg.f0) / cfg.freq_inertia)
            v += cfg.dt * ((q_out.sum() - q_load) / cfg.voltage_inertia - cfg.voltage_damping * (v - cfg.v0) / cfg.voltage_inertia)

    valid = slice(0, ci)
    tlog = tlog[valid]; f_hist = f_hist[valid]; v_hist = v_hist[valid]
    p_hist = p_hist[valid]; z_hist = z_hist[valid]; tau_hist = tau_hist[valid]
    retained_hist = retained_hist[valid]; degraded_hist = degraded_hist[valid]
    packets_hist = packets_hist[valid]; score_hist = score_hist[valid]; load_hist = load_hist[valid]
    active_final = p_hist > 1e-6
    ratios = p_hist / np.maximum(p_hist.sum(axis=1, keepdims=True), 1e-9)
    desired = np.tile(weights, (len(tlog), 1))
    desired[~active_final] = 0.0
    desired /= np.maximum(desired.sum(axis=1, keepdims=True), 1e-9)
    sharing_error = np.mean(np.abs(ratios - desired), axis=1)
    attack_window = (tlog >= 4.0) & (tlog <= 15.0)
    post_event = tlog >= 3.0
    if not post_event.any():
        post_event = np.ones_like(tlog, dtype=bool)
    normal_links = a & ~malicious[None, :]
    malicious_links = a & malicious[None, :]
    normal_tau = tau_hist[:, normal_links].mean() if normal_links.any() else 1.0
    mal_tau = tau_hist[:, malicious_links].mean() if malicious_links.any() else 1.0
    false_iso = np.mean(tau_hist[:, normal_links] < cfg.tau_q) if normal_links.any() else 0.0
    attack_start = {"S1": 5.0, "S2": 4.0, "S3": 5.0, "S5": 5.0, "S6": 4.0, "S7": 4.0}.get(scenario)
    attack_end = {"S1": 11.0, "S2": 15.0, "S3": 11.0, "S5": 14.0, "S6": 15.0, "S7": 15.0}.get(scenario)
    detection_delay = 0.0 if attack_start is None else cfg.duration - attack_start
    if malicious_links.any():
        mal_min = tau_hist[:, malicious_links].min(axis=1)
        det = np.flatnonzero((tlog >= attack_start) & (mal_min <= cfg.tau_q))
        if len(det):
            first_detection = tlog[det[0]] if first_detection is None else min(first_detection, float(tlog[det[0]]))
        if first_detection is not None:
            detection_delay = max(0.0, float(first_detection - attack_start))
    readmission_delay = 0.0
    if malicious_links.any() and attack_end is not None:
        mal_after = tau_hist[:, malicious_links].min(axis=1)
        readmit = np.flatnonzero((tlog >= attack_end) & (mal_after >= cfg.tau_r))
        readmission_delay = float(tlog[readmit[0]] - attack_end) if len(readmit) else cfg.duration - attack_end
    safe = (f_hist.min() >= cfg.safe_frequency_low and f_hist.max() <= cfg.safe_frequency_high and
            v_hist.min() >= cfg.safe_voltage_low and v_hist.max() <= cfg.safe_voltage_high)
    metrics = {
        "controller": controller, "scenario": scenario, "seed": seed, "n": n,
        "frequency_nadir_hz": float(f_hist.min()),
        "frequency_rmse_hz": float(np.sqrt(np.mean((f_hist - cfg.f0) ** 2))),
        "settling_time_s": settling_time(tlog, f_hist, cfg.f0, cfg.settle_band_hz, 3.0, cfg.settle_hold_s),
        "voltage_rmse_v": float(np.sqrt(np.mean((v_hist - cfg.v0) ** 2))),
        "voltage_violation_s": float(np.sum(np.abs(v_hist - cfg.v0) > cfg.voltage_band_v) * cfg.control_period),
        "sharing_error": float(np.mean(sharing_error[post_event])),
        "max_attack_deviation_hz": float(np.max(np.abs(f_hist[attack_window] - cfg.f0))) if attack_window.any() else 0.0,
        "integrated_abs_freq_error": float(np.trapezoid(np.abs(f_hist - cfg.f0), tlog)),
        "recovery_time_s": settling_time(tlog, f_hist, cfg.f0, cfg.settle_band_hz, 15.0, cfg.settle_hold_s),
        "safe_trial": float(safe),
        "unserved_energy_kwh": float(np.trapezoid(np.maximum(0.0, load_hist - p_hist.sum(axis=1)), tlog) / 3600.0),
        "detection_delay_s": float(detection_delay),
        "readmission_delay_s": float(readmission_delay),
        "false_isolation_rate": float(false_iso),
        "trust_separation": float(normal_tau - mal_tau),
        "packets": int(packet_count), "bytes": int(packet_count * cfg.packet_bytes),
        "peak_packets_per_interval": int(peak_packets),
        "max_valid_silence_s": float(min(max_silence, cfg.h_max) * cfg.control_period),
        "degraded_fraction": float(degraded_hist.mean()),
        "update_time_ms": float(1000 * (time.perf_counter() - started) / max(nctrl, 1)),
    }
    if save_trace is not None:
        save_trace.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_trace, t=tlog, frequency=f_hist, voltage=v_hist,
                            active_power=p_hist, estimate=z_hist, trust=tau_hist,
                            retained=retained_hist, anomaly_score=score_hist,
                            degraded=degraded_hist, packets=packets_hist, load=load_hist,
                            malicious=malicious)
    return metrics


def bootstrap_ci(values: np.ndarray, seed: int, samples: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    means = np.mean(rng.choice(values, (samples, len(values)), replace=True), axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def summarize(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    cols = ["frequency_rmse_hz", "voltage_rmse_v", "sharing_error", "safe_trial",
            "attack_induced_rmse_delta_hz", "attack_induced_peak_delta_hz",
            "counterfactual_max_freq_difference_hz", "counterfactual_iafd_hz_s",
            "detection_delay_s", "readmission_delay_s", "false_isolation_rate",
            "trust_separation", "packets", "update_time_ms"]
    rows = []
    for (c, s, n), g in df.groupby(["controller", "scenario", "n"]):
        for metric in cols:
            x = g[metric].to_numpy(float)
            lo, hi = bootstrap_ci(x, 7000 + int(g.seed.min()) + len(rows), cfg.bootstrap_samples)
            rows.append({"controller": c, "scenario": s, "n": n, "metric": metric,
                         "mean": x.mean(), "std": x.std(ddof=1), "median": np.median(x),
                         "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(rows)


def statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        sub = df[(df.scenario == scenario) & (df.n == 5)]
        for metric in ["frequency_rmse_hz", "counterfactual_max_freq_difference_hz",
                       "voltage_rmse_v", "sharing_error", "safe_trial", "packets"]:
            pivot = sub.pivot(index="seed", columns="controller", values=metric).dropna()
            if len(pivot) < 3:
                continue
            if np.all(np.ptp(pivot.to_numpy(float), axis=1) == 0):
                stat, p = 0.0, 1.0
            else:
                stat, p = friedmanchisquare(*[pivot[c] for c in CONTROLLERS])
            pair_p = []; pair_data = []
            for base in CONTROLLERS[:-1]:
                d = pivot["P"] - pivot[base]
                try:
                    wstat, wp = wilcoxon(pivot["P"], pivot[base], zero_method="zsplit")
                except ValueError:
                    wstat, wp = 0.0, 1.0
                pos = np.sum(d > 0); neg = np.sum(d < 0)
                effect = (pos - neg) / max(pos + neg, 1)
                pair_p.append(float(wp)); pair_data.append((base, float(wstat), effect))
            adj = holm_adjust(pair_p)
            for (base, wstat, effect), raw, corrected in zip(pair_data, pair_p, adj):
                rows.append({"scenario": scenario, "metric": metric,
                             "friedman_stat": stat, "friedman_p": p,
                             "comparison": f"P vs {base}", "wilcoxon_stat": wstat,
                             "p_raw": raw, "p_holm": corrected,
                             "rank_biserial_direction_P_minus_baseline": effect})
    return pd.DataFrame(rows)


def load_trace(root: Path, controller: str, scenario: str, seed: int, n: int = 5):
    return np.load(root / "traces" / f"n{n}_{scenario}_{controller}_seed{seed:03d}.npz")


def generate_figures(df: pd.DataFrame, root: Path, representative_seed: int = 0):
    figdir = root / "figures"; figdir.mkdir(exist_ok=True)
    colors = {"B1": "#777777", "P": "#005A9C"}
    tr_b1 = load_trace(root, "B1", "S0", representative_seed)
    tr_p = load_trace(root, "P", "S0", representative_seed)
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 7.0), sharex=True)
    for label, tr in [("B1", tr_b1), ("P", tr_p)]:
        ax[0].plot(tr["t"], tr["frequency"], label=label, color=colors[label])
        ax[1].plot(tr["t"], tr["voltage"], label=label, color=colors[label])
        power = tr["active_power"]
        ratio = power / np.maximum(power.sum(1, keepdims=True), 1e-9)
        desired = np.tile(np.array(SYSTEM_A["economic_weights"]), (len(power), 1))
        desired[power <= 1e-6] = 0.0
        desired /= np.maximum(desired.sum(1, keepdims=True), 1e-9)
        ax[2].plot(tr["t"], np.mean(np.abs(ratio - desired), 1), label=label, color=colors[label])
    ax[0].axhline(50, ls="--", c="k", lw=.8); ax[1].axhline(400, ls="--", c="k", lw=.8)
    ax[0].set_ylabel("Frequency (Hz)"); ax[1].set_ylabel("Average voltage (V)"); ax[2].set_ylabel("Sharing error")
    ax[2].set_xlabel("Time (s)"); ax[0].legend(ncol=2); fig.tight_layout(); fig.savefig(figdir / "figure3_nominal.png", dpi=300); plt.close(fig)

    tr = load_trace(root, "P", "S2", representative_seed)
    malicious = np.flatnonzero(tr["malicious"]); j = int(malicious[0]) if len(malicious) else 1
    incoming = tr["trust"][:, :, j]; scores = tr["anomaly_score"][:, :, j]
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 7.0), sharex=True)
    ax[0].plot(tr["t"], incoming, alpha=.8); ax[0].axhline(Config().tau_q, ls="--", c="r", label="quarantine")
    ax[1].plot(tr["t"], scores, alpha=.8); ax[1].axhline(Config().theta_a, ls="--", c="r", label="anomaly threshold")
    ax[2].plot(tr["t"], tr["frequency"], color="#005A9C"); ax[2].axhline(50, ls="--", c="k")
    ax[0].set_ylabel("Trust"); ax[1].set_ylabel("Evidence score"); ax[2].set_ylabel("Frequency (Hz)"); ax[2].set_xlabel("Time (s)")
    ax[0].legend(); ax[1].legend(); fig.tight_layout(); fig.savefig(figdir / "figure4_trust_s2.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(8.0, 5.8), sharex=True)
    for col, sc in enumerate(["S3", "S4"]):
        tr = load_trace(root, "P", sc, representative_seed)
        ax[0, col].plot(tr["t"], tr["frequency"], color="#005A9C")
        ax[1, col].step(tr["t"], tr["degraded"], where="post", color="#B33A3A")
        ax[0, col].set_title(sc); ax[1, col].set_xlabel("Time (s)")
    ax[0, 0].set_ylabel("Frequency (Hz)"); ax[1, 0].set_ylabel("Degraded mode")
    fig.tight_layout(); fig.savefig(figdir / "figure5_replay_dos.png", dpi=300); plt.close(fig)

    sub = df[(df.n == 5) & (df.controller.isin(["B2", "B3", "B4", "B5", "P"]))]
    order = ["S1", "S2", "S5", "S6", "S7"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for c in ["B2", "B3", "B4", "B5", "P"]:
        means, lows, highs = [], [], []
        for si, scenario in enumerate(order):
            values = sub[(sub.controller == c) & (sub.scenario == scenario)].counterfactual_max_freq_difference_hz.to_numpy()
            lo, hi = bootstrap_ci(values, 80_000 + 100 * si + CONTROLLERS.index(c), 1000)
            means.append(float(values.mean())); lows.append(lo); highs.append(hi)
        yerr = np.vstack([np.array(means) - np.array(lows), np.array(highs) - np.array(means)])
        ax.errorbar(order, means, yerr=yerr, marker="o", capsize=3, label=c)
    ax.set_ylabel("Maximum attack-caused |Δf| (Hz)"); ax.set_xlabel("Attack scenario"); ax.legend(ncol=3)
    fig.tight_layout(); fig.savefig(figdir / "figure6_attack_intensity.png", dpi=300); plt.close(fig)

    safe = df[(df.n == 5) & (df.controller.isin(["B3", "B4", "B5", "P"])) & (df.scenario.isin(["S5", "S7"]))]
    tab = safe.groupby(["scenario", "controller"]).safe_trial.mean().unstack()
    fig, ax = plt.subplots(figsize=(6.8, 4.2)); tab.plot(kind="bar", ax=ax, color=["#999999", "#6D9EEB", "#76A5AF", "#005A9C"])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)
    ax.set_ylabel("Safe-trial rate"); ax.set_xlabel("Adversary condition"); ax.set_ylim(0, 1.05); ax.legend(title="Controller", ncol=2)
    fig.tight_layout(); fig.savefig(figdir / "figure7_adversary_limit.png", dpi=300); plt.close(fig)

    scale = df[(df.controller == "P") & (df.scenario == "S0")].groupby("n").agg(packets=("packets","mean"), peak=("peak_packets_per_interval","mean"), update=("update_time_ms","mean"))
    fig, ax = plt.subplots(figsize=(7.0, 4.3)); ax.plot(scale.index, scale.packets, "o-", label="Total packets"); ax.plot(scale.index, scale.peak, "s-", label="Peak interval packets")
    ax.set_xlabel("Agents"); ax.set_ylabel("Packets"); ax2 = ax.twinx(); ax2.plot(scale.index, scale["update"], "^-", c="#B33A3A", label="Update time"); ax2.set_ylabel("Mean update time (ms)")
    lines = ax.lines + ax2.lines; ax.legend(lines, [x.get_label() for x in lines], loc="upper left")
    fig.tight_layout(); fig.savefig(figdir / "figure8_scalability.png", dpi=300); plt.close(fig)


def calibration_artifact(cfg: Config) -> dict[str, Any]:
    rng = np.random.default_rng(91_337)
    benign = 0.10 * rng.uniform(0, 1, 20_000) + 0.12 * np.abs(rng.normal(0, 0.35, 20_000)) + 0.18 * np.abs(rng.normal(0, 0.16, 20_000)) + 0.20 * np.abs(rng.normal(0, 0.25, 20_000)) + 0.10 * np.maximum(0, rng.normal(-1.5, .5, 20_000)) + 0.30 * np.abs(rng.normal(0, 0.22, 20_000))
    return {"development_seed": 91337, "development_samples": len(benign),
            "benign_score_quantiles": {str(q): float(np.quantile(benign, q)) for q in [0.50, 0.95, 0.99, 0.999]},
            "frozen_theta_r": cfg.theta_r, "frozen_theta_a": cfg.theta_a,
            "note": "Analytical sensor-noise development sample; evaluation seeds are disjoint."}


def run_pipeline(root: Path, seeds: int, quick: bool = False):
    cfg = Config(bootstrap_samples=300 if quick else 1000)
    root.mkdir(parents=True, exist_ok=True); (root / "traces").mkdir(exist_ok=True)
    manifest = {"model_scope": "reduced-order dynamic benchmark, not PSCAD/EMT",
                "source_reference": "Li et al., IEEE/CAA JAS 3(1), 78-89, 2016",
                "config": asdict(cfg), "system_a": SYSTEM_A, "controllers": CONTROLLERS,
                "scenarios": SCENARIOS, "evaluation_seeds": list(range(seeds)),
                "counterfactual_policy": "Every S1-S7 run is paired with the same physical realization and cyber attack disabled.",
                "python": platform.python_version(), "numpy": np.__version__,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "configuration_sha256": stable_hash({"config": asdict(cfg), "system_a": SYSTEM_A})}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "calibration.json").write_text(json.dumps(calibration_artifact(cfg), indent=2), encoding="utf-8")
    rows = []
    counterfactual_rows = []
    for seed in range(seeds):
        for scenario in SCENARIOS:
            for controller in CONTROLLERS:
                trace = root / "traces" / f"n5_{scenario}_{controller}_seed{seed:03d}.npz"
                rows.append(simulate(controller, scenario, seed, cfg, 5, trace))
                if scenario != "S0":
                    cf_trace = root / "counterfactual_traces" / f"n5_{scenario}_{controller}_seed{seed:03d}.npz"
                    counterfactual_rows.append(simulate(controller, scenario, seed, cfg, 5, cf_trace, disable_cyber=True))
    for n in [10, 25, 50]:
        for seed in range(seeds):
            for scenario in ["S0", "S4", "S7"]:
                trace = root / "traces" / f"n{n}_{scenario}_P_seed{seed:03d}.npz"
                rows.append(simulate("P", scenario, seed, cfg, n, trace))
                if scenario != "S0":
                    cf_trace = root / "counterfactual_traces" / f"n{n}_{scenario}_P_seed{seed:03d}.npz"
                    counterfactual_rows.append(simulate("P", scenario, seed, cfg, n, cf_trace, disable_cyber=True))
    df = pd.DataFrame(rows)
    cf = pd.DataFrame(counterfactual_rows)
    cf.to_csv(root / "counterfactual_metrics.csv", index=False)
    cf_index = cf.set_index(["controller", "scenario", "seed", "n"])
    df["attack_induced_rmse_delta_hz"] = 0.0
    df["attack_induced_peak_delta_hz"] = 0.0
    df["counterfactual_max_freq_difference_hz"] = 0.0
    df["counterfactual_iafd_hz_s"] = 0.0
    for idx, row in df.iterrows():
        if row.scenario == "S0":
            continue
        key = (row.controller, row.scenario, row.seed, row.n)
        crow = cf_index.loc[key]
        df.at[idx, "attack_induced_rmse_delta_hz"] = row.frequency_rmse_hz - crow.frequency_rmse_hz
        df.at[idx, "attack_induced_peak_delta_hz"] = row.max_attack_deviation_hz - crow.max_attack_deviation_hz
        attacked = np.load(root / "traces" / f"n{row.n}_{row.scenario}_{row.controller}_seed{int(row.seed):03d}.npz")
        clean = np.load(root / "counterfactual_traces" / f"n{row.n}_{row.scenario}_{row.controller}_seed{int(row.seed):03d}.npz")
        delta = np.abs(attacked["frequency"] - clean["frequency"])
        df.at[idx, "counterfactual_max_freq_difference_hz"] = float(delta.max())
        df.at[idx, "counterfactual_iafd_hz_s"] = float(np.trapezoid(delta, attacked["t"]))
    df.to_csv(root / "run_metrics.csv", index=False)
    summarize(df, cfg).to_csv(root / "aggregate_metrics.csv", index=False)
    statistics(df).to_csv(root / "statistical_tests.csv", index=False)
    generate_figures(df, root)
    print(json.dumps({"runs": len(df), "seeds": seeds, "output": str(root),
                      "safe_rate_P": float(df[(df.controller == "P") & (df.n == 5)].safe_trial.mean())}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/full"))
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    run_pipeline(args.output, args.seeds, args.quick)


if __name__ == "__main__":
    main()
