"""arXiv:2610.17283 の主張群をハーネスの Check として登録する。

各 Check は「構成を組み立てて全入力で照合する」最小コードに対応し、
他の論文を検証するときのテンプレートになる。
"""
from __future__ import annotations

from itertools import product

import numpy as np

from .asymmetry import best_threshold_accuracy
from .circuits import (
    adder_circuit,
    mlp_relu_circuit,
    parity_circuit,
    random_threshold_circuit,
)
from .harness import Check, PaperVerifier
from .learn import fit_cot_perceptron
from .simulate import build_neuron, designated_values, run_fast, run_reference


def _inputs(n: int) -> np.ndarray:
    return np.array(list(product([0, 1], repeat=n)), dtype=float)


def _check_thm2(fast: bool):
    circuits = [parity_circuit(3), parity_circuit(4), adder_circuit(2)]
    if not fast:
        circuits += [adder_circuit(3), random_threshold_circuit(8, 12, seed=0)]
    details = []
    ok = True
    for c in circuits:
        plan = build_neuron(c)
        X = _inputs(c.n)
        tok = run_fast(plan, X)
        equal = bool((tok[:, -1] == c.forward(X)[:, -1]).all())
        binary = bool(((tok == 0) | (tok == 1)).all())
        ok &= equal and binary
        details.append(f"s={plan.s},n={c.n}:{int(equal)}")
    return ok, f"{len(circuits)} circuits / all inputs; " + "; ".join(details)


def _check_prop1(fast: bool):
    circuits = [parity_circuit(3), adder_circuit(2)]
    ok = True
    for c in circuits:
        plan = build_neuron(c)
        X = _inputs(c.n)
        tok = run_fast(plan, X)
        des = designated_values(tok, plan)
        ok &= bool((des == c.forward(X)).all())
        mask = np.ones(plan.T, dtype=bool)
        mask[np.array(plan.designated) - 1] = False
        ok &= bool((tok[:, mask] == 0).all())
    return bool(ok), f"C_i on designated steps of {len(circuits)} circuits; 0 elsewhere"


def _check_bounds(fast: bool):
    circuits = [parity_circuit(3), parity_circuit(4), adder_circuit(2)]
    if not fast:
        circuits += [adder_circuit(3), random_threshold_circuit(8, 12, seed=0)]
    rT, rD = 0.0, 0.0
    for c in circuits:
        plan = build_neuron(c)
        n, s = plan.n, plan.s
        assert plan.d == 2 * plan.T + plan.k * plan.t_prime
        assert plan.designated[-1] == plan.T
        rT = max(rT, plan.T / (s * s))
        rD = max(rD, plan.d / (n * s + s * s))
    ok = bool(rT <= 16.0 and rD <= 100.0)
    return ok, f"max T/s^2 = {rT:.2f} (=O(1)), max d/(ns+s^2) = {rD:.2f} (=O(1))"


def _check_thm3_relu(fast: bool):
    c = mlp_relu_circuit()
    plan = build_neuron(c)
    X = _inputs(3)
    tok = run_fast(plan, X)
    gv = c.forward(X)
    e2e_err = float(np.abs(tok[:, -1] - gv[:, -1]).max())
    des_err = float(np.abs(designated_values(tok, plan) - gv).max())
    t_arr = np.array(plan.designated) - 1
    mask = np.ones(plan.T, dtype=bool)
    mask[t_arr] = False
    zero_err = float(np.abs(tok[:, mask]).max())
    ok = bool(e2e_err < 1e-9 and des_err < 1e-9 and zero_err < 1e-9)
    return ok, f"e2e err {e2e_err:.1e}, gate err {des_err:.1e}, undesignated {zero_err:.1e}"


def _check_no_cot_gap(fast: bool):
    X = _inputs(3)
    y = (X.sum(axis=1) % 2).astype(float)
    acc, _wt = best_threshold_accuracy(X, y, weight_cap=12)
    ok = bool(acc < 1.0)
    return ok, f"best LTF agreement on parity3 = {acc:.3f} (< 1.0: not a threshold function)"


def _check_cor2(fast: bool):
    out = []
    ok = True
    for name, factory, n_train, seed in [
        ("parity3", lambda: parity_circuit(3), 6, 0),
        ("adder2", lambda: adder_circuit(2), 12, 0),
    ]:
        plan = build_neuron(factory(), bias_margin=0.5)
        X = _inputs(plan.n)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        train, test = perm[:n_train], perm[n_train:]
        w, info = fit_cot_perceptron(plan, X[train])
        tok_t = run_fast(plan, X)
        tok_s = run_fast(plan.with_weights(w), X)
        train_ok = bool((tok_s[train] == tok_t[train]).all())
        acc = float((tok_s[test][:, -1] == tok_t[test][:, -1]).mean())
        ok &= bool(info["converged"] and train_ok and acc >= 0.99)
        out.append(f"{name}: conv={info['converged']}, held-out {acc:.0%}")
    return ok, "; ".join(out)


def _check_impl_consistency(fast: bool):
    c = adder_circuit(2)
    plan = build_neuron(c)
    X = _inputs(4)
    same = all(
        np.array_equal(run_reference(plan, x), run_fast(plan, x[None, :])[0])
        for x in X
    )
    return bool(same), "reference (dense right-aligned dot) == fast (sparse) trajectories"


def build_verifier(fast: bool = False) -> PaperVerifier:
    v = PaperVerifier(
        "The Surprising Expressivity of a Single Neuron with Autoregressive CoT",
        "arXiv:2610.17283 (Magen, Joshi, Vardi, 2026) / unofficial verification",
    )
    v.add(Check(id="thm2-exact", claim="size-s threshold circuit simulated exactly by one AR neuron",
                reference="Theorem 2 / C.1", fn=lambda: _check_thm2(fast)))
    v.add(Check(id="prop1-cot", claim="gate values appear at designated steps, 0 elsewhere",
                reference="Property 1", fn=lambda: _check_prop1(fast)))
    v.add(Check(id="bounds", claim="d = 2T + kT' = O(ns+s^2), T = O(s^2) with Sidon scheduling",
                reference="Sec. 5 / C.1", fn=lambda: _check_bounds(fast)))
    v.add(Check(id="thm3-relu", claim="ReLU circuit (real-valued) simulated exactly",
                reference="Theorem 3 (exact)", fn=lambda: _check_thm3_relu(fast)))
    v.add(Check(id="no-cot-gap", claim="without CoT a single threshold cannot compute parity3",
                reference="Sec. 1 (vs Joshi+25)", fn=lambda: _check_no_cot_gap(fast)))
    v.add(Check(id="cor2-learning", claim="efficient learnability from complete CoT trajectories",
                reference="Corollary 2", fn=lambda: _check_cor2(fast)))
    v.add(Check(id="impl-consistency", claim="two simulators produce identical trajectories",
                reference="implementation", fn=lambda: _check_impl_consistency(fast)))
    return v
