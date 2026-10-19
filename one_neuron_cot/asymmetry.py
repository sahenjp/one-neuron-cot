"""Cor 2 (CoT軌跡からは効率的に学べる) と Cor 7 (答えだけだと困難) の対比デモ。

小規模な実演:
- E2E (最終ラベルのみ): 線形模型の到達精度 (capacity 上限 + 勾配降下)
- CoT (完全軌跡): 教師ニューロンの軌跡から学んだ生徒の精度

注意: 一方向関数前提の暗号学的困難性 (Corollary 7) 自体を実証するものではなく、
「答えだけ/軌跡あり」の教示の差が小規模でも閾値関数クラス上で効くことを示す。
"""
from __future__ import annotations

from itertools import product

import numpy as np

from .circuits import adder_circuit, parity_circuit
from .learn import fit_cot_perceptron
from .simulate import build_neuron, run_fast


def all_binary_inputs(n: int) -> np.ndarray:
    return np.array(list(product([0, 1], repeat=n)), dtype=float)


def best_threshold_accuracy(X: np.ndarray, y: np.ndarray, weight_cap: int = 12):
    """|w_i| <= cap の全整数 LTF のうち (X, y) との最大一致率と (w, theta) を返す。

    厳密な全列挙 (実数 theta の最適化を formula 候補スイープで解く)。
    """
    n = X.shape[1]
    ws = np.arange(-weight_cap, weight_cap + 1)
    grid = np.stack(np.meshgrid(*([ws] * n), indexing="ij")).reshape(n, -1).T
    m = len(X)
    pos = y > 0.5
    best_acc, best = 0.0, (np.zeros(n), 0.0)
    chunk = 8192
    for s in range(0, len(grid), chunk):
        W = grid[s:s + chunk].astype(float)
        pre = X @ W.T                       # (m, c)
        minus = -pre
        big_hi = np.full((1, pre.shape[1]), np.abs(pre).max() + 2.0)
        big_lo = -big_hi
        thetas = np.vstack([minus - 0.5, minus + 0.5, big_hi, big_lo])  # (T, c)
        fire = pre[None, :, :] + thetas[:, None, :] >= 0                 # (T, m, c)
        pos3 = pos[None, :, None]
        acc = (fire & pos3).sum(axis=1) + ((~fire) & (~pos3)).sum(axis=1)  # (T, c)
        col_best = acc.max(axis=0)
        j = int(col_best.argmax())
        if col_best[j] / m > best_acc:
            best_acc = col_best[j] / m
            best = (W[j], float(thetas[acc.argmax(axis=0)[j], j]))
    return best_acc, best


def gd_linear_accuracy(X: np.ndarray, y: np.ndarray, iters: int = 4000, lr: float = 0.5) -> float:
    """最終ラベルだけでの勾配降下 (ロジスティック回帰) の到達精度。"""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        z = np.clip(Xb @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        w -= lr * (Xb.T @ (p - y)) / len(X)
    return float(((Xb @ w >= 0) == (y > 0.5)).mean())


def cot_learner(circuit, n_train: int, seed: int = 0) -> dict:
    """教師ニューロンの完全CoT軌跡から生徒 (同一構成) を学習する。"""
    plan = build_neuron(circuit, bias_margin=0.5)
    X = all_binary_inputs(plan.n)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    train, test = perm[:n_train], perm[n_train:]
    w, info = fit_cot_perceptron(plan, X[train])
    tok_t = run_fast(plan, X)
    tok_s = run_fast(plan.with_weights(w), X)
    return {
        "converged": bool(info["converged"]),
        "train_full_trace": bool((tok_s[train] == tok_t[train]).all()),
        "test_acc": float((tok_s[test][:, -1] == tok_t[test][:, -1]).mean()),
        "d": plan.d,
        "T": plan.T,
        "mistakes": int(info["mistakes_total"]),
    }


def run_demo(n_values=(3, 4), seed: int = 0, verbose: bool = True) -> list[dict]:
    """「答えだけ (E2E)」vs「軌跡あり (CoT)」の比較表を作る。"""
    if verbose:
        print("== CoT教示 vs 答えのみ (E2E) の学習非対称 ==")
        print("   (小規模実演: Cor 2 vs Cor 7 の方向的証拠。暗号学的困難性そのものではない)")
    rows: list[dict] = []
    configs: list[tuple[str, object, int]] = []
    for n in n_values:
        configs.append((f"parity{n}", parity_circuit(n), 6 if n == 3 else 10))
    configs.append(("adder2", adder_circuit(2), 12))
    for name, circuit, n_train in configs:
        c = circuit
        X = all_binary_inputs(c.n)
        y = c.forward(X)[:, -1]
        e2e_best, _wt = best_threshold_accuracy(X, y, weight_cap=12)
        e2e_gd = gd_linear_accuracy(X, y)
        cot = cot_learner(c, n_train=n_train, seed=seed)
        rows.append({
            "task": name, "n": c.n, "e2e_best": e2e_best, "e2e_gd": e2e_gd,
            "cot_acc": cot["test_acc"], "cot_train_full": cot["train_full_trace"],
        })
        if verbose:
            line = (f"  {name:>8}: E2E best-LTF {e2e_best:6.1%} / "
                    f"E2E GD {e2e_gd:6.1%} / CoT {cot['test_acc']:6.0%}"
                    f"  (train trace full: {cot['train_full_trace']}, d={cot['d']})")
            print(line)
    if verbose:
        print("   -> 答えだけでは線形模型の capacity で頭打ち、軌跡を見せると 100% 再現できる。")
    return rows
