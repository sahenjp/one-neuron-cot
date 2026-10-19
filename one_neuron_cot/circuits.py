"""σ-circuit DSL (arXiv:2610.17283 Sec. 2 の定義) とテスト用回路ビルダー。

σ-circuit: サイズ s の重み付き DAG。入力ノード x_1..x_n (深さ0) と
非入力ノード v_1..v_s (位相順) からなり、各ゲートは
    val(v_i) = sigma(theta_i + sum_(u->v_i) w(u, v_i) * val(u))
と計算する。threshold (Σ={0,1}) と relu (Σ=R, Assumption 1) をサポート。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTIVATIONS = ("threshold", "relu")

# ソース指定: ("x", q) = 入力 x_q, ("g", j) = ゲート v_j (j は自身より前), ("one",) = 定数 1
Src = tuple


@dataclass
class Gate:
    name: str
    inputs: tuple[tuple[Src, float], ...]
    bias: float


class Circuit:
    """ゲート列を位相順に保持する σ-circuit。出力ゲートは最後の v_s。"""

    def __init__(self, n: int, gates: list[Gate], activation: str = "threshold"):
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        self.n = int(n)
        self.activation = activation
        self.gates = list(gates)
        self._validate()

    def _validate(self) -> None:
        seen = set()
        for j, g in enumerate(self.gates):
            if g.name in seen:
                raise ValueError(f"duplicate gate name: {g.name}")
            seen.add(g.name)
            for src, _w in g.inputs:
                if src[0] == "x":
                    if not 0 <= src[1] < self.n:
                        raise ValueError(f"input index out of range: {src}")
                elif src[0] == "g":
                    if not 0 <= src[1] < j:
                        raise ValueError(f"edge must point to an earlier gate: {src} at {j}")
                elif src[0] != "one":
                    raise ValueError(f"unknown source: {src}")

    @property
    def s(self) -> int:
        return len(self.gates)

    def _act(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "threshold":
            return (z >= 0).astype(float)
        return np.maximum(z, 0.0)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """(m, n) の入力に対し各ゲート値 (m, s) を返す。"""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n:
            raise ValueError(f"X must have shape (m, {self.n})")
        m = X.shape[0]
        out = np.empty((m, self.s), dtype=float)
        for i, g in enumerate(self.gates):
            pre = np.full(m, float(g.bias))
            for src, w in g.inputs:
                if src[0] == "x":
                    pre += w * X[:, src[1]]
                elif src[0] == "g":
                    pre += w * out[:, src[1]]
                else:
                    pre += w
            out[:, i] = self._act(pre)
        return out

    def output(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[:, -1]

    def gate_index(self, name: str) -> int:
        for j, g in enumerate(self.gates):
            if g.name == name:
                return j
        raise KeyError(name)

    def magnitude_bound(self) -> float:
        """MC = sup |C_i(x)| の上界 (x in [0,1]^n / {0,1}^n)。

        threshold は値が {0,1} なので 1。relu は重みの絶対値で伝播する
        保守的な上界を使う (Assumption 1 の suppression 定数 A に使用)。
        """
        if self.activation == "threshold":
            return 1.0
        bounds = np.zeros(self.s)
        for i, g in enumerate(self.gates):
            b = abs(float(g.bias))
            for src, w in g.inputs:
                if src[0] == "x":
                    b += abs(w)  # |x_q| <= 1
                elif src[0] == "g":
                    b += abs(w) * bounds[src[1]]
            bounds[i] = b
        return float(bounds.max()) if self.s else 1.0


def _append(gates: list[Gate], name: str, inputs, bias: float) -> tuple:
    gates.append(Gate(name, tuple(inputs), float(bias)))
    return ("g", len(gates) - 1)


def _and(gates, a, b, name):
    return _append(gates, name, [(a, 1.0), (b, 1.0)], -2.0)


def _or(gates, a, b, name):
    return _append(gates, name, [(a, 1.0), (b, 1.0)], -1.0)


def _xor_gadget(gates, a, b, prefix):
    """threshold ゲート3つで a XOR b を計算する (o - 2h - 1 >= 0)。"""
    o = _or(gates, a, b, f"{prefix}_or")
    h = _and(gates, a, b, f"{prefix}_and")
    return _append(gates, f"{prefix}_xor", [(o, 1.0), (h, -2.0)], -1.0)


def parity_circuit(n: int) -> Circuit:
    """n ビットの XOR (パリティ)。threshold 専用 (s = 3(n-1))。"""
    gates: list[Gate] = []
    cur: tuple = ("x", 0)
    for i in range(1, n):
        cur = _xor_gadget(gates, cur, ("x", i), f"x{i}")
    return Circuit(n, gates, "threshold")


def adder_circuit(k: int) -> Circuit:
    """k ビットリプルキャリー加算器。入力 [a_0..a_{k-1}, b_0..b_{k-1}]。

    s = 2k+1 ゲート。出力ゲートは最上位桁 s_{k-1}。
    各 sum ゲートは CoT 上の指定ステップに出現する (Property 1)。
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    gates: list[Gate] = []
    and0 = _and(gates, ("x", 0), ("x", k), "and0")  # carry_1 = MAJ(a0, b0, 0)
    or0 = _or(gates, ("x", 0), ("x", k), "or0")
    _append(gates, "s0", [(or0, 1.0), (and0, -2.0)], -1.0)  # s0 = a0 XOR b0
    carry = and0
    for i in range(1, k):
        ai, bi = ("x", i), ("x", k + i)
        maj = _append(gates, f"c{i + 1}", [(ai, 1.0), (bi, 1.0), (carry, 1.0)], -2.0)
        _append(gates, f"s{i}", [(ai, 1.0), (bi, 1.0), (carry, 1.0), (maj, -2.0)], -1.0)
        carry = maj
    return Circuit(2 * k, gates, "threshold")


def majority_circuit(n: int) -> Circuit:
    """n ビット多数決 (単一ゲート: count >= floor(n/2)+1)。s = 1。"""
    inputs = [(("x", q), 1.0) for q in range(n)]
    gates = [Gate("maj", tuple(inputs), -(n // 2 + 1))]
    return Circuit(n, gates, "threshold")


def mlp_relu_circuit() -> Circuit:
    """小さなReLU MLP (n=3, 4隠れ+1出力 = s=5, Assumption 1)。"""
    W1 = [(1.0, -1.0, 0.5), (0.25, 0.5, -0.25), (-0.75, 1.0, 0.25), (0.5, 0.5, 0.5)]
    b1 = [0.0, -0.5, 0.5, -0.25]
    gates: list[Gate] = [
        Gate(f"h{q}", [(("x", q2), W1[q][q2]) for q2 in range(3)], b1[q])
        for q in range(4)
    ]
    W2 = (1.0, -0.5, 0.25, 0.75)
    out = Gate("y", [(("g", j), W2[j]) for j in range(4)], 0.1)
    gates.append(out)
    return Circuit(3, gates, "relu")


def random_threshold_circuit(n: int, s: int, seed: int = 0) -> Circuit:
    """決定論的なランダム threshold 回路 (fan-in 1..3, 小整数重み)。"""
    rng = np.random.default_rng(seed)
    wchoices = np.array([-2.0, -1.0, 1.0, 2.0])
    gates: list[Gate] = []
    for i in range(s):
        cands = [("x", q) for q in range(n)] + [("g", j) for j in range(i)]
        npar = int(rng.integers(1, 4))
        pick = rng.choice(len(cands), size=min(npar, len(cands)), replace=False)
        inputs = [(cands[p], float(wchoices[int(rng.integers(0, 4))])) for p in np.atleast_1d(pick)]
        gates.append(Gate(f"r{i}", tuple(inputs), float(rng.integers(-3, 4))))
    return Circuit(n, gates, "threshold")
