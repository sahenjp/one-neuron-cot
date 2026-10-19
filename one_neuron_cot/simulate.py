"""Appendix C.1 の exact simulation をそのまま実装する。

単一ニューロン f_w(x) = sigma(<w, x>) を自己回帰的に適用する。
<w, x> は右寄り (right-aligned) 内積: 短い方を左にゼロパディングして
内積を取る (論文の Sec. 2 "padding the shorter vector with zeros on the left")。

レイアウト (負インデックス = 右端からの距離 1..len):
    w   = [ v (T) | z_k (T') ... z_1 (T') | z (T) ]
    phi = [ 0_{T-1} | 1 | u_k ... u_1 | 0_T ]

- v[-t]      : t = t_i でゲート i のバイアス theta_i, それ以外は c_sigma - A - 1
- u_r[-m]    : m = t'_j で入力ブロック値 x_r[j], それ以外 0
- z_r[-l]    : l = t_i + t'_j - 1 で w_in[(r-1)s+j, i] (Requirement 1: 和が一意)
- z[-Delta]  : Delta = t_j - t_i (j > i) で w[i, j] (Requirement 2: 差が一意)
- t_i = a_{s+i}, t'_j = a_j  (a = sidon_golomb(2s)), T = t_s, T' = T + t'_s

生成: コンテキスト = [phi(x), g_1, ..., g_{t-1}] の末尾 d 座標の右寄り内積。
ステップ t では phi の "1" が v[-t] に、u_r 値が z_r[-(t + t'_j - 1)] に、
生成済みトークン g_tau が z[-(t - tau)] に整列する。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .circuits import Circuit
from .sidon import sidon_golomb

# Assumption 1 の零領域: sigma(z) = 0 for z <= c_sigma
C_SIGMA = {"threshold": -1.0, "relu": 0.0}


def _act(activation: str, z):
    if activation == "threshold":
        z = np.asarray(z)
        return (z >= 0).astype(float)
    return np.maximum(z, 0.0)


def _act_scalar(activation: str, z: float) -> float:
    if activation == "threshold":
        return 1.0 if z >= 0 else 0.0
    return z if z > 0 else 0.0


@dataclass
class ArPlan:
    """構成済みの単一ニューロン phi/w と高速シミュレーション用の表。"""

    circuit: Circuit
    n: int
    s: int
    activation: str
    d: int
    T: int
    t_prime: int
    k: int
    designated: list[int]
    w: np.ndarray
    phi_template: np.ndarray
    u_slots: list[tuple[int, int]]
    v_term: np.ndarray
    input_coef: np.ndarray
    z_steps: list[tuple[np.ndarray, np.ndarray]]

    def phi(self, X: np.ndarray) -> np.ndarray:
        """phi_{n,s}(x) を適用する (Theorem 2/3 の feature map)。"""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.n:
            raise ValueError(f"X must have {self.n} columns")
        out = np.tile(self.phi_template, (X.shape[0], 1)).astype(float)
        for idx, q in self.u_slots:
            out[:, idx] = X[:, q]
        return out

    def with_weights(self, w: np.ndarray) -> "ArPlan":
        """同じ phi/スケジュールで重みだけ差し替えた plan (学習された生徒用)。"""
        w = np.asarray(w, dtype=float)
        if w.shape != (self.d,):
            raise ValueError(f"w must have shape ({self.d},)")
        return replace(self, w=w)


def build_neuron(circuit: Circuit, bias_margin: float | None = None) -> ArPlan:
    """C.1 の構成で (phi_{n,s}, w) を作る (Theorem 2 / Theorem 3-exact)。

    bias_margin は threshold 用の学習補助: 指定ステップのバイアスを
    margin だけ持ち上げる。threshold 回路の preactivation は整数値で
    隙間が 1 以上あるため、0 <= margin < 0.5 なら全入力で挙動は不変
    (ラベル1の preact=0 の境界を解消しパーセプトロンが収束する)。
    """
    n, s, activation = circuit.n, circuit.s, circuit.activation
    if s < 1:
        raise ValueError("circuit must have at least one gate")
    if bias_margin is None:
        bias_margin = 0.5 if activation == "threshold" else 0.0
    if activation != "threshold" and bias_margin != 0.0:
        raise ValueError("bias_margin is threshold-only")

    a = sidon_golomb(2 * s)
    t = a[s:2 * s]        # t_i, i = 1..s (designated steps)
    tprime = a[:s]        # t'_j, j = 1..s (input locations)
    T = t[-1]             # t_s = T (Property 1: 最後の指定ステップが最終ステップ)
    t_prime = T + tprime[-1]  # T' >= max t_i + t'_j
    k = (n + s - 1) // s  # ceil(n / s)
    d = 2 * T + k * t_prime

    # 重みの抽出
    win = np.zeros((n, s))      # win[q, i] = w(x_q -> v_i)
    gate_w = np.zeros((s, s))   # gate_w[j, i] = w(v_j -> v_i), j < i
    for i, g in enumerate(circuit.gates):
        for src, wgt in g.inputs:
            if src[0] == "x":
                win[src[1], i] = wgt
            elif src[0] == "g":
                gate_w[src[1], i] = wgt

    # --- z_r ブロック: distance l = t_i + t'_j - 1 (Requirement 1) ---
    zr = [dict() for _ in range(k)]
    for r in range(k):
        for j in range(s):
            q = r * s + j
            for i in range(s):
                coef = win[q, i] if q < n else 0.0
                if coef == 0.0:
                    continue
                ell = t[i] + tprime[j] - 1
                if ell in zr[r]:
                    raise AssertionError("sum collision (Requirement 1 violated)")
                zr[r][ell] = coef

    # --- z ブロック: distance Delta = t_j - t_i, j > i (Requirement 2) ---
    z = {}
    for i in range(s):
        for j in range(i + 1, s):
            coeff = gate_w[i, j]
            if coeff == 0.0:
                continue
            delta = t[j] - t[i]
            if delta in z:
                raise AssertionError("difference collision (Requirement 2 violated)")
            z[delta] = coeff

    # suppression 定数 A = sum_r ||z_r||_1 + MC ||z||_1
    mc = circuit.magnitude_bound()
    norm_zr = sum(abs(c) for blk in zr for c in blk.values())
    norm_z = sum(abs(c) for c in z.values())
    big = C_SIGMA[activation] - (norm_zr + mc * norm_z) - 1.0

    # --- v ブロック: v[-t] ---
    des_set = {}
    for i, ti in enumerate(t):
        des_set[ti] = i
    v_term = np.empty(T)
    for tau in range(1, T + 1):
        if tau in des_set:
            v_term[tau - 1] = circuit.gates[des_set[tau]].bias + bias_margin
        else:
            v_term[tau - 1] = big

    # --- w の組み立て (右端 = z ブロック) ---
    w = np.zeros(d)
    for tau in range(1, T + 1):          # v[-t] は左端ブロック: idx T - t
        w[T - tau] = v_term[tau - 1]
    for r in range(k):
        for ell, coeff in zr[r].items():
            idx = d - (T + r * t_prime + ell)
            w[idx] = coeff
    for delta, coeff in z.items():
        w[d - delta] = coeff

    # --- phi テンプレート ([0_{T-1}, 1, u_k..u_1, 0_T]) と u スロット ---
    phi_template = np.zeros(d)
    phi_template[d - (T + k * t_prime + 1)] = 1.0
    u_slots = []
    for r in range(k):
        for j in range(s):
            q = r * s + j
            if q >= n:
                continue
            idx = d - (T + r * t_prime + tprime[j])
            u_slots.append((idx, q))

    # --- 高速シミュレーション用テーブル ---
    # input_coef[(r, j), t-1] = z_r[-(t + t'_j - 1)]
    input_coef = np.zeros((k * s, T))
    for r in range(k):
        for j in range(s):
            q = r * s + j
            for i in range(s):
                if q < n and win[q, i] != 0.0:
                    input_coef[r * s + j, t[i] - 1] = win[q, i]
    # ステップ tau での伝播項: tau_idx = tau - Delta (0-based g 列)
    z_steps = []
    deltas = np.array(sorted(z), dtype=int)
    coefs_all = np.array([z[dd] for dd in deltas], dtype=float)
    for tau in range(1, T + 1):
        mask = deltas < tau
        taus = (tau - deltas[mask]).astype(int) - 1
        z_steps.append((taus, coefs_all[mask]))

    return ArPlan(
        circuit=circuit, n=n, s=s, activation=activation, d=d, T=T,
        t_prime=t_prime, k=k, designated=list(t), w=w,
        phi_template=phi_template, u_slots=u_slots, v_term=v_term,
        input_coef=input_coef, z_steps=z_steps,
    )


def run_fast(plan: ArPlan, X: np.ndarray) -> np.ndarray:
    """バッチ自己回帰生成。X: (m, n) -> tokens: (m, T)。

    run_reference と同じ右寄り内積の因果則を、非零係数だけを集めた
    行列積で計算する (値は完全一致する)。
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[1] != plan.n:
        raise ValueError(f"X must have {plan.n} columns")
    m = X.shape[0]
    T = plan.T
    inputs_term = np.zeros((m, T))
    if plan.k * plan.s:
        Xfeat = np.zeros((m, plan.k * plan.s))
        Xfeat[:, : plan.n] = X
        inputs_term = Xfeat @ plan.input_coef
    G = np.zeros((m, T))
    for tau in range(1, T + 1):
        pre = plan.v_term[tau - 1] + inputs_term[:, tau - 1]
        taus, coff = plan.z_steps[tau - 1]
        if taus.size:
            pre = pre + G[:, taus] @ coff
        G[:, tau - 1] = _act(plan.activation, pre)
    return G


def run_reference(plan: ArPlan, x: np.ndarray) -> np.ndarray:
    """素朴な実装: コンテキストを右寄りパディングして全長内積 (1入力)。"""
    x = np.asarray(x, dtype=float).reshape(1, -1)
    ctx = plan.phi(x)[0]  # (d,)
    tokens = np.empty(plan.T)
    for tau in range(1, plan.T + 1):
        # コンテキスト [phi, g_1..g_{t-1}] の末尾 d 座標 = phi[t-1:] + g_1..g_{t-1}
        if tau == 1:
            window = ctx
        else:
            window = np.concatenate([ctx[tau - 1:], tokens[: tau - 1]])
        pre = float(plan.w @ window)
        tokens[tau - 1] = _act_scalar(plan.activation, pre)
    return tokens


def designated_values(tokens: np.ndarray, plan: ArPlan) -> np.ndarray:
    """指定ステップ t_i の生成値 (m, s) を取り出す (Property 1)。"""
    cols = [ti - 1 for ti in plan.designated]
    return tokens[:, cols]
