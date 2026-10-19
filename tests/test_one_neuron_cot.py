"""arXiv:2610.17283
"The Surprising Expressivity of a Single Neuron with Autoregressive Chain-of-Thought"
(Magen, Joshi, Vardi, 2026) の主張を検証するテスト。

検証対象:
- Theorem 2: サイズsのBoolean threshold回路を、次元 d=O(ns+s^2) の単一ニューロンが
  T=O(s^2) ステップの自己回帰生成で完全シミュレートする。
- Property 1: 指定ステップ t_i でゲート値 C_i(x) が生成され、他のステップは 0 (CoT保存)。
- Theorem 3 (exact): Assumption 1 の活性化 (ReLU) の実数値回路も同様にシミュレート。
- Section 5 / C.1: 次元・生成長の具体的境界 (d = 2T + kT', T = t_s、t_i/t'_j は Sidon 集合)。
- Joshi et al. (COLT 2025) との対比: CoTなしの単一thresholdでは tiny parity も計算不能。
- Corollary 2 (実験的): 完全なCoT軌跡からの教師あり学習 (パーセプトロン) で
  同一構造の生徒ニューロンが学習できる。
"""
from itertools import product

import numpy as np
import unittest

from one_neuron_cot.sidon import sidon_golomb
from one_neuron_cot.circuits import (
    Circuit,
    adder_circuit,
    mlp_relu_circuit,
    parity_circuit,
    random_threshold_circuit,
)
from one_neuron_cot.simulate import (
    build_neuron,
    designated_values,
    run_fast,
    run_reference,
)
from one_neuron_cot.learn import fit_cot_perceptron


def all_binary_inputs(n: int) -> np.ndarray:
    return np.array(list(product([0, 1], repeat=n)), dtype=float)


class TestSidonGolomb(unittest.TestCase):
    """C.1: Erdős–Turán 型 Sidon 集合 (= Golomb ruler) の構成。

    Requirement 1 (t_i + t'_j の一意性) と Requirement 2 (t_j - t_i の一意性) の
    基盤。大きさ m の集合が Sidon かつ Golomb で、最大値が O(m^2) ([1, 4m^2] 相当)。
    """

    def test_properties_small(self):
        for m in range(2, 21):
            a = sidon_golomb(m)
            self.assertEqual(len(a), m)
            self.assertTrue(all(a[i] < a[i + 1] for i in range(m - 1)),
                            f"not strictly increasing at m={m}: {a}")
            self.assertGreaterEqual(a[0], 1)
            # Golomb ruler: すべてのペア差が相異なる
            diffs = [a[j] - a[i] for i in range(m) for j in range(i + 1, m)]
            self.assertEqual(len(diffs), len(set(diffs)), f"diff collision at m={m}")
            # Sidon: すべてのペア和 (多重集合) が相異なる
            seen = {}
            for i in range(m):
                for j in range(i, m):
                    s = a[i] + a[j]
                    self.assertNotIn(s, seen, f"sum {s} collision at m={m}")
                    seen[s] = (i, j)
            # 論文の Claim 1 相当: a_m = O(m^2) (構成は [1, 4m^2] 程度に収まる)
            self.assertLessEqual(a[-1], 4 * m * m, f"a_m too large at m={m}")


class TestCircuitBuilders(unittest.TestCase):
    """sigma-circuit DSL (Sec. 2 の定義) の sanity check。"""

    def test_parity_matches_xor_of_bits(self):
        for n in (3, 4):
            c = parity_circuit(n)
            X = all_binary_inputs(n)
            out = c.forward(X)[:, -1]
            expected = X.sum(axis=1) % 2
            self.assertTrue((out == expected).all())

    def test_adder_msb_and_sum_bits(self):
        k = 3
        c = adder_circuit(k)
        X = all_binary_inputs(2 * k)
        a = X[:, :k]
        b = X[:, k:]
        total = (a @ (1 << np.arange(k)) + b @ (1 << np.arange(k))).astype(np.int64)
        out = c.forward(X)[:, -1]
        msb = ((total >> (k - 1)) & 1).astype(float)
        self.assertTrue((out == msb).all())
        for i in range(k):
            vals = c.forward(X)[:, c.gate_index(f"s{i}")]
            want = ((total >> i) & 1).astype(float)
            self.assertTrue((vals == want).all())

    def test_relu_mlp_matches_reference(self):
        c = mlp_relu_circuit()
        X = all_binary_inputs(3)
        got = c.forward(X)[:, -1]
        ref = mlp_relu_reference(X)
        self.assertTrue(np.allclose(got, ref, atol=1e-12))

    def test_random_threshold_circuit_deterministic(self):
        c1 = random_threshold_circuit(4, 6, seed=0)
        c2 = random_threshold_circuit(4, 6, seed=0)
        X = all_binary_inputs(4)
        np.testing.assert_array_equal(c1.forward(X), c2.forward(X))
        self.assertEqual(c1.forward(X).shape, (16, 6))


def mlp_relu_reference(X: np.ndarray) -> np.ndarray:
    W1 = np.array([[1.0, -1.0, 0.5], [0.25, 0.5, -0.25],
                   [-0.75, 1.0, 0.25], [0.5, 0.5, 0.5]])
    b1 = np.array([0.0, -0.5, 0.5, -0.25])
    W2 = np.array([1.0, -0.5, 0.25, 0.75])
    b2 = 0.1
    h = np.maximum(X @ W1.T + b1, 0.0)
    return np.maximum(h @ W2 + b2, 0.0)


class TestTheorem2ThresholdSimulation(unittest.TestCase):
    """Theorem 2: Boolean threshold 回路の単一ニューロン AR シミュレーション。"""

    def _check_exact(self, circuit):
        plan = build_neuron(circuit)
        X = all_binary_inputs(circuit.n)
        tokens = run_fast(plan, X)
        self.assertEqual(tokens.shape, (2 ** circuit.n, plan.T))
        # トークンは Sigma = {0, 1}
        self.assertTrue(((tokens == 0) | (tokens == 1)).all())
        # e2e: 最終生成トークンが回路出力と完全一致
        e2e = tokens[:, -1]
        want = circuit.forward(X)[:, -1]
        self.assertTrue((e2e == want).all(),
                        f"e2e mismatch for n={circuit.n}")

    def test_parity3(self):
        self._check_exact(parity_circuit(3))

    def test_parity4(self):
        self._check_exact(parity_circuit(4))

    def test_adder2(self):
        self._check_exact(adder_circuit(2))

    def test_adder3(self):
        self._check_exact(adder_circuit(3))

    def test_random_circuit(self):
        self._check_exact(random_threshold_circuit(8, 12, seed=0))


class TestProperty1CotPreservation(unittest.TestCase):
    """Property 1: t_i で C_i(x)、それ以外は 0 が生成される。"""

    def _check_cot(self, circuit):
        plan = build_neuron(circuit)
        X = all_binary_inputs(circuit.n)
        tokens = run_fast(plan, X)
        gate_vals = circuit.forward(X)  # (m, s)
        designated = designated_values(tokens, plan)  # (m, s)
        np.testing.assert_array_equal(designated, gate_vals)
        t_arr = np.array(plan.designated) - 1
        mask = np.ones(plan.T, dtype=bool)
        mask[t_arr] = False
        self.assertTrue((tokens[:, mask] == 0).all())

    def test_parity3(self):
        self._check_cot(parity_circuit(3))

    def test_adder3_all_sum_bits_in_cot(self):
        c = adder_circuit(3)
        plan = build_neuron(c)
        X = all_binary_inputs(6)
        a = X[:, :3]
        b = X[:, 3:]
        total = (a @ (1 << np.arange(3)) + b @ (1 << np.arange(3))).astype(np.int64)
        des = designated_values(run_fast(plan, X), plan)  # (m, s)
        for i in range(3):
            col = des[:, c.gate_index(f"s{i}")]
            want = ((total >> i) & 1).astype(float)
            self.assertTrue((col == want).all(),
                            f"CoT does not carry sum bit {i}")


class TestTheorem3ReluSimulation(unittest.TestCase):
    """Theorem 3 (exact): Assumption 1 を満たす ReLU の実数値回路。"""

    def test_relu_mlp_exact_simulation(self):
        c = mlp_relu_circuit()
        plan = build_neuron(c)
        X = all_binary_inputs(3)
        tokens = run_fast(plan, X)
        gate_vals = c.forward(X)
        self.assertTrue(np.allclose(tokens[:, -1], gate_vals[:, -1], atol=1e-9))
        designated = designated_values(tokens, plan)
        self.assertTrue(np.allclose(designated, gate_vals, atol=1e-9))
        t_arr = np.array(plan.designated) - 1
        mask = np.ones(plan.T, dtype=bool)
        mask[t_arr] = False
        self.assertTrue(np.allclose(tokens[:, mask], 0.0, atol=1e-9))


class TestBoundsDimensionAndLength(unittest.TestCase):
    """次元 d=O(ns+s^2) / 生成長 T=O(s^2) の具体的境界 (Sec. 5, C.1)。"""

    def test_bounds_hold(self):
        circuits = [
            parity_circuit(3),
            parity_circuit(4),
            adder_circuit(2),
            adder_circuit(3),
            random_threshold_circuit(8, 12, seed=0),
        ]
        rows = []
        for c in circuits:
            plan = build_neuron(c)
            n, s = plan.n, plan.s
            # C.1: d = n' = 2T + kT'
            self.assertEqual(plan.d, 2 * plan.T + plan.k * plan.t_prime)
            self.assertEqual(plan.designated, sorted(plan.designated))
            self.assertEqual(plan.designated[-1], plan.T)
            self.assertLessEqual(plan.T, 16 * s * s,
                                 f"generation length not O(s^2): T={plan.T}, s={s}")
            self.assertLessEqual(plan.d, 100 * (n * s + s * s),
                                 f"dimension not O(ns+s^2): d={plan.d}")
            x0 = all_binary_inputs(n)[:3]
            phi = plan.phi(x0)
            self.assertEqual(phi.shape[1], plan.d)
            self.assertTrue(((phi == 0) | (phi == 1)).all())
            rows.append((n, s, plan.T, plan.d, plan.T / (s * s),
                         plan.d / (n * s + s * s)))
        print("\n[n, s, T, d, T/s^2, d/(ns+s^2)]")
        for r in rows:
            print("  ", tuple(round(v, 3) for v in r))


class TestReferenceVsFastSimulator(unittest.TestCase):
    """素朴な右寄り内積シミュレータと高速実装が同一の軌跡を出す。"""

    def test_trajectories_identical(self):
        c = adder_circuit(2)
        plan = build_neuron(c)
        X = all_binary_inputs(4)[:5]
        for x in X:
            t_ref = run_reference(plan, x)
            t_fast = run_fast(plan, x[None, :])[0]
            np.testing.assert_array_equal(t_ref, t_fast)


class TestNoCotExpressivityGap(unittest.TestCase):
    """CoT なしの単一 threshold では parity3 が計算できない (Joshi+25 との対比)。

    n=3 の任意の LTF は |w_i| <= 12 の整数表現を持つ (Muroga 型の重み上限) ので、
    その範囲の全列挙で theta を消去すれば厳密な非表現性の証明になる。
    """

    def test_parity3_is_not_a_threshold_function(self):
        X = all_binary_inputs(3)
        par = (X.sum(axis=1) % 2).astype(bool)
        ws = np.arange(-12, 13)
        W = np.array(np.meshgrid(*([ws] * 3), indexing="ij")).reshape(3, -1).T
        pre = X @ W.T  # (8, nW)
        pos = pre[par]    # パリティ1の入力行
        neg = pre[~par]   # パリティ0の入力行
        theta_lo = (-pos).max(axis=0)  # pos 行: pre + theta >= 0
        theta_hi = (-neg).min(axis=0)  # neg 行: pre + theta < 0
        feasible = theta_lo < theta_hi
        self.assertFalse(bool(feasible.any()),
                         "parity3 が threshold で表現できてしまった")

    def test_ar_neuron_with_cot_computes_parity3(self):
        c = parity_circuit(3)
        plan = build_neuron(c)
        X = all_binary_inputs(3)
        tokens = run_fast(plan, X)
        want = X.sum(axis=1) % 2
        self.assertTrue((tokens[:, -1] == want).all())


class TestCotLearnability(unittest.TestCase):
    """Corollary 2 (実験版): 完全なCoT軌跡からの教師あり学習。

    教師 = 構成された AR threshold ニューロン (F_{d,lin} の要素)。
    生徒 = 同一 phi パイプライン・次元 d の共有重みニューロンで,
    (context, next-token) ペアへのパーセプトロン (原点通過) で学習する。
    """

    def test_parity3_teacher_student(self):
        plan = build_neuron(parity_circuit(3))
        X = all_binary_inputs(3)
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(X))
        train, test = perm[:6], perm[6:]
        w, info = fit_cot_perceptron(plan, X[train])
        self.assertTrue(info["converged"],
                        f"perceptron did not converge: {info}")
        student = plan.with_weights(w)
        tok_teacher = run_fast(plan, X)
        tok_student = run_fast(student, X)
        # 学習入力上では完全な軌跡再現 (帰納的に保証される)
        np.testing.assert_array_equal(tok_student[train], tok_teacher[train])
        acc = float((tok_student[test][:, -1] == tok_teacher[test][:, -1]).mean())
        print(f"\n[learn parity3] test e2e agreement = {acc:.2f}, info={info}")
        self.assertGreaterEqual(acc, 0.5)

    def test_adder2_teacher_student(self):
        plan = build_neuron(adder_circuit(2))
        X = all_binary_inputs(4)
        rng = np.random.default_rng(1)
        perm = rng.permutation(len(X))
        train, test = perm[:12], perm[12:]
        w, info = fit_cot_perceptron(plan, X[train])
        self.assertTrue(info["converged"], f"perceptron did not converge: {info}")
        student = plan.with_weights(w)
        tok_teacher = run_fast(plan, X)
        tok_student = run_fast(student, X)
        np.testing.assert_array_equal(tok_student[train], tok_teacher[train])
        acc = float((tok_student[test][:, -1] == tok_teacher[test][:, -1]).mean())
        print(f"\n[learn adder2] test e2e agreement = {acc:.2f}, info={info}")
        self.assertGreaterEqual(acc, 0.5)


if __name__ == "__main__":
    unittest.main()
