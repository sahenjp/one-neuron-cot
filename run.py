"""one-neuron-cot クイックデモ。

    python3 run.py              # 5秒ツアーデモ
    python3 run.py adder 3 110 011
    python3 run.py parity 4 1011
    python3 run.py test         # ユニットテスト全実行
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np

from one_neuron_cot.circuits import adder_circuit, parity_circuit
from one_neuron_cot.simulate import build_neuron, designated_values, run_fast


def _bar():
    print("=" * 64)


def tour():
    _bar()
    print("単一ニューロン自己回帰CoT デモ (arXiv:2610.17283 の検証実装)")
    _bar()

    print("\n[1/4] 3bit加算器を「ニューロン1個」で計算する")
    c = adder_circuit(3)
    plan = build_neuron(c)
    x = np.array([[1, 1, 0, 0, 1, 1]], dtype=float)   # 110 + 011
    tokens = run_fast(plan, x)
    des = designated_values(tokens, plan)
    print(f"      回路: {plan.s} ゲート -> ニューロン: 次元 d={plan.d}, 生成長 T={plan.T}")
    print("      重みベクトルは 1 個だけ。同じベクトルを T 回呼んで回路を再現します。")
    print(f"      入力 x = 110011 (110 + 011)")
    print(f"      指定ステップ出力 = {des[0].astype(int).tolist()}  (各ゲート値 C_i)")
    tot = 6 + 3
    print(f"      e2e 出力 = {int(tokens[0, -1])} = sum bit2 "
          f"(110+011 = {tot:04b}, carry = {tot >> 3})")

    print("\n[2/4] 全 64 入力で「回路 == ニューロン」を確認")
    from itertools import product
    X = np.array(list(product([0, 1], repeat=6)), dtype=float)
    ok = int((run_fast(plan, X)[:, -1] == c.forward(X)[:, -1]).sum())
    print(f"      一致 {ok}/64")

    print("\n[3/4] 学習: 教師のCoT軌跡だけから生徒ニューロンを学習 (Corollary 2)")
    from one_neuron_cot.learn import fit_cot_perceptron
    p = build_neuron(parity_circuit(3), bias_margin=0.5)
    X3 = np.array(list(product([0, 1], repeat=3)), dtype=float)
    rng = np.random.default_rng(0)
    perm = rng.permutation(8)
    train, test = perm[:6], perm[6:]
    w, info = fit_cot_perceptron(p, X3[train])
    tok_t = run_fast(p, X3)
    tok_s = run_fast(p.with_weights(w), X3)
    acc = float((tok_s[test][:, -1] == tok_t[test][:, -1]).mean())
    print(f"      学習データ: 8 入力中 6 個のCoT軌跡のみ (更新 {info['mistakes_total']} 回)")
    print(f"      学習入力の全軌跡を完全再現し、未学習 2 入力の一致率 {acc:.0%}")

    print("\n[4/4] テスト一式")
    _bar()
    print("  python3 run.py test    で 24 件の検証テストが動きます")
    print("  python3 run.py verify 論文主張のレポート / distill 学習非対称デモ / playground/ はブラウザ用")


def run_tests() -> int:
    r = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_one_neuron_cot", "-v"],
    )
    return r.returncode


def main(argv: list[str]) -> None:
    cmd = argv[0] if argv else "tour"
    if cmd in ("add", "adder"):
        from one_neuron_cot import demo
        demo.main(["adder", *argv[1:]])
    elif cmd == "parity":
        from one_neuron_cot import demo
        demo.main(["parity", *argv[1:]])
    elif cmd in ("test", "tests"):
        sys.exit(run_tests())
    elif cmd == "verify":
        from one_neuron_cot.harness import run_verify_paper
        summary = run_verify_paper(fast=not any(a == "full" for a in argv[1:]))
        print(summary["report"])
        sys.exit(0 if summary["all_passed"] else 1)
    elif cmd == "distill":
        from one_neuron_cot.asymmetry import run_demo
        rows = run_demo()
        ok = all(row["cot_acc"] == 1.0 for row in rows)
        sys.exit(0 if ok else 1)
    else:
        tour()


if __name__ == "__main__":
    main(sys.argv[1:])
