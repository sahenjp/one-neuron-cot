"""単一ニューロン自己回帰CoTのデモ。

例:
    python3 -m one_neuron_cot.demo adder 3 110 011   # 3bit加算器: 110 + 011 = 1001
    python3 -m one_neuron_cot.demo parity 4 1011     # 4bitパリティ: 1011 -> 1
"""
from __future__ import annotations

import sys

import numpy as np

from .circuits import adder_circuit, parity_circuit
from .simulate import build_neuron, designated_values, run_fast


def _bits(arg: str, k: int, what: str) -> list[int]:
    vals = [int(c) for c in arg]
    if len(vals) != k or any(v not in (0, 1) for v in vals):
        raise SystemExit(f"{what} must be {k} bits of 0/1, got: {arg!r}")
    return vals


def main(argv: list[str]) -> None:
    kind = argv[0] if argv else "parity"
    if kind == "adder":
        k = int(argv[1]) if len(argv) > 1 else 3
        a = _bits(argv[2] if len(argv) > 2 else "110", k, "operand a")
        b = _bits(argv[3] if len(argv) > 3 else "011", k, "operand b")
        circuit = adder_circuit(k)
        x = np.array([a + b], dtype=float)   # [a0..a_{k-1}, b0..b_{k-1}]
        total = sum(v << i for i, v in enumerate(a)) + sum(v << i for i, v in enumerate(b))
        # 出力ゲート s_{k-1} = sum の第 k-1 ビット (桁上がりは CoT の c_k ゲートに現れる)
        expect = f"sum bit{k - 1} = {(total >> (k - 1)) & 1}  (a+b = {total}, carry = {total >> k})"
    elif kind == "parity":
        n = int(argv[1]) if len(argv) > 1 else 4
        row = _bits(argv[2] if len(argv) > 2 else "1011"[:n], n, "input")
        circuit = parity_circuit(n)
        x = np.array([row], dtype=float)
        expect = f"parity = {sum(row) % 2}"
    else:
        raise SystemExit("usage: demo.py {adder k A B | parity n BITS}")

    plan = build_neuron(circuit)
    tokens = run_fast(plan, x)
    des = designated_values(tokens, plan)

    print(f"回路: {kind} (n={plan.n}, s={plan.s}), 次元 d={plan.d}, 生成長 T={plan.T}")
    print(f"入力 x  = {''.join(str(int(v)) for v in x[0])}")
    print(f"CoT     = {''.join(tokens[0].astype(int).astype(str))}")
    print(f"指定出力 = {des[0].astype(int).tolist()}  (各ゲート値 C_i)")
    print(f"最終出力 = {int(tokens[0, -1])}  (期待: {expect})")


if __name__ == "__main__":
    main(sys.argv[1:])
