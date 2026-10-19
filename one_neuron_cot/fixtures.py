"""プレイグラウンド用フィクスチャ (python 構成による正解トークン列) の出力。

    python3 -m one_neuron_cot.fixtures   # playground/fixtures.js を再生成

JS 側の再実装 (playground/app.js) がこのトークン列と完全一致するか
playground/selftest.js (node) で検証できる。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .circuits import adder_circuit, majority_circuit, parity_circuit
from .simulate import build_neuron, run_fast

SAMPLE_BITS = {
    ("parity", 4): [[1, 0, 1, 1], [0, 0, 1, 1], [1, 1, 1, 0]],
    ("adder", 3): [[1, 0, 1, 1, 0, 1], [0, 1, 1, 0, 0, 1], [1, 1, 0, 0, 1, 1]],
    ("maj", 5): [[1, 0, 1, 0, 1], [0, 0, 1, 1, 0], [1, 1, 1, 0, 0]],
}

_BUILDERS = {
    "parity": parity_circuit,
    "adder": adder_circuit,
    "maj": majority_circuit,
}


def build_fixtures() -> dict:
    programs = []
    for (kind, n), samples in SAMPLE_BITS.items():
        circuit = _BUILDERS[kind](n)
        plan = build_neuron(circuit)
        rows = []
        for bits in samples:
            tok = run_fast(plan, np.array([bits], dtype=float))
            rows.append({"bits": list(bits), "tokens": [int(v) for v in tok[0]]})
        programs.append({
            "kind": kind,
            "param": int(n),
            "n": int(plan.n),
            "s": int(plan.s),
            "d": int(plan.d),
            "T": int(plan.T),
            "k": int(plan.k),
            "designated": [int(t) for t in plan.designated],
            "samples": rows,
        })
    return {"generated": time.strftime("%Y-%m-%d"), "programs": programs}


def main() -> None:
    data = build_fixtures()
    out = Path(__file__).resolve().parent.parent / "playground" / "fixtures.js"
    out.parent.mkdir(exist_ok=True)
    js = ("(typeof window !== 'undefined' ? window : globalThis).PLAYGROUND_FIXTURES = "
          + json.dumps(data, separators=(",", ":")) + "\n")
    js += "if (typeof module !== 'undefined') module.exports = globalThis.PLAYGROUND_FIXTURES;\n"
    out.write_text(js, encoding="utf-8")
    print("wrote " + str(out) + " (" + str(out.stat().st_size) + " bytes, "
          + str(len(data["programs"])) + " programs)")


if __name__ == "__main__":
    main()
