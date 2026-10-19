"""ニューロン1個で「文章」を生成するデモ。

プロンプト (3bit: 元気/晴れ/余裕) を入れると、s 個のゲートが
指定ステップで順に発火し、その生成列 (CoT) がそのまま一文に組み上がる。
内部ゲートの値も「途中の思考」としてログに表示される (Property 1)。
ニューロンは d 次元の重みベクトル 1 本だけ -- トークンを1個出すたびに
その同一ベクトルとの内積で次トークンを決めている (Theorem 2)。
"""
from __future__ import annotations

import numpy as np

from .circuits import Circuit, Gate
from .simulate import build_neuron, designated_values, run_fast


def build_story_circuit() -> Circuit:
    """3bit の気分プロンプトから「今日の見通し」ゲート列を組み立てる。"""
    gates: list[Gate] = []
    gates.append(Gate("genki", ((("x", 0), 1.0),), -0.5))
    gates.append(Gate("rain", ((("x", 1), -1.0),), 0.5))
    gates.append(Gate("and01", ((("x", 0), 1.0), (("x", 1), 1.0)), -2.0))
    gates.append(Gate("or01", ((("x", 0), 1.0), (("x", 1), 1.0)), -1.0))
    gates.append(Gate("and2", ((("x", 2), 1.0), (("g", 3), 1.0)), -2.0))
    gates.append(Gate("maj", ((("g", 2), 1.0), (("g", 4), 1.0)), -1.0))
    gates.append(Gate("or12", ((("x", 1), 1.0), (("x", 2), 1.0)), -1.0))
    gates.append(Gate("and12", ((("x", 1), 1.0), (("x", 2), 1.0)), -2.0))
    gates.append(Gate("spark", ((("g", 6), 1.0), (("g", 7), -2.0)), -1.0))
    gates.append(Gate("adventure", ((("g", 8), 1.0), (("x", 2), 1.0)), -2.0))
    return Circuit(3, gates, "threshold")


PHRASES = {
    "genki": ("やる気に満ちて", "少しゆったりとした"),
    "rain": ("空は今ひとつで", "空は澄みわたって"),
    "and01": ("外に誘われる", "部屋の安心感が勝る"),
    "or01": ("なにかが動き出しそう", "静かな流れに乗って"),
    "and2": ("選択肢が重なって", "優先順位はひとつに絞られ"),
    "maj": ("今日は転機の予感", "今日は小さな幸せ拾い"),
    "or12": ("可能性が広がり", "シンプルに絞り"),
    "and12": ("計画は加速し", "予定はゆるやかで"),
    "spark": ("好奇心がむくむく湧き", "いつものペースで"),
    "adventure": ("夜は冒険の予感で締めくくる", "夜は温かい飲み物で締めくくる"),
}


def generate(bits) -> dict:
    """プロンプト bits からニューロンの自己回帰生成で一文を作る。"""
    circuit = build_story_circuit()
    plan = build_neuron(circuit)
    tok = run_fast(plan, np.array([bits], dtype=float))
    vals = designated_values(tok, plan)[0].astype(int)
    gen_steps = plan.designated
    log, clauses = [], []
    for i, (g, val) in enumerate(zip(circuit.gates, vals)):
        phrase = PHRASES[g.name][1 - int(val)]
        log.append({"step": int(gen_steps[i]), "gate": g.name, "val": int(val), "phrase": phrase})
        clauses.append(("今日は" if i == 0 else "") + phrase)
    text = "、".join(clauses) + "。"
    return {
        "prompt": list(bits),
        "text": text,
        "log": log,
        "tokens": tok[0].astype(int).tolist(),
        "d": int(plan.d),
        "T": int(plan.T),
    }


PROMPT_NAMES = ("元気", "晴れ", "余裕")


def print_story(bits) -> None:
    r = generate(bits)
    tags = " / ".join(n + "=" + str(b) for n, b in zip(PROMPT_NAMES, r["prompt"]))
    print("プロンプト [" + tags + "]  ->  ニューロン (d=" + str(r["d"]) +
          ", T=" + str(r["T"]) + ") が自己回帰生成:")
    parts = []
    last = len(r["log"]) - 1
    for i, entry in enumerate(r["log"]):
        parts.append(entry["phrase"])
        mark = "1" if entry["val"] else "0"
        tail = "。" if i == last else "…"
        print("  [step " + str(entry["step"]) + "] " + entry["gate"] + "=" + mark
              + "  ->  " + "、".join(parts) + tail)
    print("生成文: " + r["text"])
    print()
