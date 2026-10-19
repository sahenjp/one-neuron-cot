# one_neuron_cot — 単一ニューロン自己回帰CoT の検証テストAI

arXiv:[2610.17283](https://arxiv.org/abs/2610.17283)
**"The Surprising Expressivity of a Single Neuron with Autoregressive Chain-of-Thought"**
(Roey Magen, Nirmit Joshi, Gal Vardi, 2026) の主張を、numpyのみで実装・検証するテストスイート。

**ニューロン1個を自己回帰的に呼ぶだけ**で、任意の σ-circuit（threshold/ReLU回路）が
Chain-of-Thought 生成として完全シミュレートできる、という表現力の結果。
論文 Appendix C.1 の exact simulation の構成（Sidon 集合による serialization）を
そのままコードに落としている。

> **このリポジトリについて**: 論文の主張を検証するための非公式・独立実装です
> （著者の公式コードは 2026-10 時点の簡易検索では見当たりませんでした）。
> 論文: Magen, Joshi, Vardi, *"The Surprising Expressivity of a Single Neuron
> with Autoregressive Chain-of-Thought"*, arXiv:2610.17283 (CC BY 4.0)。
> 引用する場合:
>
> ```bibtex
> @misc{magen2026surprising,
>   title={The Surprising Expressivity of a Single Neuron with Autoregressive Chain-of-Thought},
>   author={Magen, Roey and Joshi, Nirmit and Vardi, Gal},
>   year={2026},
>   eprint={2610.17283},
>   archivePrefix={arXiv}
> }
> ```

## 実行方法

```bash
python3 -m unittest tests.test_one_neuron_cot     # 19 tests
python3 -m one_neuron_cot.demo adder 3 110 011    # デモ
python3 -m one_neuron_cot.demo parity 4 1011
```

## 論文の主張 ↔ テスト対応表

| # | 論文の主張 | テスト | 結果 |
|---|---|---|---|
| 1 | Claim 1 ([ET41]): 大きさ m の Sidon 集合 (= Golomb ruler) が O(m^2) で作れる | `TestSidonGolomb` (m=2..20 を全列挙検証) | PASS |
| 2 | Theorem 2: サイズ s の Boolean threshold 回路が d=O(ns+s^2)・T=O(s^2) の単一ニューロンで完全シミュレート | `TestTheorem2ThresholdSimulation` (パリティ3/4, 加算器2/3bit, ランダム回路を全入力で網羅) | PASS |
| 3 | Property 1: 指定ステップ t_i で C_i(x)、それ以外は 0 が生成 (CoT保存) | `TestProperty1CotPreservation` | PASS |
| 4 | Theorem 3 (exact): Assumption 1 の ReLU 回路も同様 (実数値) | `TestTheorem3ReluSimulation` (3→4→1 ReLU MLP) | PASS (atol 1e-9) |
| 5 | C.1: d = 2T + kT', T = t_s、t_i+t'_j / t_j−t_i の衝突なし | `TestBoundsDimensionAndLength` | PASS |
| 6 | Joshi et al. (COLT 2025) との対比: CoT なしの単一 threshold では計算不能 | `TestNoCotExpressivityGap` (parity3 が LTF でないことを重み全列挙で証明) | PASS |
| 7 | Corollary 2: 完全なCoT軌跡からの効率的学習 | `TestCotLearnability` (原点通過パーセプトロン) | PASS (未学習入力一致率 100%) |

実測のスケーリング（`TestBoundsDimensionAndLength` の出力）:

```
[n, s, T, d, T/s^2, d/(ns+s^2)]
   (3, 6, 285, 959, 7.917, 17.759)
   (4, 9, 609, 2081, 7.519, 17.786)
   (4, 5, 180, 619, 7.2, 13.756)
   (6, 7, 446, 1567, 9.102, 17.22)
   (8, 12, 1334, 4633, 9.264, 19.304)
```

T = O(s^2)・d = O(ns+s^2) が定数倍込みで確認できる。

## 実装ノート (Appendix C.1 の構成)

- 右寄り内積: ⟨w, ctx⟩ はコンテキスト末尾 d 座標を左ゼロパディングして内積。
  新しいトークンほど w の右端に整列する（時間不変・パラメータ共有）。
- 重みブロック: `w = [v(T) | z_k(T') … z_1(T') | z(T)]`、
  特徴マップ: `φ(x) = [0_{T−1} | 1 | u_k … u_1 | 0_T]`。
  - `v[−t]`: 指定ステップ t_i でゲート i のバイアス、それ以外は c_σ − A − 1（抑制）
  - `u_r[−t'_j] = x_r[j]`（入力を s 個ずつのブロックに分割して配置）
  - `z_r[−(t_i+t'_j−1)]`: 入力→ゲート重み（**Requirement 1**: t_i+t'_j が一意 → Sidon）
  - `z[−(t_j−t_i)]`: ゲート間重み（**Requirement 2**: t_j−t_i が一意 → Golomb）
- 指定ステップは `t_i = a_{s+i}`、入力位置は `t'_j = a_j`（`sidon_golomb(2s)`）。
  Sidon/Golomb 性が重みの衝突なし配置を保証する（論文の主技術 contributions）。
- シミュレータは2実装: `run_reference`（素朴な右寄り内積・1入力）と
  `run_fast`（非零係数に絞ったバッチ行列積）。全軌跡が一致することをテスト済み。

## 構成

```
one_neuron_cot/
  sidon.py      # Erdős–Turán 型 Sidon/Golomb 集合 (Claim 1)
  circuits.py   # σ-circuit DSL + パリティ/加算器/ReLU-MLP/ランダム回路ビルダー
  simulate.py   # C.1 構成 (φ, w) + run_reference / run_fast
  learn.py      # CoT教師ありパーセプトロン (Corollary 2 の実験版)
  demo.py       # CLIデモ
tests/test_one_neuron_cot.py
```

（論文PDF/全文テキストはローカル参照用でリポジトリには含めません）

既知の範囲（本テストAIのスコープ）: Boolean threshold 回路の完全シミュレーション、
ReLU回路の exact シミュレーション、次元/生成長の境界、CoT あり/なしの表現力差、
CoT 教師あり学習。未実装: Turing機械→回路のコンパイル (Corollary 1)、
近似シミュレーション (Theorem 5)、CIFAR-10 の教師・生徒実験 (Sec. 6)。
