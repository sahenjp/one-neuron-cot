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

## クイックスタート (30秒)

```bash
git clone https://github.com/sahenjp/one-neuron-cot && cd one-neuron-cot
pip install numpy          # 唯一の依存
python3 run.py             # 5秒ツアーデモ
python3 run.py test        # 27件のユニットテスト
python3 run.py adder 3 110 011   # 3bit加算器を単一ニューロンのCoTで計算
python3 run.py parity 4 1011
python3 run.py verify      # 論文の主張を1件ずつ検証してレポート表示
python3 run.py distill     # 「答えだけ vs CoTあり」学習非対称のデモ
python3 run.py story 101   # ニューロン1個が気分プロンプトから一文を生成 (プロンプト0〜7で8通り)
```

ブラウザで試すなら: **[1ニューロンCPUプレイグラウンド](https://sahenjp.github.io/one-neuron-cot/)** (プログラム→回路→単一ニューロンの構成を可視化。pythonの正解トークン列と相互検証済み)。文章生成のみ試すなら**: [1ニューロン文章ジェネレータ](https://sahenjp.github.io/one-neuron-cot/generator.html)** (学習データ不要。重みは構成型で、温度つきソフトマックス標本抽出により入力した気分・話題から毎回違う文を生成)。

`python3 run.py` の出力:

```
[1/4] 3bit加算器を「ニューロン1個」で計算する
      回路: 7 ゲート -> ニューロン: 次元 d=1567, 生成長 T=446
      重みベクトルは 1 個だけ。同じベクトルを T 回呼んで回路を再現します。
      入力 x = 110011 (110 + 011)
      e2e 出力 = 0 = sum bit2 (110+011 = 1001, carry = 1)
[2/4] 全 64 入力で「回路 == ニューロン」を確認
      一致 64/64
[3/4] 学習: 教師のCoT軌跡だけから生徒ニューロンを学習 (Corollary 2)
      学習入力の全軌跡を完全再現し、未学習 2 入力の一致率 100%
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

## 検証ハーネス (他の理論論文への適用)

one_neuron_cot/harness.py は「主張1件 = 検証関数1本」を集めてレポートする最小エンジンで、
one_neuron_cot/claims.py が本論文の主張 (Theorem 2/3, Property 1, サイズ上限, Corollary 2, …) を登録する。
他の構成的証明の論文を検証するときは claims.py の Check 定義がそのままテンプレートになる。

```
$ python3 run.py verify
| check         | claim                          | reference   | result | detail |
| thm2-exact    | ARニューロン1個で回路を正確に再現   | Thm 2 / C.1 | PASS   | 3 circuits / all inputs; … |
| cor2-learning | CoT軌跡から効率的に学習可能        | Corollary 2 | PASS   | parity3: conv=True, held-out 100% |
**Summary: ALL PASS (7/7)**
```

(`run.py verify full` でより大きな回路まで検証)

## CoT教示 vs 「答えだけ」(学習非対称, Corollary 2 vs 7)

run.py distill: 同じ生徒構成に対し、教示を「最終ラベルのみ (E2E)」と「完全なCoT軌跡」で変えて比較する。

```
  parity3: E2E best-LTF  75.0% / E2E GD  50.0% / CoT   100%
  parity4: E2E best-LTF  68.8% / E2E GD  50.0% / CoT   100%
   adder2: E2E best-LTF  75.0% / E2E GD  50.0% / CoT   100%
```

E2Eの線形学習者は閾値関数クラスの表現力で頭打ちになる一方、軌跡を見せると未学習入力まで100%再現できる。
(小規模な方向的実演であり、一方向関数前提の暗号学的困難性そのものを実証するものではない)

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
run.py           # クイックデモ / テスト / verify / distill の入口
docs/      # プレイグラウンド + 文章ジェネレータ (依存なしHTML/JS, 自己検証つき)
requirements.txt
one_neuron_cot/
  sidon.py      # Erdős–Turán 型 Sidon/Golomb 集合 (Claim 1)
  circuits.py   # σ-circuit DSL + パリティ/加算器/ReLU-MLP/ランダム回路ビルダー
  simulate.py   # C.1 構成 (φ, w) + run_reference / run_fast
  story.py      # ニューロン1個で一文を生成するデモ (prompt 3bit -> 文)
  learn.py      # CoT教師ありパーセプトロン (Corollary 2 の実験版)
  demo.py       # CLIデモ
tests/test_one_neuron_cot.py
```

（論文PDF/全文テキストはローカル参照用でリポジトリには含めません）

既知の範囲（本テストAIのスコープ）: Boolean threshold 回路の完全シミュレーション、
ReLU回路の exact シミュレーション、次元/生成長の境界、CoT あり/なしの表現力差、
CoT 教師あり学習。未実装: Turing機械→回路のコンパイル (Corollary 1)、
近似シミュレーション (Theorem 5)、CIFAR-10 の教師・生徒実験 (Sec. 6)。
