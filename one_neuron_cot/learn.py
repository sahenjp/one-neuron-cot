"""Corollary 2 (CoT教師あり学習) の最小実験実装。

論文の設定: 教師 f* in F_{d,lin} の「完全なCoT軌跡」(x, f*^{CoT-T}(x)) から、
同じ phi パイプライン・次元 d を共有する単一ニューロン w を学習する。
学習則は (context_t, 次トークン) の各ペアに対する原点通過パーセプトロン。
教師 w* が全ペアを (正のマージンで) 線形分離するため、Novikoff の収束定理により
有限回の更新で収束する。
"""
from __future__ import annotations

import numpy as np

from .simulate import ArPlan, run_fast


def cot_training_set(plan: ArPlan, X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """教師の完全なCoT軌跡から (特徴ベクトル, 次トークン) ペアを作る。

    ステップ t の特徴ベクトルは右寄り内積の窓に一致する:
        F_t = concat(phi(x)[t-1:], g_1, ..., g_{t-1})  in R^d
    """
    tokens = run_fast(plan, X_train)  # (m, T)
    PHI = plan.phi(X_train)           # (m, d)
    m, T = tokens.shape
    d = plan.d
    feats = np.empty((m * T, d))
    labels = np.empty(m * T)
    row = 0
    for i in range(m):
        phi_row = PHI[i]
        for t in range(1, T + 1):
            head = d - (t - 1)
            feats[row, :head] = phi_row[t - 1:]
            if t > 1:
                feats[row, head:] = tokens[i, : t - 1]
            labels[row] = tokens[i, t - 1]
            row += 1
    return feats, labels


def fit_cot_perceptron(plan: ArPlan, X_train: np.ndarray,
                       max_epochs: int = 2000) -> tuple[np.ndarray, dict]:
    """共有重み w を原点通過パーセプトロンで学習する。

    境界の取り扱いは threshold 型 sigma(z)=1[z>=0] に合わせる:
      ラベル1: pre < 0 で更新 (+F)、ラベル0: pre >= 0 で更新 (-F)。
    返り値は (w, 情報辞書)。info["converged"] が True のとき
    学習サンプルの全軌跡を完全に再現する。
    """
    F, y = cot_training_set(plan, np.atleast_2d(X_train))
    is_one = y > 0.5
    w = np.zeros(plan.d)
    epochs = 0
    total = 0
    mistakes = 0
    converged = False
    while epochs < max_epochs:
        mistakes = 0
        for j in range(F.shape[0]):
            pre = float(w @ F[j])
            if is_one[j]:
                if pre < 0.0:
                    w += F[j]
                    mistakes += 1
            elif pre >= 0.0:
                w -= F[j]
                mistakes += 1
        total += mistakes
        epochs += 1
        if mistakes == 0:
            converged = True
            break
    info = {
        "converged": converged,
        "epochs": epochs,
        "mistakes_total": total,
        "final_epoch_mistakes": mistakes,
    }
    return w, info
