"""Erdős–Turán 型の Sidon 集合 (= Golomb ruler) の構成。

arXiv:2610.17283 の Appendix C.1 (Claim 1, [ET41]) に対応する部品。
集合 {a_1 < ... < a_m} が次の性質を持つことを tests 側で全列挙により検証する:

- Sidon: a_i + a_j (= a_k + a_l, i <= j, k <= l) ならば {i, j} = {k, l}
  → Requirement 1 (t_i + t'_j の一意性) の基盤
- Golomb ruler: i < j なるペア差 a_j - a_i がすべて相異なる
  (Observation 3: Sidon <=> Golomb) → Requirement 2 (t_j - t_i の一意性) の基盤
-最大値 a_m <= 4 m^2 (Claim 1 の [1, 4n^2] 境界相当)

構成: p を m 以上の最小素数, B = 2m - 1 とし
  f(x) = x + B * (x^2 mod p),   x in {0, ..., m-1}
(x, x^2) が F_p 上で Sidon であることを、B が整数のcarryなし復号
(x 和 < B) を可能にすることで Z 上の Sidon 性へ持ち上げる。
f(0) = 0 を避けるため全要素に 1 を加算する (和・差の一意性は保存)。
"""
from __future__ import annotations


def _next_prime(m: int) -> int:
    """m 以上の最小の素数 (Bertrand のPOSTULATEより m < 2m に存在)。"""

    def is_prime(v: int) -> bool:
        if v < 2:
            return False
        if v % 2 == 0:
            return v == 2
        d = 3
        while d * d <= v:
            if v % d == 0:
                return False
            d += 2
        return True

    p = max(m, 2)
    while not is_prime(p):
        p += 1
    return p


def sidon_golomb(m: int) -> list[int]:
    """大きさ m > 1 の Sidon 集合 (Golomb ruler) を昇順リストで返す。

    すべての要素は正で、a_m <= 4*m^2 を満たす。
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    if m == 1:
        return [1]
    p = _next_prime(m)
    B = 2 * m - 1  # x 和の最大 2(m-1) < B なので復号で carry が起きない
    values = sorted(x + B * ((x * x) % p) for x in range(m))
    return [v + 1 for v in values]  # 正にシフト (Sidon/Golomb 性は保存)
