"""汎用「論文 -> 実行可能構成 -> 検証」ハーネス (エンジン部分)。

構成的証明を伴う理論論文を numpy で検証するための最小フレームワーク。
- Check: 主張1件 = (id, claim, reference, 検証関数 fn -> (passed, detail))
- PaperVerifier: 逐次実行し、例外は FAIL 扱い、markdown レポートを生成する

他の論文へ適用する場合は one_neuron_cot/claims.py の Check 定義が
そのままテンプレートになる (fn の中で構成を組み立て、全入力で照合する)。
"""
from __future__ import annotations

import dataclasses
import time
import traceback
from typing import Callable


@dataclasses.dataclass
class Check:
    id: str
    claim: str
    reference: str
    fn: Callable[[], tuple[bool, str]]


@dataclasses.dataclass
class CheckResult:
    check: Check
    passed: bool
    detail: str
    seconds: float


class PaperVerifier:
    """Check 群を実行し PASS/FAIL レポートを作る。"""

    def __init__(self, title: str, citation: str):
        self.title = title
        self.citation = citation
        self.checks: list[Check] = []

    def add(self, check: Check) -> None:
        self.checks.append(check)

    def run(self) -> list[CheckResult]:
        results: list[CheckResult] = []
        for c in self.checks:
            t0 = time.perf_counter()
            try:
                passed, detail = c.fn()
                ok = bool(passed)
                detail = str(detail)
            except Exception:
                ok = False
                tail = traceback.format_exc(limit=3).strip().splitlines()
                detail = "exception: " + (tail[-1] if tail else "unknown")
            results.append(CheckResult(c, ok, detail, time.perf_counter() - t0))
        return results

    def report(self, results: list[CheckResult]) -> str:
        lines = [
            "# 検証レポート: " + self.title,
            "",
            "- citation: " + self.citation,
            "- date: " + time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            "- engine: one_neuron_cot (numpy)",
            "",
            "| check | claim | reference | result | detail |",
            "|---|---|---|---|---|",
        ]
        for r in results:
            mark = "PASS" if r.passed else "FAIL"
            det = r.detail.replace("|", "/").replace("\n", " ")
            lines.append("| " + r.check.id + " | " + r.check.claim + " | "
                         + r.check.reference + " | " + mark + " | " + det + " |")
        n_ok = sum(r.passed for r in results)
        verdict = "ALL PASS" if n_ok == len(results) else "FAILURES PRESENT"
        lines += ["", "**Summary: " + verdict + " (" + str(n_ok) + "/" + str(len(results)) + ")**", ""]
        return chr(10).join(lines)


def run_verify_paper(fast: bool = False) -> dict:
    """本論文 (arXiv:2610.17283) の主張群を検証する。fast=True は簡易サイズ。"""
    from .claims import build_verifier

    verifier = build_verifier(fast=fast)
    results = verifier.run()
    rep = verifier.report(results)
    return {
        "results": results,
        "report": rep,
        "all_passed": all(r.passed for r in results),
        "title": verifier.title,
        "citation": verifier.citation,
    }
