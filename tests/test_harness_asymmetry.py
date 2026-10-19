"""検証ハーネス (paper -> claims -> report) と CoT/E2E 学習非対称デモのテスト。"""
import itertools
import unittest

import numpy as np

from one_neuron_cot.asymmetry import all_binary_inputs, best_threshold_accuracy, run_demo
from one_neuron_cot.harness import Check, PaperVerifier, run_verify_paper


class TestHarnessEngine(unittest.TestCase):
    def test_report_contains_results_and_flags(self):
        v = PaperVerifier("test paper", "arXiv:0000.00000")
        v.add(Check(id="ok-check", claim="always true", reference="Th. 0",
                    fn=lambda: (True, "detail-A")))
        v.add(Check(id="bad-check", claim="always false", reference="Th. -1",
                    fn=lambda: (False, "detail-B")))
        results = v.run()
        rep = v.report(results)
        self.assertIn("ok-check", rep)
        self.assertIn("PASS", rep)
        self.assertIn("FAIL", rep)
        self.assertIn("detail-A", rep)
        self.assertEqual([r.passed for r in results], [True, False])

    def test_exception_becomes_failure(self):
        v = PaperVerifier("t", "x")
        v.add(Check(id="boom", claim="raises", reference="R",
                    fn=lambda: 1 / 0))
        results = v.run()
        self.assertFalse(results[0].passed)
        self.assertIn("exception", results[0].detail.lower())

    def test_run_verify_paper_all_pass(self):
        summary = run_verify_paper(fast=True)
        self.assertTrue(summary["all_passed"], summary["report"])
        ids = {r.check.id for r in summary["results"]}
        self.assertTrue({"thm2-exact", "prop1-cot", "bounds", "thm3-relu",
                         "no-cot-gap", "cor2-learning"} <= ids)


class TestAsymmetry(unittest.TestCase):
    def test_e2e_capacity_of_parity_is_bounded(self):
        # 決定論的な全列挙なので、理論上限の値も固定して検証する
        expected = {3: 0.75, 4: 0.6875}
        for n in (3, 4):
            X = all_binary_inputs(n)
            y = (X.sum(axis=1) % 2).astype(float)
            best_acc, _wt = best_threshold_accuracy(X, y, weight_cap=12)
            self.assertLess(best_acc, 1.0,
                            f"parity{n} が LTF で完全表現できてしまった")
            self.assertAlmostEqual(best_acc, expected[n],
                                   msg=f"parity{n} best LTF accuracy")
            self.assertGreaterEqual(best_acc, 0.5)

    def test_demo_matches_cot_vs_e2e(self):
        rows = run_demo(n_values=(3, 4), seed=0)
        for row in rows:
            self.assertLess(row["e2e_best"], 1.0)
            self.assertLessEqual(row["e2e_gd"], row["e2e_best"] + 1e-9)
            self.assertEqual(row["cot_acc"], 1.0,
                             f"CoT learner must reach 100% on {row['task']}")


if __name__ == "__main__":
    unittest.main()
