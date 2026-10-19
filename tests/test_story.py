"""story デモ (ニューロン1個による文章生成) のテスト。"""
import itertools
import unittest

import numpy as np

from one_neuron_cot.story import build_story_circuit, generate


class TestStory(unittest.TestCase):
    def test_all_prompts_generate_text(self):
        for bits in itertools.product([0, 1], repeat=3):
            r = generate(list(bits))
            self.assertEqual(len(r["log"]), 10)
            self.assertTrue(r["text"].startswith("今日は"))
            self.assertTrue(r["text"].endswith("。"))
            for entry in r["log"]:
                self.assertIn(entry["val"], (0, 1))

    def test_generated_bits_equal_circuit_values(self):
        circuit = build_story_circuit()
        X = np.array([[1, 0, 1]], dtype=float)
        r = generate([1, 0, 1])
        self.assertEqual([e["val"] for e in r["log"]],
                         circuit.forward(X)[0].astype(int).tolist())

    def test_prompt_changes_story(self):
        texts = {tuple(p): generate(p)["text"] for p in ([0, 0, 0], [1, 1, 1])}
        self.assertNotEqual(texts[(0, 0, 0)], texts[(1, 1, 1)])


if __name__ == "__main__":
    unittest.main()
