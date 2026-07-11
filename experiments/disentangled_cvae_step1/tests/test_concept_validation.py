from __future__ import annotations

import unittest

from experiments.disentangled_cvae_step1.concept_validation import run_concept_validation


class ConceptValidationTests(unittest.TestCase):
    def test_recovers_known_condition_mixtures(self) -> None:
        result = run_concept_validation(seed=42, epochs=80)
        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.macro_f1, 0.80)
        self.assertGreaterEqual(result.gate_target_correlation, 0.75)
        self.assertGreaterEqual(result.condition_reconstruction_gain, 0.05)
        self.assertGreater(result.macro_f1, result.shuffled_target_macro_f1 + 0.20)


if __name__ == "__main__":
    unittest.main()
