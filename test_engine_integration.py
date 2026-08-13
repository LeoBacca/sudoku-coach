import unittest

import backend
from sudoku_engine import HumanSudokuEngine


PUZZLE = "600000100000030000480609053000895000009040300000213000720301086000070000003000004"


class EngineIntegrationTests(unittest.TestCase):
    def test_engine_exposes_one_verified_option_per_human_technique(self):
        options = HumanSudokuEngine().available_steps(PUZZLE)

        self.assertGreater(len(options), 1)
        self.assertEqual(options[0].technique, "Naked Single")
        self.assertEqual(options[0].placements, [{"cell": "R3C5", "digit": 2}])
        self.assertTrue(any(step.technique == "Hidden Pair" for step in options))

    def test_backend_evidence_comes_from_engine_steps(self):
        evidence = backend.build_evidence([int(char) for char in PUZZLE], {})

        self.assertTrue(evidence["valid"])
        self.assertTrue(evidence["facts"])
        self.assertEqual(evidence["facts"][0]["technique"], "naked single")
        self.assertIn("description", evidence["facts"][0])
        self.assertEqual(evidence["facts"][0]["cell"], "R3C5")
        self.assertEqual(evidence["facts"][0]["digit"], 2)

    def test_generic_tip_direction_hides_digit_and_points_to_cell(self):
        evidence = backend.build_evidence([int(char) for char in PUZZLE], {})
        fact = backend.select_facts(evidence["facts"], "Dammi un tip")[0]

        self.assertEqual(fact["technique"], "naked single")
        direction = backend.coaching_direction(fact, "Dammi un tip")
        self.assertIn("R3C5", direction)
        self.assertIn("Non nominare il digit", direction)

    def test_unit_phrase_names_box_position(self):
        self.assertEqual(backend.unit_phrase("box 8"), "box in basso al centro")


if __name__ == "__main__":
    unittest.main()
