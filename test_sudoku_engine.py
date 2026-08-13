import unittest

from sudoku_engine import HumanSudokuEngine


PUZZLE = "600000100000030000480609053000895000009040300000213000720301086000070000003000004"
SOLUTION = "637584192295137468481629753342895617819746325576213849724351986968472531153968274"


class HumanSudokuEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = HumanSudokuEngine()

    def test_first_step_is_a_verified_human_deduction(self):
        step = self.engine.next_step(PUZZLE)

        self.assertIsNotNone(step)
        self.assertEqual(step.technique, "Naked Single")
        self.assertEqual(step.placements, [{"cell": "R3C5", "digit": 2}])
        self.assertEqual(step.eliminations, [])
        self.assertTrue(step.description)

    def test_logic_path_solves_seed_without_backtracking(self):
        result = self.engine.solve(PUZZLE)

        self.assertTrue(result.solved)
        self.assertEqual(result.final_grid, SOLUTION)
        self.assertFalse(result.used_backtracking)
        self.assertGreater(len(result.steps), 1)
        self.assertNotIn("Backtracking", [step.technique for step in result.steps])

    def test_candidate_elimination_is_part_of_the_live_state(self):
        eliminated = {"R2C7": [2], "R4C7": [2]}

        state = self.engine.inspect(PUZZLE, eliminated_candidates=eliminated)

        self.assertNotIn("2", state.candidates["R2C7"])
        self.assertNotIn("2", state.candidates["R4C7"])
        self.assertEqual(state.eliminated_candidates, eliminated)

    def test_invalid_user_note_is_reported_but_does_not_change_truth(self):
        state = self.engine.inspect(PUZZLE, notes={"R3C5": [1, 2, 7]})

        self.assertEqual(state.candidates["R3C5"], [2])
        self.assertEqual(state.invalid_notes[0]["cell"], "R3C5")
        self.assertEqual(state.invalid_notes[0]["digits"], [1, 7])


if __name__ == "__main__":
    unittest.main()
