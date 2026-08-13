"""Human-style Sudoku engine used as the source of truth for Sudo Coach.

Dedoku supplies the logical technique implementations. This module owns the
coach-facing contract: immutable snapshots, one next deduction, candidate
eliminations that persist in live state, and note validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dedoku import ContradictionError, Grid, SudokuSolver


@dataclass(frozen=True)
class Deduction:
    technique: str
    description: str
    placements: list[dict[str, int]] = field(default_factory=list)
    eliminations: list[dict[str, int]] = field(default_factory=list)


@dataclass(frozen=True)
class EngineState:
    grid: str
    candidates: dict[str, list[int]]
    eliminated_candidates: dict[str, list[int]]
    invalid_notes: list[dict[str, Any]]
    valid: bool = True


@dataclass(frozen=True)
class SolveResult:
    solved: bool
    final_grid: str
    steps: list[Deduction]
    used_backtracking: bool = False


class HumanSudokuEngine:
    """Adapter around Dedoku's logic-only, human-style solving pipeline."""

    def __init__(self, *, assume_unique: bool = False) -> None:
        self.assume_unique = assume_unique

    @staticmethod
    def _label(row: int, column: int) -> str:
        return f"R{row + 1}C{column + 1}"

    @staticmethod
    def _normalise_eliminations(raw: Any) -> dict[str, list[int]]:
        if not isinstance(raw, dict):
            return {}
        result: dict[str, list[int]] = {}
        for label, digits in raw.items():
            if not isinstance(label, str) or len(label) != 4 or label[0] != "R" or label[2] != "C":
                continue
            try:
                row, column = int(label[1]), int(label[3])
            except ValueError:
                continue
            if row not in range(1, 10) or column not in range(1, 10) or not isinstance(digits, (list, tuple, set)):
                continue
            clean = sorted({int(d) for d in digits if isinstance(d, int) and 1 <= d <= 9})
            if clean:
                result[label] = clean
        return result

    def _grid(self, puzzle: str, eliminated_candidates: Any = None) -> tuple[Grid, dict[str, list[int]]]:
        grid = Grid.from_string(puzzle)
        eliminated = self._normalise_eliminations(eliminated_candidates)
        for label, digits in eliminated.items():
            row, column = int(label[1]) - 1, int(label[3]) - 1
            cell = grid.cell(row, column)
            for digit in digits:
                if not cell.is_solved and digit in cell.candidates:
                    cell.remove_candidate(digit)
        return grid, eliminated

    @staticmethod
    def _candidate_map(grid: Grid) -> dict[str, list[int]]:
        return {
            cell.label: sorted(cell.candidates)
            for cell in grid.cells
            if not cell.is_solved
        }

    @staticmethod
    def _invalid_notes(grid: Grid, notes: Any) -> list[dict[str, Any]]:
        if not isinstance(notes, dict):
            return []
        candidates = HumanSudokuEngine._candidate_map(grid)
        invalid: list[dict[str, Any]] = []
        for label, raw_digits in notes.items():
            if label not in candidates or not isinstance(raw_digits, (list, tuple, set)):
                continue
            digits = sorted({int(d) for d in raw_digits if isinstance(d, int) and 1 <= d <= 9})
            wrong = sorted(set(digits) - set(candidates[label]))
            if wrong:
                invalid.append({"cell": label, "digits": wrong, "valid_candidates": candidates[label]})
        return invalid

    def inspect(
        self,
        puzzle: str,
        *,
        notes: Any = None,
        eliminated_candidates: Any = None,
    ) -> EngineState:
        grid, eliminated = self._grid(puzzle, eliminated_candidates)
        return EngineState(
            grid=grid.to_string(empty="0"),
            candidates=self._candidate_map(grid),
            eliminated_candidates=eliminated,
            invalid_notes=self._invalid_notes(grid, notes),
            valid=grid.is_valid(),
        )

    def _next_step_on_grid(self, grid: Grid) -> Deduction | None:
        solver = SudokuSolver(assume_unique=self.assume_unique)
        for technique in solver.techniques:
            step = technique.apply(grid)
            if step is not None:
                return self._deduction_from_step(step)
        return None

    @staticmethod
    def _deduction_from_step(step: Any) -> Deduction:
        return Deduction(
            technique=step.technique,
            description=step.description,
            placements=[
                {"cell": HumanSudokuEngine._label(p.row, p.column), "digit": p.digit}
                for p in step.placements
            ],
            eliminations=[
                {"cell": HumanSudokuEngine._label(e.row, e.column), "digit": e.digit}
                for e in step.eliminations
            ],
        )

    def next_step(
        self,
        puzzle: str,
        *,
        eliminated_candidates: Any = None,
    ) -> Deduction | None:
        grid, _ = self._grid(puzzle, eliminated_candidates)
        return self._next_step_on_grid(grid)

    def available_steps(
        self,
        puzzle: str,
        *,
        eliminated_candidates: Any = None,
    ) -> list[Deduction]:
        """Return one verified example for every technique available now.

        Each technique is evaluated against a fresh copy of the same live
        state. This exposes a catalogue for the coach without applying a
        whole solving chain or allowing one candidate technique to hide the
        others.
        """
        _, eliminated = self._grid(puzzle, eliminated_candidates)
        available: list[Deduction] = []
        for technique in SudokuSolver(assume_unique=self.assume_unique).techniques:
            grid, _ = self._grid(puzzle, eliminated)
            try:
                step = technique.apply(grid)
            except ContradictionError:
                continue
            if step is not None:
                available.append(self._deduction_from_step(step))
        return available

    def solve(self, puzzle: str) -> SolveResult:
        grid = Grid.from_string(puzzle)
        result = SudokuSolver(assume_unique=self.assume_unique).solve(grid)
        steps = [self._deduction_from_step(step) for step in result.steps]
        return SolveResult(
            solved=result.solved,
            final_grid=result.grid.to_string(empty="0") if result.grid else puzzle,
            steps=steps,
            used_backtracking=result.used_backtracking,
        )
