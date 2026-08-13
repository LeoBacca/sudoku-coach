# Sudo — Sudoku Coach

A small, client-side Sudoku coaching playground. The board stays visible beside a mascot-style coach chat so the player can reason through the puzzle without being handed the solution.

## Current MVP

- Sudoku board with given cells and editable cells
- Number entry and candidate-note mode
- Coach-mark mode for highlighting notes
- Related-cell and same-number highlighting
- Local browser persistence via `localStorage`
- Mascot-style coach chat with tips, technique prompts, explanations, and move-check prompts
- Responsive layout for desktop and mobile

## Run locally

No build step or dependencies are required:

```bash
python3 -m http.server 4173
```

Then open <http://localhost:4173>.

The test puzzle is seeded from the current coaching session. The coach logic is intentionally lightweight in this MVP; the next step is to connect the validated Sudoku coaching engine and a real chat model/API.
