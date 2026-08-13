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

The test puzzle is seeded from the current coaching session.

## Human-style engine

`sudo-coach` uses the MIT-licensed [`dedoku`](https://github.com/n36l3c7/Dedoku) package through the local `sudoku_engine.py` adapter. Dedoku runs a logic-only pipeline by default and returns immutable steps containing:

- the technique used;
- a human-readable description;
- placements;
- candidate eliminations.

The adapter adds the live-state contract needed by the coach: persistent candidate eliminations, candidate snapshots, invalid-note checks, one next step, and a catalogue of one verified example per currently available technique. Backtracking is not enabled for coaching. Hermes receives only the selected deterministic fact and writes the explanation; it does not solve or validate the Sudoku.

The engine is covered by deterministic tests in `test_sudoku_engine.py` and `test_engine_integration.py`.


`backend.py` is a FastAPI service that validates the grid, calculates deterministic Sudoku evidence through the human-style `sudoku_engine.py` adapter (Dedoku), and calls the OpenAI-compatible Hermes API. The engine is the source of truth; Hermes only turns verified deductions into coaching language. It never puts the Hermes API key in the frontend.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
API_SERVER_KEY=... HERMES_API_URL=http://127.0.0.1:8642/v1/chat/completions uvicorn backend:app --host 127.0.0.1 --port 8787
```

The deployed frontend uses the HTTPS backend endpoint. For local development or to override it, set the backend URL in the browser console before loading the page:

```js
localStorage.setItem('sudo-api-url', 'http://127.0.0.1:8787')
```

