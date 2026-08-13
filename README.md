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

## Real LLM backend

`backend.py` is a FastAPI service that validates the grid, calculates deterministic Sudoku evidence, and calls the OpenAI-compatible Hermes API. It never puts the Hermes API key in the frontend.

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

