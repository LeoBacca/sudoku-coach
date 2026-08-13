#!/usr/bin/env python3
"""AI-vs-AI Sudoku coaching test.

Runs novice-player simulations against the real Momo backend. The novice is
another LLM constrained to act like a beginner; every proposed move is checked
by the deterministic Sudoku engine before it is applied.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import backend

PUZZLE = "600000100000030000480609053000895000009040300000213000720301086000070000003000004"
RUNS = 5
MAX_ROUNDS = 20
MAX_EXCHANGES = 2


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key, value.strip().strip("'\""))


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    if fenced:
        text = fenced.group(1)
    else:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            text = match.group(0)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {"action": "stop", "reason": "non-object JSON"}
    except json.JSONDecodeError:
        return {"action": "stop", "reason": f"invalid JSON: {text[:180]}"}


def novice_call(prompt: str, style: str) -> dict[str, Any]:
    api_url = os.getenv("HERMES_API_URL", "http://127.0.0.1:8642/v1/chat/completions")
    api_key = os.environ["API_SERVER_KEY"]
    system = f"""Sei un giocatore principiante di Sudoku in un test controllato.
Stile di questo test: {style}
Non risolvere la griglia con un solver completo. Puoi usare solo candidati ricavati da riga, colonna e box 3x3 e ciò che Momo ha appena detto.
Non inventare una tecnica o una coordinata. Se non hai abbastanza informazioni, chiedi un altro aiuto.
Rispondi SOLO con JSON valido, senza markdown, usando esattamente uno di questi schemi:
{{"action":"ask","message":"..."}}
{{"action":"place","cell":"R1C1","digit":1,"reason":"..."}}
{{"action":"add_note","cell":"R1C1","digit":1,"reason":"..."}}
{{"action":"eliminate","cell":"R1C1","digit":1,"reason":"Momo ha verificato che il candidato va escluso."}}
{{"action":"stop","reason":"..."}}
Per place, scegli una sola cifra e una sola cella. Per add_note, scegli solo un candidato realmente possibile."""
    payload = {
        "model": "hermes-agent",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": 0.45,
        "max_tokens": 220,
    }
    req = urllib.request.Request(api_url, data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=90) as response:
        data = json.loads(response.read())
    return parse_json(data["choices"][0]["message"]["content"])


def solve_grid(grid: list[int]) -> list[int] | None:
    grid = list(grid)

    def search() -> bool:
        best = None
        best_candidates = None
        for i, value in enumerate(grid):
            if value:
                continue
            candidates = backend.ALL - {grid[p] for p in backend.PEERS[i] if grid[p]}
            if not candidates:
                return False
            if best is None or len(candidates) < len(best_candidates):
                best, best_candidates = i, candidates
        if best is None:
            return True
        for digit in sorted(best_candidates):
            grid[best] = digit
            if search():
                return True
        grid[best] = 0
        return False

    return grid if search() else None


def state_payload(grid: list[int], notes: dict[str, list[int]], eliminated: dict[str, list[int]], selected: str | None, recent: list[dict[str, str]]) -> dict[str, Any]:
    evidence = backend.build_evidence(grid, notes, eliminated)
    evidence["grid_rows"] = [grid[r * 9:(r + 1) * 9] for r in range(9)]
    evidence["givens_rows"] = [list(map(int, PUZZLE[r * 9:(r + 1) * 9])) for r in range(9)]
    givens = list(map(int, PUZZLE))
    evidence["user_entries"] = {backend.cell_label(i): grid[i] for i in range(81) if not givens[i] and grid[i]}
    evidence["selected_cell"] = selected
    evidence["recent_messages"] = recent[-12:]
    return evidence


def momo_call(message: str, grid: list[int], notes: dict[str, list[int]], eliminated: dict[str, list[int]], selected: str | None, recent: list[dict[str, str]]) -> tuple[str, list[dict[str, Any]]]:
    evidence = state_payload(grid, notes, eliminated, selected, recent)
    reply = backend.call_hermes(message, evidence, 2)
    facts = backend.select_facts(evidence["facts"], message)
    return reply, facts


def extract_mentioned_techniques(text: str) -> set[str]:
    lower = text.lower().replace("–", "-")
    names = {
        "naked single": "naked single", "hidden single": "hidden single",
        "locked candidates": "locked candidates", "pointing": "pointing", "claiming": "claiming",
        "naked pair": "naked pair", "hidden pair": "hidden pair",
        "naked triple": "naked triple", "hidden triple": "hidden triple",
        "x-wing": "x-wing", "swordfish": "swordfish", "xy-wing": "xy-wing", "xyz-wing": "xyz-wing",
    }
    return {value for term, value in names.items() if term in lower}


def cell_index(label: str) -> int | None:
    match = re.fullmatch(r"R([1-9])C([1-9])", str(label or ""))
    return (int(match.group(1)) - 1) * 9 + int(match.group(2)) - 1 if match else None


def apply_action(action: dict[str, Any], grid: list[int], notes: dict[str, list[int]], eliminated: dict[str, list[int]], givens: list[int]) -> tuple[bool, str]:
    cell = cell_index(action.get("cell"))
    digit = action.get("digit")
    if action.get("action") not in {"place", "add_note", "eliminate"}:
        return False, "not-a-move"
    if cell is None or not isinstance(digit, int) or not 1 <= digit <= 9:
        return False, "malformed-move"
    if givens[cell]:
        return False, "attempted-to-edit-given"
    if action["action"] == "eliminate":
        raw_candidates = backend.candidate_map(grid).get(cell, set())
        if digit not in raw_candidates:
            return False, f"cannot-eliminate-{digit}-valid-{sorted(raw_candidates)}"
        eliminated.setdefault(action["cell"], [])
        if digit not in eliminated[action["cell"]]:
            eliminated[action["cell"]].append(digit)
            eliminated[action["cell"]].sort()
        return True, "candidate-eliminated"
    if grid[cell] and action["action"] == "place":
        return False, "attempted-to-overwrite-cell"
    valid = backend.candidate_map(grid, eliminated).get(cell, set())
    if digit not in valid:
        return False, f"illegal-digit-{digit}-valid-{sorted(valid)}"
    if action["action"] == "place":
        grid[cell] = digit
        notes.pop(action["cell"], None)
    else:
        notes.setdefault(action["cell"], [])
        if digit not in notes[action["cell"]]:
            notes[action["cell"]].append(digit)
            notes[action["cell"]].sort()
    return True, "applied"


def run_game(run_id: int) -> dict[str, Any]:
    styles = [
        "faccio domande molto semplici e non riconosco subito le tecniche",
        "sono paziente ma tendo a confondere candidati e numeri già piazzati",
        "mi fisso sulle tecniche avanzate e chiedo spesso il nome della tecnica",
        "provo a verificare ogni mossa e posso fare domande ambigue come 'qui?'",
        "sono un principiante realistico: capisco solo una cosa per volta",
    ]
    grid = list(map(int, PUZZLE))
    givens = list(grid)
    notes: dict[str, list[int]] = {}
    eliminated: dict[str, list[int]] = {}
    recent: list[dict[str, str]] = []
    log: list[dict[str, Any]] = []
    errors: list[str] = []
    technique_counts: Counter[str] = Counter()
    question = "Dammi un piccolo tip, senza dirmi subito la soluzione completa."
    solved = solve_grid(grid)
    for round_no in range(1, MAX_ROUNDS + 1):
        if not any(v == 0 for v in grid):
            break
        for exchange in range(1, MAX_EXCHANGES + 1):
            try:
                reply, facts = momo_call(question, grid, notes, eliminated, None, recent)
            except Exception as exc:  # keep one failed run diagnosable
                errors.append(f"momo-error:{type(exc).__name__}:{exc}")
                log.append({"round": round_no, "exchange": exchange, "question": question, "momo_error": str(exc)})
                return {"run": run_id, "solved": False, "rounds": round_no, "placements": sum(1 for a in log if a.get("applied")), "errors": errors, "techniques": dict(technique_counts), "log": log}
            fact_techniques = [f["technique"] for f in facts]
            technique_counts.update(fact_techniques)
            mentioned = extract_mentioned_techniques(reply)
            if mentioned and not any(any(name in ft for name in mentioned) for ft in fact_techniques):
                errors.append(f"unsupported-technique-claim:{sorted(mentioned)} facts={fact_techniques}")
            context = {
                "grid": "".join(map(str, grid)),
                "candidates": {backend.cell_label(i): sorted(v) for i, v in backend.candidate_map(grid, eliminated).items()},
                "eliminated_candidates": eliminated,
                "notes": notes,
                "momo_reply": reply,
                "verified_facts": facts,
                "round": round_no,
                "exchange": exchange,
            }
            decision = novice_call(json.dumps(context, ensure_ascii=False), styles[run_id - 1])
            record = {"round": round_no, "exchange": exchange, "question": question, "momo": reply, "facts": fact_techniques, "decision": decision}
            action = decision.get("action")
            if action in {"place", "add_note", "eliminate"}:
                ok, status = apply_action(decision, grid, notes, eliminated, givens)
                record["applied"] = ok
                record["apply_status"] = status
                if not ok:
                    errors.append(status)
                else:
                    recent.extend([{"role": "user", "content": question}, {"role": "assistant", "content": reply}])
                    log.append(record)
                    break
            elif action == "stop":
                errors.append(f"novice-stopped:{decision.get('reason', '')}")
                log.append(record)
                return {"run": run_id, "solved": False, "rounds": round_no, "placements": sum(1 for a in log if a.get("applied") and a.get("decision", {}).get("action") == "place"), "notes": len(notes), "errors": errors, "techniques": dict(technique_counts), "log": log}
            else:
                question = str(decision.get("message") or "Quale singola casella devo guardare?")
                recent.extend([{"role": "user", "content": question}, {"role": "assistant", "content": reply}])
                log.append(record)
                continue
            question = "Ho fatto il passo. Qual è il prossimo piccolo indizio?"
        else:
            errors.append("exchange-limit-without-progress")
            break
    complete = not any(v == 0 for v in grid)
    solution_ok = solved is not None and complete and grid == solved
    if complete and not solution_ok:
        errors.append("completed-grid-does-not-match-solution")
    if not complete:
        errors.append(f"stalled-with-{grid.count(0)}-empty-cells")
    return {"run": run_id, "solved": solution_ok, "rounds": MAX_ROUNDS if not complete else round_no, "placements": sum(1 for a in log if a.get("applied") and a.get("decision", {}).get("action") == "place"), "notes": len(notes), "errors": errors, "techniques": dict(technique_counts), "remaining": grid.count(0), "log": log}


def main() -> int:
    load_env(Path.home() / ".hermes/.env")
    if not os.getenv("API_SERVER_KEY"):
        print("API_SERVER_KEY missing", file=sys.stderr)
        return 2
    started = time.time()
    results: list[dict[str, Any]] = []
    # Parallelize independent games, while each game's state remains isolated.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_game, i) for i in range(1, RUNS + 1)]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({k: result.get(k) for k in ("run", "solved", "rounds", "placements", "remaining", "errors", "techniques")}, ensure_ascii=False), flush=True)
    results.sort(key=lambda x: x["run"])
    summary = {
        "runs": RUNS,
        "solved_runs": sum(1 for r in results if r["solved"]),
        "avg_placements": round(sum(r.get("placements", 0) for r in results) / RUNS, 2),
        "total_errors": sum(len(r.get("errors", [])) for r in results),
        "elapsed_seconds": round(time.time() - started, 1),
        "results": results,
    }
    Path("ai_vs_ai_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ("runs", "solved_runs", "avg_placements", "total_errors", "elapsed_seconds")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
