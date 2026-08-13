#!/usr/bin/env python3
"""Real Sudoku Coach backend.

The deterministic engine finds only validated Sudoku facts. Hermes turns those
facts into Momo's conversational reply; it is never allowed to invent a move.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from http import HTTPStatus
from itertools import combinations
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from sudoku_engine import HumanSudokuEngine

ALL = set(range(1, 10))
GRID_RE = re.compile(r"^[0-9.·_\-\s]{81}$")


def normalize_grid(value: Any) -> list[int]:
    if isinstance(value, list):
        if len(value) != 81:
            raise ValueError("grid list must contain exactly 81 cells")
        try:
            cells = [int(x) for x in value]
        except (TypeError, ValueError) as exc:
            raise ValueError("grid list contains a non-number") from exc
    elif isinstance(value, str):
        raw = value.replace(" ", "").replace("\n", "").replace("\r", "")
        if not GRID_RE.match(value) or len(raw) != 81:
            raise ValueError("grid must be an 81-character Sudoku string")
        cells = [0 if ch in ".·_-" else int(ch) for ch in raw]
    else:
        raise ValueError("grid must be a string or list")
    if any(n < 0 or n > 9 for n in cells):
        raise ValueError("grid values must be between 0 and 9")
    return cells


def cell_label(i: int) -> str:
    return f"R{i // 9 + 1}C{i % 9 + 1}"


def units() -> list[tuple[str, int, list[int]]]:
    result: list[tuple[str, int, list[int]]] = []
    for r in range(9):
        result.append(("row", r, [r * 9 + c for c in range(9)]))
    for c in range(9):
        result.append(("col", c, [r * 9 + c for r in range(9)]))
    for b in range(9):
        br, bc = (b // 3) * 3, (b % 3) * 3
        result.append(("box", b, [(br + dr) * 9 + bc + dc for dr in range(3) for dc in range(3)]))
    return result

UNITS = units()
HUMAN_ENGINE = HumanSudokuEngine(assume_unique=False)
PEERS: list[set[int]] = []
for i in range(81):
    r, c = divmod(i, 9)
    b = (r // 3) * 3 + c // 3
    peers = set(range(r * 9, r * 9 + 9)) | {rr * 9 + c for rr in range(9)} | set(UNITS[18 + b][2])
    peers.discard(i)
    PEERS.append(peers)


def validate_grid(grid: list[int]) -> list[str]:
    errors: list[str] = []
    for kind, number, indexes in UNITS:
        values = [grid[i] for i in indexes if grid[i]]
        duplicates = sorted({n for n in values if values.count(n) > 1})
        if duplicates:
            label = {"row": "riga", "col": "colonna", "box": "box"}[kind]
            errors.append(f"{label} {number + 1} contiene duplicati: {duplicates}")
    return errors


def candidate_map(grid: list[int], eliminated: Any = None) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    removed: dict[int, set[int]] = {}
    if isinstance(eliminated, dict):
        for label, digits in eliminated.items():
            match = re.fullmatch(r"R([1-9])C([1-9])", str(label))
            if not match or not isinstance(digits, (list, tuple, set)):
                continue
            i = (int(match.group(1)) - 1) * 9 + int(match.group(2)) - 1
            removed[i] = {int(d) for d in digits if isinstance(d, int) and 1 <= d <= 9}
    for i, value in enumerate(grid):
        if not value:
            used = {grid[p] for p in PEERS[i] if grid[p]}
            result[i] = (ALL - used) - removed.get(i, set())
    return result


def box_index(i: int) -> int:
    return (i // 9 // 3) * 3 + (i % 9 // 3)


def note_facts(grid: list[int], user_notes: Any, eliminated: Any = None) -> list[dict[str, Any]]:
    if not isinstance(user_notes, dict):
        return []
    candidates = candidate_map(grid, eliminated)
    result: list[dict[str, Any]] = []
    for label, raw in user_notes.items():
        match = re.fullmatch(r"R([1-9])C([1-9])", str(label))
        if not match or not isinstance(raw, (list, tuple, set)):
            continue
        i = (int(match.group(1)) - 1) * 9 + int(match.group(2)) - 1
        if grid[i]:
            result.append({"technique": "invalid annotation", "cell": label, "detail": "cell is already filled", "kind": "annotation_check"})
            continue
        try:
            notes = {int(n) for n in raw}
        except (TypeError, ValueError):
            continue
        invalid = sorted(notes - candidates.get(i, set()))
        if invalid:
            result.append({"technique": "invalid annotation", "cell": label, "digits": invalid, "valid_candidates": sorted(candidates.get(i, set())), "kind": "annotation_check"})
    return result


def fact_key(fact: dict[str, Any]) -> str:
    stable = {
        key: fact[key]
        for key in ("technique", "unit", "cell", "cells", "digit", "digits", "source", "target", "rows", "columns", "eliminations", "valid_candidates", "detail")
        if key in fact
    }
    return json.dumps(stable, sort_keys=True, ensure_ascii=False)


def requested_focus(message: str) -> str:
    text = message.lower()
    if any(word in text for word in ("x-wing", "x wing", "swordfish", "xy-wing", "xyz-wing", "catena", "colori")):
        return "advanced_exact"
    if any(word in text for word in ("pair", "coppia", "triple", "tripla", "subset", "hidden", "naked", "locked", "claiming", "pointing")):
        return "advanced_related"
    if "tecnic" in text or "strateg" in text:
        return "advanced_any"
    if any(word in text for word in ("annot", "cancell", "appunt")):
        return "annotation"
    if any(word in text for word in ("controll", "mossa", "inser")):
        return "check"
    return "general"


def unit_phrase(unit: str | None) -> str:
    """Turn an engine unit label into a player-facing location."""
    match = re.fullmatch(r"(row|col|box) (\d+)", str(unit or ""))
    if not match:
        return "questa unità"
    kind, raw_number = match.groups()
    number = int(raw_number)
    if kind == "row":
        return f"riga {number}"
    if kind == "col":
        return f"colonna {number}"
    positions = ["in alto a sinistra", "in alto al centro", "in alto a destra", "al centro a sinistra", "centrale", "al centro a destra", "in basso a sinistra", "in basso al centro", "in basso a destra"]
    return f"box {positions[number - 1]}" if 1 <= number <= 9 else f"box {number}"


def coaching_direction(fact: dict[str, Any] | None, message: str) -> str | None:
    """Give the LLM a strict direction for a low-help generic tip."""
    if not fact or requested_focus(message) != "general":
        return None
    technique = fact.get("technique")
    if technique == "hidden single":
        return f"TIP FORMATO OBBLIGATORIO: indica solo l'unità {unit_phrase(fact.get('unit'))} e dì che lì c'è un hidden single. Non nominare la cella e non nominare il digit verificato."
    if technique == "naked single":
        return f"TIP FORMATO OBBLIGATORIO: inizia esattamente con 'Guarda {fact.get('cell')}'. Dì che quella casella ha un solo candidato e chiedi di ricavare il numero dai vincoli. Non nominare il digit verificato e non sostituire la casella con un box o una cifra."
    return "Per questo tip indica prima l'area o la relazione da osservare. Non iniziare da una cifra isolata."


def select_facts(all_facts: list[dict[str, Any]], message: str) -> list[dict[str, Any]]:
    focus = requested_focus(message)
    advanced = {"locked candidates / pointing", "locked candidates / claiming", "naked pair", "hidden pair", "naked triple", "hidden triple", "X-Wing"}
    if focus == "advanced_exact":
        selected = [f for f in all_facts if f["technique"].lower() in {"x-wing", "xy-wing", "xyz-wing", "swordfish"}]
    elif focus in {"advanced_related", "advanced_any"}:
        text = message.lower()
        exact_terms = {
            "x-wing": "X-Wing", "x wing": "X-Wing", "swordfish": "Swordfish",
            "naked pair": "naked pair", "coppia nuda": "naked pair",
            "hidden pair": "hidden pair", "coppia nascosta": "hidden pair",
            "naked triple": "naked triple", "tripla nuda": "naked triple",
            "hidden triple": "hidden triple", "tripla nascosta": "hidden triple",
            "claiming": "locked candidates / claiming", "pointing": "locked candidates / pointing",
        }
        requested = next((technique for term, technique in exact_terms.items() if term in text), None)
        selected = [f for f in all_facts if f["technique"] == requested] if requested else [f for f in all_facts if f["technique"] in advanced]
    elif focus == "annotation":
        selected = [f for f in all_facts if f["technique"] == "invalid annotation"]
    elif focus == "check":
        selected = all_facts[:12]
    else:
        # A generic "tip" follows the teaching order. Do not jump to a pair
        # or an X-Wing while a verified single is available; advanced facts
        # are selected only when the player asks for an advanced technique.
        selected = all_facts[:4]
    advanced_order = {
        "X-Wing": 0,
        "hidden triple": 1,
        "naked triple": 2,
        "hidden pair": 3,
        "naked pair": 4,
        "locked candidates / claiming": 5,
        "locked candidates / pointing": 6,
    }
    if focus in {"advanced_related", "advanced_any"}:
        selected.sort(key=lambda fact: advanced_order.get(fact["technique"], 99))
    # One authoritative fact per turn. Passing a menu of unrelated facts to
    # the LLM is what made it blend a hidden pair with a locked candidate.
    return selected[:1]


def fact_cells(fact: dict[str, Any]) -> list[str]:
    cells: list[str] = []
    if fact.get("cell"):
        cells.append(fact["cell"])
    for item in fact.get("cells", []):
        if item not in cells:
            cells.append(item)
    for item in fact.get("eliminations", []):
        label = item.get("cell") if isinstance(item, dict) else item
        if label and label not in cells:
            cells.append(label)
    return cells


def fact_guidance(fact: dict[str, Any] | None) -> dict[str, Any] | None:
    if not fact:
        return None
    guidance = {"technique": fact.get("technique"), "kind": fact.get("kind")}
    for key in ("cell", "digit", "unit", "cells", "digits", "eliminations", "source", "target"):
        if fact.get(key):
            guidance[key] = fact[key]
    return guidance


def fact_elimination_map(fact: dict[str, Any]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    default_digit = fact.get("digit")
    for item in fact.get("eliminations", []):
        if isinstance(item, dict):
            label = item.get("cell")
            digits = item.get("digits", [])
        else:
            label = item
            digits = [default_digit] if default_digit else []
        if not re.fullmatch(r"R[1-9]C[1-9]", str(label)):
            continue
        valid_digits = sorted({int(d) for d in digits if isinstance(d, int) and 1 <= d <= 9})
        if valid_digits:
            result[str(label)] = valid_digits
    return result


def engine_facts(grid: list[int], eliminated_candidates: Any = None) -> list[dict[str, Any]]:
    """Convert verified Dedoku deductions to the backend fact contract."""
    puzzle = "".join(str(value) for value in grid)
    names = {
        "Pointing Candidates": "locked candidates / pointing",
        "Claiming Candidates": "locked candidates / claiming",
    }
    converted: list[dict[str, Any]] = []
    for step in HUMAN_ENGINE.available_steps(puzzle, eliminated_candidates=eliminated_candidates):
        fact: dict[str, Any] = {
            "technique": names.get(step.technique, step.technique.lower()),
            "description": step.description,
            "kind": "elimination" if step.eliminations else "placement",
        }
        if step.placements:
            fact["cell"] = step.placements[0]["cell"]
            fact["digit"] = step.placements[0]["digit"]
        if step.eliminations:
            fact["eliminations"] = [
                {"cell": item["cell"], "digits": [item["digit"]]}
                for item in step.eliminations
            ]
            fact["cells"] = [item["cell"] for item in step.eliminations]
            fact["digits"] = sorted({item["digit"] for item in step.eliminations})
        converted.append(fact)
    return converted


def facts(grid: list[int], user_notes: Any = None, eliminated_candidates: Any = None) -> list[dict[str, Any]]:
    candidates = candidate_map(grid, eliminated_candidates)
    found: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for i, nums in candidates.items():
        if len(nums) == 1:
            key = ("naked_single", i, tuple(sorted(nums)))
            if key not in seen:
                seen.add(key)
                found.append({"technique": "naked single", "cell": cell_label(i), "digit": next(iter(nums)), "kind": "placement"})

    for kind, number, indexes in UNITS:
        for digit in range(1, 10):
            hits = [i for i in indexes if i in candidates and digit in candidates[i]]
            if len(hits) == 1:
                key = ("hidden_single", kind, number, digit, hits[0])
                if key not in seen:
                    seen.add(key)
                    found.append({"technique": "hidden single", "unit": f"{kind} {number + 1}", "cell": cell_label(hits[0]), "digit": digit, "kind": "placement"})

    for kind, number, indexes in UNITS:
        if kind != "box":
            continue
        for digit in range(1, 10):
            hits = [i for i in indexes if i in candidates and digit in candidates[i]]
            if len(hits) < 2:
                continue
            rows = {i // 9 for i in hits}
            cols = {i % 9 for i in hits}
            if len(rows) == 1:
                row = next(iter(rows))
                eliminated = [i for i in range(row * 9, row * 9 + 9) if i not in indexes and i in candidates and digit in candidates[i]]
                if eliminated:
                    found.append({"technique": "locked candidates / pointing", "digit": digit, "source": f"box {number + 1}", "target": f"riga {row + 1}", "eliminations": [cell_label(i) for i in eliminated], "kind": "elimination"})
            if len(cols) == 1:
                col = next(iter(cols))
                eliminated = [i for i in range(col, 81, 9) if i not in indexes and i in candidates and digit in candidates[i]]
                if eliminated:
                    found.append({"technique": "locked candidates / pointing", "digit": digit, "source": f"box {number + 1}", "target": f"colonna {col + 1}", "eliminations": [cell_label(i) for i in eliminated], "kind": "elimination"})

    # Claiming: a line confines a digit to one box, so remove it from the
    # other cells of that box. This is the reverse direction of pointing.
    for kind, number, indexes in UNITS:
        if kind not in {"row", "col"}:
            continue
        for digit in range(1, 10):
            hits = [i for i in indexes if i in candidates and digit in candidates[i]]
            boxes = {box_index(i) for i in hits}
            if len(hits) < 2 or len(boxes) != 1:
                continue
            box = next(iter(boxes))
            box_indexes = UNITS[18 + box][2]
            eliminated = [i for i in box_indexes if i not in indexes and i in candidates and digit in candidates[i]]
            if eliminated:
                found.append({"technique": "locked candidates / claiming", "digit": digit, "source": f"{kind} {number + 1}", "target": f"box {box + 1}", "eliminations": [cell_label(i) for i in eliminated], "kind": "elimination"})

    # Naked pairs and triples. Only candidate sets with at least two values
    # are considered, avoiding a duplicate explanation of a single.
    for kind, number, indexes in UNITS:
        unit = f"{kind} {number + 1}"
        open_cells = [i for i in indexes if i in candidates and 2 <= len(candidates[i]) <= 3]
        for size, name in ((2, "pair"), (3, "triple")):
            for combo in combinations(open_cells, size):
                union = set().union(*(candidates[i] for i in combo))
                if len(union) != size or any(not candidates[i] <= union for i in combo):
                    continue
                eliminated = [i for i in indexes if i not in combo and i in candidates and candidates[i] & union]
                if eliminated:
                    eliminations = [{"cell": cell_label(i), "digits": sorted(candidates[i] & union)} for i in eliminated]
                    found.append({"technique": f"naked {name}", "unit": unit, "cells": [cell_label(i) for i in combo], "digits": sorted(union), "eliminations": eliminations, "kind": "elimination"})

        # Hidden pairs/triples: selected digits occur only in the selected
        # cells. Any other candidates in those cells can be removed.
        for size, name in ((2, "pair"), (3, "triple")):
            for digits in combinations(range(1, 10), size):
                positions_by_digit = {digit: {i for i in indexes if i in candidates and digit in candidates[i]} for digit in digits}
                positions = set().union(*positions_by_digit.values())
                # A hidden pair requires both digits to occupy exactly the
                # same two cells. Without this equality a hidden single was
                # being mislabeled as a hidden pair (e.g. one digit in only
                # one cell and the other in two).
                if len(positions) != size or not positions or any(positions_by_digit[digit] != positions for digit in digits):
                    continue
                eliminated = [i for i in positions if candidates[i] - set(digits)]
                if eliminated:
                    eliminations = [{"cell": cell_label(i), "digits": sorted(candidates[i] - set(digits))} for i in eliminated]
                    found.append({"technique": f"hidden {name}", "unit": unit, "cells": [cell_label(i) for i in sorted(positions)], "digits": list(digits), "eliminations": eliminations, "kind": "elimination"})

    # X-Wing for rows and columns. The two lines must have exactly the same
    # two positions for the same digit; only then are eliminations valid.
    for digit in range(1, 10):
        row_patterns = []
        for r in range(9):
            cols = tuple(c for c in range(9) if (r * 9 + c) in candidates and digit in candidates[r * 9 + c])
            if len(cols) == 2:
                row_patterns.append((r, cols))
        for (r1, cols), (r2, other_cols) in combinations(row_patterns, 2):
            if cols != other_cols:
                continue
            eliminated = [cell_label(r * 9 + c) for r in range(9) if r not in {r1, r2} for c in cols if r * 9 + c in candidates and digit in candidates[r * 9 + c]]
            if eliminated:
                found.append({"technique": "X-Wing", "digit": digit, "rows": [f"riga {r1 + 1}", f"riga {r2 + 1}"], "columns": [c + 1 for c in cols], "eliminations": eliminated, "kind": "elimination"})
        col_patterns = []
        for c in range(9):
            rows = tuple(r for r in range(9) if (r * 9 + c) in candidates and digit in candidates[r * 9 + c])
            if len(rows) == 2:
                col_patterns.append((c, rows))
        for (c1, rows), (c2, other_rows) in combinations(col_patterns, 2):
            if rows != other_rows:
                continue
            eliminated = [cell_label(r * 9 + c) for c in range(9) if c not in {c1, c2} for r in rows if r * 9 + c in candidates and digit in candidates[r * 9 + c]]
            if eliminated:
                found.append({"technique": "X-Wing", "digit": digit, "columns": [f"colonna {c1 + 1}", f"colonna {c2 + 1}"], "rows": [r + 1 for r in rows], "eliminations": eliminated, "kind": "elimination"})

    found.extend(note_facts(grid, user_notes, eliminated_candidates))

    deduped: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for fact in found:
        key = fact_key(fact)
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(fact)
    found = deduped

    # Keep ordering deterministic while exposing advanced facts to the LLM.
    priority = {"invalid annotation": 0, "naked single": 1, "hidden single": 2, "locked candidates / pointing": 3, "locked candidates / claiming": 4, "naked pair": 5, "hidden pair": 6, "naked triple": 7, "hidden triple": 8, "X-Wing": 9}
    found.sort(key=lambda f: (priority.get(f["technique"], 99), f.get("cell", ""), f.get("digit", 0)))
    return found[:80]


def build_evidence(grid: list[int], user_notes: Any, eliminated_candidates: Any = None) -> dict[str, Any]:
    errors = validate_grid(grid)
    if errors:
        return {"valid": False, "errors": errors, "facts": []}
    cmap = candidate_map(grid, eliminated_candidates)
    compact_candidates = {cell_label(i): "".join(map(str, sorted(nums))) for i, nums in cmap.items()}
    present = sorted(set(n for n in grid if n))
    unit_missing: dict[str, list[int]] = {}
    for kind, number, indexes in UNITS:
        unit_missing[f"{kind} {number + 1}"] = sorted(ALL - {grid[i] for i in indexes if grid[i]})
    return {
        "valid": True,
        "errors": [],
        "facts": engine_facts(grid, eliminated_candidates) + note_facts(grid, user_notes, eliminated_candidates),
        "candidates": compact_candidates,
        "eliminated_candidates": eliminated_candidates or {},
        "numbers_present": present,
        "numbers_missing_from_grid": sorted(ALL - set(present)),
        "numbers_missing_by_unit": unit_missing,
        "notes_received": bool(user_notes),
        "notes": user_notes or {},
    }


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    grid: Any
    notes: Any = None
    givens: Any = None
    selected_cell: str | None = Field(default=None, max_length=6)
    recent_messages: list[dict[str, str]] = Field(default_factory=list, max_length=20)
    eliminated_candidates: dict[str, list[int]] = Field(default_factory=dict)
    help_level: int = Field(default=2, ge=0, le=7)

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message cannot be empty")
        return value


app = FastAPI(title="Sudo Coach API", version="0.1.0")
allowed_origins = [x.strip() for x in os.getenv("SUDO_CORS_ORIGINS", "https://leobacca.github.io,http://127.0.0.1:4173,http://localhost:4173").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["POST", "GET", "OPTIONS"], allow_headers=["Content-Type"], max_age=3600)

_REQUESTS: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = int(os.getenv("SUDO_RATE_LIMIT", "30"))
RATE_WINDOW = 60


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path == "/chat":
        now = time.time()
        key = request.client.host if request.client else "unknown"
        recent = [t for t in _REQUESTS[key] if now - t < RATE_WINDOW]
        if len(recent) >= RATE_LIMIT:
            return JSONResponse({"detail": "Too many requests; try again shortly."}, status_code=429)
        recent.append(now)
        _REQUESTS[key] = recent
    return await call_next(request)


def call_hermes(message: str, evidence: dict[str, Any], help_level: int) -> str:
    api_url = os.getenv("HERMES_API_URL", "http://127.0.0.1:8642/v1/chat/completions")
    api_key = os.getenv("API_SERVER_KEY", "")
    if not api_key:
        raise RuntimeError("API_SERVER_KEY is not configured for the backend")
    selected_fact = select_facts(evidence.get("facts", []), message)[:1]
    facts_json = json.dumps(selected_fact, ensure_ascii=False)
    direction = coaching_direction(selected_fact[0] if selected_fact else None, message) or "Nessuna direzione speciale: rispondi usando il fatto verificato e il livello di aiuto richiesto."
    state_json = json.dumps({
        "grid_rows": evidence.get("grid_rows"),
        "givens_rows": evidence.get("givens_rows"),
        "user_entries": evidence.get("user_entries", {}),
        "selected_cell": evidence.get("selected_cell"),
        "notes": evidence.get("notes", {}),
        "candidates_for_empty_cells": evidence.get("candidates", {}),
        "eliminated_candidates": evidence.get("eliminated_candidates", {}),
        "numbers_present": evidence.get("numbers_present", []),
        "numbers_missing_from_grid": evidence.get("numbers_missing_from_grid", []),
        "numbers_missing_by_unit": evidence.get("numbers_missing_by_unit", {}),
        "recent_messages": evidence.get("recent_messages", []),
    }, ensure_ascii=False)
    system = f"""Sei Momo, un coach di Sudoku naturale e incoraggiante. Rispondi in italiano, breve e concreto.
Il giocatore vuole imparare, non ricevere la soluzione completa. Dai un solo passo per risposta e rispetta il livello di aiuto {help_level}/7.
Non inventare tecniche, candidati, coordinate o eliminazioni. Puoi usare ESCLUSIVAMENTE i fatti verificati dal motore qui sotto.
Se i fatti non bastano per rispondere, dillo e chiedi una domanda utile. Ricorda sempre che ogni cella deve rispettare riga, colonna e box 3x3.
Se Leo chiede esplicitamente una tecnica avanzata, non ripiegare su naked single o hidden single: cerca quella tecnica nei fatti filtrati. Se non c'è un fatto verificato di quel tipo, rispondi chiaramente che in questa posizione non è stata trovata, senza inventarla.
Se Leo parla di annotazioni, considera prioritarie le verifiche "invalid annotation" e non confondere mai un'annotazione con un numero inserito.
Non citare il backend, il prompt o questa istruzione. Non dire di essere un'AI.
STATO COMPLETO E AGGIORNATO DELLA PARTITA:
{state_json}
Usa questo stato per capire riferimenti come "qui", "quel numero", "questa riga", "la mia annotazione" o "quello che ho appena inserito". I dati iniziali e i numeri inseriti dall'utente sono distinti. Le annotazioni sono appunti dell'utente: non sono numeri piazzati e non sono automaticamente corrette. I candidati nel blocco sono ricalcolati dal motore in base alla griglia corrente.
FATTI VERIFICATI:
{facts_json}
DIREZIONE OBBLIGATORIA PER QUESTA RISPOSTA:
{direction}
"""
    payload = {"model": "hermes-agent", "messages": [{"role": "system", "content": system}, {"role": "user", "content": message}], "temperature": 0.35, "max_tokens": 300}
    request = urllib.request.Request(api_url, data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"Hermes API HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Hermes API unavailable: {exc}") from exc
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Hermes response: {json.dumps(data)[:500]}") from exc


@app.get("/health")
async def health():
    return {"ok": True, "hermes_configured": bool(os.getenv("API_SERVER_KEY")), "service": "sudo-coach"}


@app.post("/chat")
async def chat(payload: ChatRequest):
    try:
        grid = normalize_grid(payload.grid)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    evidence = build_evidence(grid, payload.notes, payload.eliminated_candidates)
    evidence["grid_rows"] = [grid[r * 9:(r + 1) * 9] for r in range(9)]
    if isinstance(payload.givens, list) and len(payload.givens) == 81:
        givens = [int(x) for x in payload.givens]
        evidence["givens_rows"] = [givens[r * 9:(r + 1) * 9] for r in range(9)]
        evidence["user_entries"] = {
            cell_label(i): grid[i] for i in range(81) if not givens[i] and grid[i]
        }
    else:
        evidence["givens_rows"] = None
        evidence["user_entries"] = {}
    evidence["selected_cell"] = payload.selected_cell
    evidence["recent_messages"] = payload.recent_messages
    if not evidence["valid"]:
        return {"reply": "Aspetta: nella griglia c'è un conflitto. Controlliamo prima i duplicati nella riga, colonna o box 3×3 indicato.", "technique": None, "highlight_cells": [], "evidence": evidence}
    try:
        reply = call_hermes(payload.message, evidence, payload.help_level)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    selected_facts = select_facts(evidence["facts"], payload.message)
    highlight: list[str] = []
    if selected_facts:
        first = selected_facts[0]
        highlight.extend(fact_cells(first))
    suggested_eliminations = fact_elimination_map(first) if selected_facts and first.get("kind") == "elimination" else {}
    return {"reply": reply, "technique": selected_facts[0]["technique"] if selected_facts else None, "highlight_cells": highlight, "suggested_eliminations": suggested_eliminations, "evidence": {"valid": True, "facts": selected_facts}}


@app.get("/")
async def root():
    return {"service": "Sudo Coach API", "docs": "/docs", "health": "/health"}
