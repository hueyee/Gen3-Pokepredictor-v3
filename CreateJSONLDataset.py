import json
from pathlib import Path
from typing import List, Dict, Set

import pandas as pd

from game_parser import (
    parse_replay_data,
    extract_features,
    refine_features,
)


def find_parquet_files(limit: int = 5) -> List[Path]:
    """Locate Parquet replay metadata files, return up to limit paths.

    Searches in priority order: data/raw/Replay Reference, data/raw/Replays, Parquets/.
    """
    search_roots = [
        Path("data/raw/Replay Reference"),
        Path("data/raw/Replays"),
        Path("Parquets"),
    ]
    files: List[Path] = []
    for root in search_roots:
        if root.exists():
            files.extend(sorted(root.glob("*.parquet")))
        if len(files) >= limit:
            break
    return files[:limit]


def collect_full_team(refined: Dict) -> Dict[str, List[str]]:
    """Collect the complete set of Pokemon that appeared for each player over the battle."""
    teams = {"p1": [], "p2": []}
    seen = {"p1": set(), "p2": set()}
    for turn in refined.get("turns", []):
        for pid in ["p1", "p2"]:
            active = turn["active_pokemon"][pid]
            if active and active not in seen[pid]:
                seen[pid].add(active)
                teams[pid].append(active)
    return teams


def generate_jsonl_records(
    game_id: str,
    refined: Dict,
    rating_override: Dict[str, int] | None = None,
    upload_time: int | None = None,
) -> List[Dict]:
    """Generate per-turn JSONL records for both perspectives."""
    records: List[Dict] = []

    # Ratings: prefer override provided (from Parquet row rating), otherwise use refined
    p1_rating = (rating_override or {}).get("p1", refined.get("player_ratings", {}).get("p1"))
    p2_rating = (rating_override or {}).get("p2", refined.get("player_ratings", {}).get("p2"))

    # Skip logic per instructions
    lower_id = game_id.lower()
    default_sites = ("smogtours", "azure", "gold", "bigbang")
    if p1_rating is None:
        if any(site in lower_id for site in default_sites):
            p1_rating = 1800
        else:
            raise ValueError(f"Skip game {game_id}: missing p1 rating")
    if p2_rating is None:
        if any(site in lower_id for site in default_sites):
            p2_rating = 1800
        else:
            raise ValueError(f"Skip game {game_id}: missing p2 rating")

    full_teams = collect_full_team(refined)
    opponent_full_team = {"p1": full_teams["p2"], "p2": full_teams["p1"]}

    # Track revealed order and moves seen so far
    revealed_order = {"p1": [], "p2": []}
    revealed_set = {"p1": set(), "p2": set()}
    moves_seen: Dict[str, Dict[str, Set[str]]] = {"p1": {}, "p2": {}}

    prev_active = {"p1": None, "p2": None}
    for turn_index, turn in enumerate(refined.get("turns", []), start=1):
        # Update moves from this turn
        for move_event in turn.get("moves_used", []):
            species = move_event.get("pokemon")
            move = move_event.get("move")
            if not species or not move:
                continue
            # Determine which player the species belongs to based on active_pokemon
            # Fallback: if species matches current active of p1 use p1 else if p2 then p2.
            owner = None
            if species == turn["active_pokemon"]["p1"]:
                owner = "p1"
            elif species == turn["active_pokemon"]["p2"]:
                owner = "p2"
            # If ambiguous, add to both if already revealed there; else skip.
            if owner is None:
                for pid in ["p1", "p2"]:
                    if species in revealed_set[pid]:
                        owner = pid
                        break
            if owner is None:
                continue
            moves_seen[owner].setdefault(species, set()).add(move)

        # Update revealed order based on active pokemon
        for pid in ["p1", "p2"]:
            active = turn["active_pokemon"][pid]
            if active and active not in revealed_set[pid]:
                revealed_set[pid].add(active)
                revealed_order[pid].append(active)

        # Determine actions for both sides on this turn
        actions = {"p1": None, "p2": None}
        # If a move by current active exists, record as move; otherwise if active changed from prev, record switch
        for pid in ["p1", "p2"]:
            active = turn["active_pokemon"][pid]
            move_name = next((m.get("move") for m in turn.get("moves_used", []) if m.get("pokemon") == active), None)
            if move_name:
                actions[pid] = {"type": "move", "value": move_name}
            else:
                if prev_active[pid] is not None and active != prev_active[pid]:
                    actions[pid] = {"type": "switch", "value": active}
                elif prev_active[pid] is None and active is not None:
                    # First appearance counts as a switch-into-field
                    actions[pid] = {"type": "switch", "value": active}

        # Build two perspective records for this turn
        for perspective in ["p1", "p2"]:
            opp = "p2" if perspective == "p1" else "p1"
            rating = p1_rating if perspective == "p1" else p2_rating

            observer_active = turn["active_pokemon"][perspective]
            opponent_active = turn["active_pokemon"][opp]

            # Observer revealed team with moves so far
            observer_revealed = []
            for species in revealed_order[perspective]:
                species_moves = sorted(list(moves_seen[perspective].get(species, set())))
                observer_revealed.append({"species": species, "moves": species_moves})

            # Opponent revealed team with any moves seen so far
            opponent_revealed = []
            for s in revealed_order[opp]:
                s_moves = sorted(list(moves_seen[opp].get(s, set())))
                opponent_revealed.append({"species": s, "moves": s_moves})

            record = {
                "meta": {
                    "game_id": game_id,
                    "turn_number": turn_index,
                    "perspective": perspective,
                    "rating": rating,
                    "upload_time": upload_time,
                },
                "observer_state": {
                    "active_pokemon": observer_active,
                    "revealed_team": observer_revealed,
                },
                "opponent_state": {
                    "active_pokemon": opponent_active,
                    "revealed_team": opponent_revealed,
                },
                "actions": {
                    "observer_action": actions[perspective],
                    "opponent_action": actions[opp],
                },
                # Target: full opponent team species (ground truth) without moves
                "target": opponent_full_team[perspective],
            }
            records.append(record)
        # Update previous active tracking
        prev_active = {
            "p1": turn["active_pokemon"]["p1"],
            "p2": turn["active_pokemon"]["p2"],
        }

    return records


def process_log_row(row: Dict) -> List[Dict]:
    """Process a single Parquet row containing a replay log and metadata.

    Expected keys: id, log, players (list), rating (may be None).
    We parse player ratings from the log lines with |player| entries.
    """
    game_id = row.get("id") or "unknown-id"
    log_text = row.get("log") or ""
    players = row.get("players")
    # Ensure players is a list for downstream logic (may be numpy array or pandas object)
    if isinstance(players, (tuple, list)):
        pass
    elif players is None:
        players = []
    else:
        try:
            # Convert array-like to list safely
            players = list(players)
        except Exception:
            players = []

    if not log_text.strip():
        return [{"skip_game_id": game_id, "reason": "empty log"}]

    try:
        turns = parse_replay_data(log_text)
        features = extract_features(turns)
        refined = refine_features(features)
        # Rating override must come from Parquet row 'rating'. If missing/NaN -> None
        rating_override = None
        row_rating = row.get("rating")
        try:
            if row_rating is not None and str(row_rating).strip() != "" and str(row_rating).lower() != "nan":
                rating_val = int(float(row_rating))
                rating_override = {"p1": rating_val, "p2": rating_val}
        except Exception:
            rating_override = None

        # Upload time from Parquet row if present
        upload_time = None
        ut = row.get("uploadtime")
        try:
            if ut is not None and str(ut).strip() != "" and str(ut).lower() != "nan":
                upload_time = int(float(ut))
        except Exception:
            upload_time = None

        return generate_jsonl_records(
            game_id,
            refined,
            rating_override=rating_override,
            upload_time=upload_time,
        )
    except ValueError as e:
        return [{"skip_game_id": game_id, "reason": str(e)}]
    except Exception as e:
        return [{"error_game_id": game_id, "error": str(e)}]


def main():
    parquet_paths = find_parquet_files(limit=5)
    output_dir = Path("data/parsed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "replays.jsonl"

    all_records: List[Dict] = []
    skipped_games = []
    errored_games = []

    if not parquet_paths:
        print("No Parquet files found for replay metadata (looked in data/raw/Replay Reference, data/raw/Replays, Parquets/).")
        return
    print(f"Processing {len(parquet_paths)} parquet file(s) (limited to first 5).")

    processed_rows = 0
    for path in parquet_paths:
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue
        # Iterate rows until we have 5 games total across all files
        for _, row in df.iterrows():
            if processed_rows >= 5:
                break
            records = process_log_row(row.to_dict())
            if len(records) == 1 and ("skip_game_id" in records[0] or "error_game_id" in records[0]):
                sentinel = records[0]
                if "skip_game_id" in sentinel:
                    skipped_games.append(sentinel)
                    print(f"Skipping {sentinel['skip_game_id']}: {sentinel['reason']}")
                else:
                    errored_games.append(sentinel)
                    print(f"Error {sentinel['error_game_id']}: {sentinel['error']}")
            else:
                all_records.extend(records)
            processed_rows += 1
        if processed_rows >= 5:
            break

    with open(output_file, "w", encoding="utf-8") as out:
        for rec in all_records:
            out.write(json.dumps(rec) + "\n")

    print(f"Written {len(all_records)} JSONL records to {output_file}")
    if skipped_games:
        print(f"Skipped {len(skipped_games)} games (missing ratings).")
    if errored_games:
        print(f"Encountered errors in {len(errored_games)} games.")


if __name__ == "__main__":
    main()
