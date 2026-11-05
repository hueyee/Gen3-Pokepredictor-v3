import json
import gc
from pathlib import Path
from typing import List, Dict, Set

import pandas as pd
from tqdm import tqdm

from game_parser import (
    parse_replay_data,
    extract_features,
    refine_features,
)


def find_parquet_files() -> List[Path]:
    """Locate all Parquet replay metadata files.

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
    return files


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

    p1_rating = (rating_override or {}).get("p1", refined.get("player_ratings", {}).get("p1"))
    p2_rating = (rating_override or {}).get("p2", refined.get("player_ratings", {}).get("p2"))

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

    revealed_order = {"p1": [], "p2": []}
    revealed_set = {"p1": set(), "p2": set()}
    moves_seen: Dict[str, Dict[str, Set[str]]] = {"p1": {}, "p2": {}}

    prev_active = {"p1": None, "p2": None}
    
    for turn_index, turn in enumerate(refined.get("turns", []), start=1):
        
        # --- Update Moves Seen ---
        # Combine successful moves AND failed attempts (Focus Punch, Taunted moves)
        all_move_events = turn.get("moves_used", []) + turn.get("move_attempts", [])
        
        for move_event in all_move_events:
            species = move_event.get("pokemon")
            move = move_event.get("move")
            if not species or not move:
                continue
            
            owner = None
            if species == turn["active_pokemon"]["p1"]:
                owner = "p1"
            elif species == turn["active_pokemon"]["p2"]:
                owner = "p2"
            
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

        # --- Determine Actions ---
        actions = {"p1": None, "p2": None}
        
        for pid in ["p1", "p2"]:
            active = turn["active_pokemon"][pid]
            
            # 1. Check for Move (Successful)
            move_name = next((m.get("move") for m in turn.get("moves_used", []) if m.get("pokemon") == active), None)
            
            # 2. Check for Move (Failed/Blocked) - e.g. Focus Punch or Taunt
            if not move_name:
                move_name = next((m.get("move") for m in turn.get("move_attempts", []) if m.get("pokemon") == active), None)

            if move_name:
                actions[pid] = {"type": "move", "value": move_name}
            else:
                # 3. Check for Voluntary Switch
                # If they explicitly used |switch|, it's a switch action
                if turn.get("voluntary_switches", {}).get(pid):
                     actions[pid] = {"type": "switch", "value": active}
                
                # 4. Handle First Appearance (Lead)
                elif prev_active[pid] is None and active is not None:
                    actions[pid] = {"type": "switch", "value": active}
                
                # 5. Handle Drag/Forced Switch
                # If active changed, but it wasn't a voluntary switch (and no move was recorded),
                # it implies a drag/force-out.
                # In this case, we do NOT label it as a switch action, because the player didn't choose it.
                elif prev_active[pid] is not None and active != prev_active[pid]:
                    # Case 3: Dragged out while trying to attack (but attack didn't happen).
                    # We leave actions[pid] as None because we don't know what they selected.
                    pass

        # Build two perspective records for this turn
        for perspective in ["p1", "p2"]:
            opp = "p2" if perspective == "p1" else "p1"
            rating = p1_rating if perspective == "p1" else p2_rating

            observer_active = turn["active_pokemon"][perspective]
            opponent_active = turn["active_pokemon"][opp]

            observer_revealed = []
            for species in revealed_order[perspective]:
                species_moves = sorted(list(moves_seen[perspective].get(species, set())))
                observer_revealed.append({"species": species, "moves": species_moves})

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
                "target": opponent_full_team[perspective],
            }
            records.append(record)

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
    parquet_paths = find_parquet_files()
    output_dir = Path("data/parsed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "replays.jsonl"

    # Streaming mode: avoid storing all records in memory
    skipped_games: List[Dict] = []
    errored_games: List[Dict] = []
    written_records = 0
    processed_rows = 0

    if not parquet_paths:
        print("No Parquet files found for replay metadata (looked in data/raw/Replay Reference, data/raw/Replays, Parquets/).")
        return
    print(f"Processing {len(parquet_paths)} parquet file(s).")

    # Calculate total rows in Replay Reference for a global progress bar
    replay_ref_root = Path("data/raw/Replay Reference")
    replay_ref_files = [p for p in parquet_paths if p.parent == replay_ref_root]
    total_rows_replay_ref = 0
    for p in replay_ref_files:
        try:
            total_rows_replay_ref += len(pd.read_parquet(p))
        except Exception:
            # If a file can't be read, skip counting it for the global bar
            pass

    global_pbar = None
    if total_rows_replay_ref > 0:
        global_pbar = tqdm(total=total_rows_replay_ref, desc="Replay Reference rows", unit="row")

    # Open output file once for streaming writes
    out = open(output_file, "w", encoding="utf-8")

    # Iterate through all parquet files; update global row bar for Replay Reference
    for path in tqdm(parquet_paths, desc="Parquet files", unit="file"):
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue
        # Per-file row progress bar; iterate through items (tuples) for speed
        for row in tqdm(df.itertuples(index=False, name="Row"), total=len(df), desc=f"Rows in {path.name}", unit="row", leave=False):
            # Convert namedtuple to dict for processing
            records = process_log_row(row._asdict())
            processed_rows += 1
            if len(records) == 1 and ("skip_game_id" in records[0] or "error_game_id" in records[0]):
                sentinel = records[0]
                if "skip_game_id" in sentinel:
                    skipped_games.append(sentinel)
                else:
                    errored_games.append(sentinel)
            else:
                # Stream each record immediately to disk
                for rec in records:
                    out.write(json.dumps(rec) + "\n")
                    written_records += 1
            # Update global progress bar only for Replay Reference rows
            if global_pbar is not None and path.parent == replay_ref_root:
                # Keep a quick status of skipped/errors in the postfix
                global_pbar.update(1)
                global_pbar.set_postfix(skipped=len(skipped_games), errors=len(errored_games))

            # Periodic garbage collection to keep memory stable on huge datasets
            if processed_rows % 10000 == 0:
                gc.collect()

    if global_pbar is not None:
        global_pbar.close()

    out.flush()
    out.close()

    print(f"Processed {processed_rows} row(s) across parquet files.")
    print(f"Written {written_records} JSONL records to {output_file}")
    # Report skipped rows (each skipped row corresponds to a replay row that was not converted)
    if skipped_games:
        print(f"Skipped {len(skipped_games)} row(s) (e.g., missing ratings or empty logs).")
    if errored_games:
        print(f"Encountered errors in {len(errored_games)} games.")


if __name__ == "__main__":
    main()
