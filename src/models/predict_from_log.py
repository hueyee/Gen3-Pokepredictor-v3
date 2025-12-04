import sys
from pathlib import Path
import json

import torch

# Ensure project root on path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.predict import LivePredictor
from game_parser import parse_replay_data, extract_features, refine_features


def build_game_state_from_refined(refined: dict, perspective: str = "p1") -> dict:
    turns = refined.get("turns", [])
    if not turns:
        raise ValueError("No turns parsed from log.")

    last_turn = turns[-1]
    opp = "p2" if perspective == "p1" else "p1"

    # rating default: if refined has player_ratings use that else 1800
    rating = refined.get("player_ratings", {}).get(perspective, 1800)

    # Revealed teams up to last turn, as species only
    def revealed_species(pid: str):
        species = []
        seen = set()
        for t in turns:
            s = t["active_pokemon"].get(pid)
            if s and s not in seen:
                seen.add(s)
                species.append({"species": s, "moves": sorted(list(refined.get("moves_seen", {}).get(pid, {}).get(s, [])))})
        return species

    # Fallback if refined does not include moves_seen (older parser)
    if "moves_seen" not in refined:
        # Construct minimal revealed list without moves
        def revealed_min(pid: str):
            species = []
            seen = set()
            for t in turns:
                s = t["active_pokemon"].get(pid)
                if s and s not in seen:
                    seen.add(s)
                    species.append({"species": s, "moves": []})
            return species
        observer_revealed = revealed_min(perspective)
        opponent_revealed = revealed_min(opp)
    else:
        observer_revealed = revealed_species(perspective)
        opponent_revealed = revealed_species(opp)

    game_state = {
        "rating": rating,
        "turn": len(turns),
        "my_active": last_turn["active_pokemon"].get(perspective),
        "my_team": [r["species"] for r in observer_revealed],
        "opp_active": last_turn["active_pokemon"].get(opp),
        "opp_revealed": [r["species"] for r in opponent_revealed],
    }

    return game_state


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Predict opponent team from pasted replay log.")
    parser.add_argument("--model", type=str, default=None, help="Path to model .pth file.")
    parser.add_argument("--perspective", type=str, choices=["p1", "p2"], default="p1", help="Which player perspective to assume.")
    parser.add_argument("--top", type=int, default=10, help="Number of top predictions to show.")
    args = parser.parse_args()

    print("Paste the Showdown replay log text, then press Ctrl-D (EOF) to run prediction:\n")
    try:
        log_text = sys.stdin.read()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return

    if not log_text.strip():
        print("No log text provided.")
        return

    # Parse -> features -> refined
    try:
        turns = parse_replay_data(log_text)
        features = extract_features(turns)
        refined = refine_features(features)
    except Exception as e:
        print(f"Failed to parse log: {e}")
        return

    # Build game state from last turn
    try:
        game_state = build_game_state_from_refined(refined, perspective=args.perspective)
    except Exception as e:
        print(f"Failed to build game state: {e}")
        return

    # Run prediction
    predictor = LivePredictor(model_path=args.model)
    results = predictor.predict(game_state)

    print("\n--- Context ---")
    print(f"Perspective: {args.perspective}")
    print(f"Turn: {game_state['turn']}")
    print(f"My Active: {game_state['my_active']}")
    print(f"Opp Active: {game_state['opp_active']}")
    print(f"Opp Revealed: {', '.join(game_state['opp_revealed'])}")

    print("\n--- Prediction ---")
    print("Top likely unrevealed pokemon:")
    for name, prob in results[: args.top]:
        print(f"{name}: {prob*100:.1f}%")


if __name__ == "__main__":
    main()
