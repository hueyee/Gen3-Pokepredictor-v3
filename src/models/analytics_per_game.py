"""Per-game analytics entrypoint.

Generates and saves:
- Suspects view (6 True vs Top-3 False positives) for a single game
- Rank Race (ranks over time for the 6 True) for a single game
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import sys
import pandas as pd

# Ensure project root imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.analytics import (
    DATA_CSV,
    MODEL_PATH,
    ModelRunner,
    pick_game_for_trajectories,
    plot_suspects,
    plot_rank_race,
)


def main(game_id: Optional[str] = None, sample_games: Optional[int] = None, top_false: int = 3) -> None:
    df = pd.read_csv(DATA_CSV)
    if sample_games is not None and sample_games > 0:
        games = df['game_id'].unique().tolist()[:sample_games]
        df = df[df['game_id'].isin(games)].copy()

    runner = ModelRunner(MODEL_PATH)
    gid = game_id or pick_game_for_trajectories(df)
    if not gid:
        print('No suitable game found for per-game analytics.')
        return

    print('suspects:', plot_suspects(df, runner, game_id=gid, top_false=top_false))
    print('rank_race:', plot_rank_race(df, runner, game_id=gid))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Per-game analytics')
    parser.add_argument('--game-id', type=str, default='', help='Specific game_id to visualize')
    parser.add_argument('--sample-games', type=int, default=0, help='Limit to first N games (0 = all)')
    parser.add_argument('--top-false', type=int, default=3, help='Number of false positives to plot in Suspects view')
    args = parser.parse_args()

    main(game_id=(args.game_id or None), sample_games=(args.sample_games or None), top_false=args.top_false)
