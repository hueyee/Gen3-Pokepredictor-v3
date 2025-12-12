"""Overall (multi-game) analytics entrypoint.

Generates and saves:
- Discovery Curve (Recall@10, Jaccard Top-6)
- Log Loss over turns (Next Reveal)
- Signal vs Noise (True Team avg vs Top-50 non-true avg)
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
    build_true_team_map,
    compute_discovery_metrics,
    compute_log_loss_over_turns,
    plot_discovery_curve,
    plot_log_loss,
    plot_signal_vs_noise,
)


def main(max_turn: int = 20, sample_games: Optional[int] = None) -> None:
    df = pd.read_csv(DATA_CSV)
    if sample_games is not None and sample_games > 0:
        games = df['game_id'].unique().tolist()[:sample_games]
        df = df[df['game_id'].isin(games)].copy()

    runner = ModelRunner(MODEL_PATH)
    team_map = build_true_team_map(df)

    discovery_df = compute_discovery_metrics(df, runner, team_map, max_turn=max_turn)
    loss_df = compute_log_loss_over_turns(df, runner, max_turn=max_turn)

    print('discovery_curve:', plot_discovery_curve(discovery_df))
    print('log_loss:', plot_log_loss(loss_df))
    print('signal_vs_noise:', plot_signal_vs_noise(df, runner, team_map, max_turn=max_turn))
    # Coverage report for turns
    try:
        turn_counts = df.groupby(df['turn_id'].apply(lambda x: int(float(x)) if pd.notna(x) else -1)).size()
        valid_turns = [t for t in turn_counts.index if isinstance(t, int) and 0 <= t <= max_turn]
        print('turn_coverage:', {int(t): int(turn_counts[t]) for t in sorted(valid_turns)})
    except Exception:
        pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Overall (multi-game) analytics')
    parser.add_argument('--max-turn', type=int, default=20)
    parser.add_argument('--sample-games', type=int, default=0, help='Limit to first N games (0 = all)')
    args = parser.parse_args()
    main(max_turn=args.max_turn, sample_games=(args.sample_games or None))
