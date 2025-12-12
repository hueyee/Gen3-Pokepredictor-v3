"""Analytics for Pokemon team prediction model.

Generates:
- Discovery Curve: Recall@10 (orange) and Jaccard similarity (blue) over turns
- Log Loss over turns for the next revealed Pokemon
- Probability Trajectory Views for debugging:
    1) Signal vs Noise (True team avg vs Top-50 non-true avg)
    2) Suspects (6 True vs Top-3 False Positives within a game)
    3) Rank Race (ranks over time for the 6 True)

Data source: data/processed/Parquets/all_pokemon_moves.csv
Model: src/models/neural_network.PokemonPredictor (weights in models/pokemon_predictor.pth)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root imports
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.pokedex import ALL_POKEMON, NAME_TO_IDX, NUM_POKEMON
from src.models.neural_network import PokemonPredictor, CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = PROJECT_ROOT / 'data' / 'processed' / 'Parquets' / 'all_pokemon_moves.csv'
MODEL_PATH = PROJECT_ROOT / 'models' / 'pokemon_predictor.pth'
PLOTS_DIR = PROJECT_ROOT / 'data' / 'analytics'


def _encode_pokemon(name: Optional[str]) -> np.ndarray:
    one_hot = np.zeros(NUM_POKEMON, dtype=np.float32)
    if isinstance(name, str):
        idx = NAME_TO_IDX.get(name, -1)
        if idx != -1:
            one_hot[idx] = 1.0
    return one_hot


def _encode_team(names: List[str]) -> np.ndarray:
    multi = np.zeros(NUM_POKEMON, dtype=np.float32)
    for n in names:
        if not isinstance(n, str):
            continue
        idx = NAME_TO_IDX.get(n, -1)
        if idx != -1:
            multi[idx] = 1.0
    return multi


class ModelRunner:
    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None):
        self.device = device or CONFIG['device']
        input_dim = 2 + (NUM_POKEMON * 4)
        self.model = PokemonPredictor(input_dim, CONFIG['hidden_dim'], NUM_POKEMON, CONFIG['dropout']).to(self.device)
        weights = Path(model_path) if model_path else Path(MODEL_PATH)
        if not weights.exists():
            raise FileNotFoundError(f"Model weights not found at {weights}")
        self.model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model.eval()

    def features_from_row(self, row: pd.Series) -> np.ndarray:
        rating = float(row.get('p1_rating', 1500.0)) / 2000.0
        turn_id = float(row.get('turn_id', 0.0))
        turn_norm = min(turn_id, 50.0) / 50.0

        my_active = _encode_pokemon(row.get('p1_current_pokemon'))

        my_team_names: List[str] = []
        for i in range(1, 7):
            name = row.get(f'p1_pokemon{i}_name')
            if isinstance(name, str) and len(name) > 0:
                my_team_names.append(name)
        my_team = _encode_team(my_team_names)

        opp_active = _encode_pokemon(row.get('p2_current_pokemon'))
        opp_rev_names: List[str] = []
        revealed_ct = int(row.get('p2_number_of_pokemon_revealed', 0))
        for i in range(1, revealed_ct + 1):
            name = row.get(f'p2_pokemon{i}_name')
            if isinstance(name, str) and len(name) > 0:
                opp_rev_names.append(name)
        opp_revealed = _encode_team(opp_rev_names)

        feats = np.concatenate([[rating, turn_norm], my_active, my_team, opp_active, opp_revealed]).astype(np.float32)
        return feats

    def predict_probs(self, features: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(t)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        return probs


def build_true_team_map(df: pd.DataFrame) -> Dict[str, Set[str]]:
    team_map: Dict[str, Set[str]] = {}
    for gid, gdf in df.groupby('game_id'):
        names: Set[str] = set()
        for i in range(1, 7):
            col = f'p2_pokemon{i}_name'
            vals = gdf[col].dropna().astype(str).tolist()
            for v in vals:
                if v and v != 'nan':
                    names.add(v)
        team_map[str(gid)] = names
    return team_map


def topk_excluding_revealed(probs: np.ndarray, revealed: Set[str], k: int) -> List[str]:
    # Set revealed mons probs very negative so they don't rank
    pr = probs.copy()
    for name in revealed:
        idx = NAME_TO_IDX.get(name, -1)
        if idx != -1:
            pr[idx] = -1.0
    top_idx = np.argsort(-pr)[:k]
    return [ALL_POKEMON[i] for i in top_idx]


def compute_discovery_metrics(df: pd.DataFrame, runner: ModelRunner, team_map: Dict[str, Set[str]],
                              max_turn: int = 20) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        gid = str(row['game_id'])
        turn = int(row.get('turn_id', 0))
        if turn > max_turn:
            continue
        true_team = team_map.get(gid, set())
        # Revealed set this turn
        revealed: Set[str] = set()
        rct = int(row.get('p2_number_of_pokemon_revealed', 0))
        for i in range(1, rct + 1):
            nm = row.get(f'p2_pokemon{i}_name')
            if isinstance(nm, str) and nm:
                revealed.add(nm)

        feats = runner.features_from_row(row)
        probs = runner.predict_probs(feats)

        # Recall@10 on remaining unrevealed true mons
        top10 = set(topk_excluding_revealed(probs, revealed, k=10))
        unrevealed_truth = set([t for t in true_team if t not in revealed])
        denom = max(len(unrevealed_truth), 1)
        recall_at_10 = len(top10 & unrevealed_truth) / denom

        # Jaccard of 6-slot team: revealed + best remaining until 6
        predicted_team: List[str] = list(revealed)
        if len(predicted_team) < 6:
            k_needed = 6 - len(predicted_team)
            best_remaining = topk_excluding_revealed(probs, revealed, k=k_needed)
            predicted_team.extend(best_remaining)
        pred_set = set(predicted_team)
        union = len(pred_set | true_team) or 1
        jaccard = len(pred_set & true_team) / union

        records.append({'turn_id': turn, 'recall_at_10': recall_at_10, 'jaccard_top6': jaccard})

    met_df = pd.DataFrame(records)
    return met_df.groupby('turn_id', as_index=False).mean().sort_values('turn_id')


def compute_log_loss_over_turns(df: pd.DataFrame, runner: ModelRunner, max_turn: int = 20) -> pd.DataFrame:
    eps = 1e-9
    recs = []
    for _, row in df.iterrows():
        turn = int(row.get('turn_id', 0))
        if turn > max_turn:
            continue
        nxt = row.get('next_pokemon')
        if not isinstance(nxt, str) or len(nxt) == 0:
            continue
        feats = runner.features_from_row(row)
        probs = runner.predict_probs(feats)
        idx = NAME_TO_IDX.get(nxt, -1)
        if idx == -1:
            continue
        p = float(probs[idx])
        loss = -np.log(max(min(p, 1.0 - eps), eps))
        recs.append({'turn_id': turn, 'log_loss': loss})

    if not recs:
        return pd.DataFrame(columns=['turn_id', 'log_loss'])
    dfm = pd.DataFrame(recs)
    return dfm.groupby('turn_id', as_index=False).mean().sort_values('turn_id')


def pick_game_for_trajectories(df: pd.DataFrame) -> str:
    # Choose a game that has at least a few turns and next_pokemon labels
    cand = df.groupby('game_id')['turn_id'].count().sort_values(ascending=False).index.tolist()
    return str(cand[0]) if cand else ''


def _game_probs_matrix(df: pd.DataFrame, runner: ModelRunner, game_id: str) -> Tuple[List[int], np.ndarray]:
    gdf = df[df['game_id'] == game_id].sort_values('turn_id')
    turns: List[int] = []
    all_prob_rows: List[np.ndarray] = []
    for _, r in gdf.iterrows():
        feats = runner.features_from_row(r)
        probs = runner.predict_probs(feats)
        all_prob_rows.append(probs)
        turns.append(int(r['turn_id']))
    if not all_prob_rows:
        return turns, np.zeros((0, NUM_POKEMON))
    return turns, np.vstack(all_prob_rows)


def plot_signal_vs_noise(df: pd.DataFrame, runner: ModelRunner, team_map: Dict[str, Set[str]],
                         max_turn: int = 20, top_k_noise: int = 50) -> Path:
    rows = []
    for _, row in df.iterrows():
        raw_turn = row.get('turn_id', 0)
        # Robust turn parsing to handle mixed dtypes
        try:
            turn = int(float(raw_turn))
        except Exception:
            continue
        if turn < 0 or turn > max_turn:
            continue
        gid = str(row['game_id'])
        true_team = team_map.get(gid, set())
        feats = runner.features_from_row(row)
        probs = runner.predict_probs(feats)
        # Signal: avg prob of true team
        true_indices = [NAME_TO_IDX[n] for n in true_team if n in NAME_TO_IDX]
        signal = float(np.mean(probs[true_indices])) if true_indices else 0.0
        # Noise: top-K among non-true
        mask = np.ones(NUM_POKEMON, dtype=bool)
        mask[true_indices] = False
        non_true_probs = probs[mask]
        if non_true_probs.size == 0:
            noise = 0.0
        else:
            k = min(top_k_noise, non_true_probs.size)
            top_non_true = np.partition(non_true_probs, -k)[-k:]
            noise = float(np.mean(top_non_true))
        rows.append({'turn_id': turn, 'signal': signal, 'noise': noise})

    agg = pd.DataFrame(rows).groupby('turn_id', as_index=False).mean().sort_values('turn_id')
    # Ensure x-axis covers expected turn range even if some turns are missing
    full_index = pd.Index(range(0, max_turn + 1), name='turn_id')
    agg = agg.set_index('turn_id').reindex(full_index).reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(agg['turn_id'], agg['signal'], color='green', label='True Team (Signal)')
    plt.plot(agg['turn_id'], agg['noise'], color='grey', label=f'Top {top_k_noise} Non-True (Noise)')
    plt.xlabel('Turn')
    plt.ylabel('Average probability')
    plt.title('Signal vs Noise (Aggregation)')
    plt.legend()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / 'signal_vs_noise.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_suspects(df: pd.DataFrame, runner: ModelRunner, game_id: Optional[str] = None, top_false: int = 3) -> Optional[Path]:
    g = game_id or pick_game_for_trajectories(df)
    if not g:
        return None
    gdf = df[df['game_id'] == g].sort_values('turn_id')
    # True team
    true_team = set()
    for i in range(1, 7):
        vals = gdf[f'p2_pokemon{i}_name'].dropna().astype(str).tolist()
        for v in vals:
            if v and v != 'nan':
                true_team.add(v)

    turns, mat = _game_probs_matrix(df, runner, g)
    if mat.size == 0:
        return None

    avg_probs = mat.max(axis=0)  # use max over time to find strongest distractors
    false_scores = []
    for idx, name in enumerate(ALL_POKEMON):
        if name in true_team:
            continue
        false_scores.append((name, float(avg_probs[idx])))
    false_scores.sort(key=lambda x: x[1], reverse=True)
    suspects = [name for name, _ in false_scores[:top_false]]

    plt.figure(figsize=(12, 7))
    # Plot true team in green
    for name in sorted(true_team):
        if name not in NAME_TO_IDX:
            continue
        idx = NAME_TO_IDX[name]
        plt.plot(turns, mat[:, idx], color='green', alpha=0.8, linewidth=1.5, label=name)
    # Plot suspects in red
    for name in suspects:
        idx = NAME_TO_IDX[name]
        plt.plot(turns, mat[:, idx], color='red', alpha=0.9, linewidth=2.0, label=f"Suspect: {name}")
    plt.ylim(0.0, 1.0)
    plt.xlabel('Turn')
    plt.ylabel('Probability')
    plt.title(f'Suspects View (Game {g})')
    # Deduplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    plt.legend(*zip(*uniq), bbox_to_anchor=(1.02, 1), loc='upper left')
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f'suspects_{g}.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_rank_race(df: pd.DataFrame, runner: ModelRunner, game_id: Optional[str] = None, max_rank: int = 50) -> Optional[Path]:
    g = game_id or pick_game_for_trajectories(df)
    if not g:
        return None
    gdf = df[df['game_id'] == g].sort_values('turn_id')
    # True team
    true_team = set()
    for i in range(1, 7):
        vals = gdf[f'p2_pokemon{i}_name'].dropna().astype(str).tolist()
        for v in vals:
            if v and v != 'nan':
                true_team.add(v)

    turns, mat = _game_probs_matrix(df, runner, g)
    if mat.size == 0 or not true_team:
        return None

    # Compute ranks per turn (1 = highest probability), then cap to top-N for visualization
    ranks_over_time: Dict[str, List[int]] = {name: [] for name in true_team if name in NAME_TO_IDX}
    for t in range(mat.shape[0]):
        order = np.argsort(-mat[t])  # descending by prob
        rank_map = {int(idx): (i + 1) for i, idx in enumerate(order)}
        for name in list(ranks_over_time.keys()):
            idx = NAME_TO_IDX[name]
            r = rank_map.get(idx, NUM_POKEMON)
            # Cap to max_rank to keep the chart readable
            r = min(r, max_rank)
            ranks_over_time[name].append(r)

    plt.figure(figsize=(12, 7))
    for name, ranks in ranks_over_time.items():
        plt.plot(turns, ranks, linewidth=2.0, label=name)
    plt.gca().invert_yaxis()
    plt.ylim(max_rank, 1)
    yticks = [1, 5, 10, 20, 30, 40, max_rank]
    plt.yticks(yticks)
    plt.xlabel('Turn')
    plt.ylabel('Rank (1 = best)')
    plt.title(f'Rank Race (Game {g})')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f'rank_race_{g}.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_discovery_curve(df_discovery: pd.DataFrame) -> Path:
    plt.figure(figsize=(10, 6))
    turns = df_discovery['turn_id'].values
    plt.plot(turns, df_discovery['recall_at_10'].values, color='orange', label='Recall@10 (higher is better)')
    plt.plot(turns, df_discovery['jaccard_top6'].values, color='blue', label='Jaccard (Top-6)')
    plt.xlabel('Turn')
    plt.ylabel('Score')
    plt.ylim(0.0, 1.0)
    plt.title('Discovery Curve')
    plt.legend()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / 'discovery_curve.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_log_loss(df_loss: pd.DataFrame) -> Optional[Path]:
    if df_loss.empty:
        return None
    plt.figure(figsize=(10, 6))
    plt.plot(df_loss['turn_id'], df_loss['log_loss'], color='red', label='Log Loss (lower is better)')
    plt.xlabel('Turn')
    plt.ylabel('Negative log-likelihood')
    plt.title('Log Loss over Turns (Next Reveal)')
    plt.legend()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / 'log_loss_over_turns.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def run_analytics(max_turn: int = 20, sample_games: Optional[int] = None,
                  game_id: Optional[str] = None, top_false: int = 3) -> Dict[str, Optional[Path]]:
    df = pd.read_csv(DATA_CSV)
    if sample_games is not None and sample_games > 0:
        games = df['game_id'].unique().tolist()[:sample_games]
        df = df[df['game_id'].isin(games)].copy()

    runner = ModelRunner(MODEL_PATH)
    team_map = build_true_team_map(df)

    discovery_df = compute_discovery_metrics(df, runner, team_map, max_turn=max_turn)
    loss_df = compute_log_loss_over_turns(df, runner, max_turn=max_turn)

    paths: Dict[str, Optional[Path]] = {}
    paths['discovery_curve'] = plot_discovery_curve(discovery_df)
    paths['log_loss'] = plot_log_loss(loss_df)
    # New trajectory views
    paths['signal_vs_noise'] = plot_signal_vs_noise(df, runner, team_map, max_turn=max_turn)
    paths['suspects'] = plot_suspects(df, runner, game_id=game_id, top_false=top_false)
    paths['rank_race'] = plot_rank_race(df, runner, game_id=game_id)
    return paths


if __name__ == '__main__':
    print("This module provides analytics utilities. Use:\n"
          " - src/models/analytics_overall.py for multi-game stats\n"
          " - src/models/analytics_per_game.py for single-game views")
