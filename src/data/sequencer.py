"""Sequence extraction and enrichment for Pokemon reveal data."""
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
import tqdm
import yaml


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Maximum number of Pokemon a player can have
MAX_REVEALS = 6


def extract_pokemon_reveal_sequences(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Extract the complete ordered sequence of P2 Pokemon reveals for each game.

    Args:
        df: DataFrame containing Pokemon reveal data with game_id column

    Returns:
        Dict mapping game_id to ordered list of Pokemon names as they were revealed
    """
    if 'game_id' not in df.columns:
        raise ValueError("DataFrame must contain a 'game_id' column")

    reveal_sequences = {}

    # Process each game separately
    for game_id, game_df in tqdm.tqdm(df.groupby('game_id'), desc="Extracting reveal sequences"):
        # Sort by turn_id to ensure chronological order
        game_df_sorted = game_df.sort_values('turn_id')

        # Initialize the Pokemon sequence with the first Pokemon
        first_turn = game_df_sorted.iloc[0]
        first_pokemon = first_turn['p2_pokemon1_name']

        # The sequence starts with the already-revealed first Pokemon
        sequence = [first_pokemon]

        # Track which Pokemon we've seen to avoid duplicates
        seen_pokemon = {first_pokemon}

        # Extract remaining Pokemon from next_pokemon column
        # We iterate through the sorted turns to get the exact sequence
        for _, row in game_df_sorted.iterrows():
            next_pokemon = row.get('next_pokemon')
            if pd.notna(next_pokemon) and next_pokemon not in seen_pokemon:
                sequence.append(next_pokemon)
                seen_pokemon.add(next_pokemon)

        # Store the complete sequence for this game
        reveal_sequences[game_id] = sequence

    return reveal_sequences


def add_future_reveal_columns(df: pd.DataFrame, reveal_sequences: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Add columns showing the sequence of future Pokemon reveals consistently for each game.

    Args:
        df: Original DataFrame
        reveal_sequences: Dict mapping game_id to ordered list of Pokemon names

    Returns:
        DataFrame with added future reveal columns
    """
    # Create a copy of the DataFrame to avoid modifying the original
    enriched_df = df.copy()

    # Initialize new columns for future reveals
    for i in range(1, MAX_REVEALS + 1):
        enriched_df[f'p2_next_reveal_{i}'] = np.nan

    # Process each game to add future reveal information
    print("Adding future reveal columns...")

    for game_id, sequence in tqdm.tqdm(reveal_sequences.items(), desc="Processing games"):
        game_mask = (enriched_df['game_id'] == game_id)
        game_df = enriched_df[game_mask]

        if game_df.empty:
            continue

        # For each row in this game, determine future reveals based on current state
        for idx, row in game_df.iterrows():
            # Number of already revealed Pokemon
            current_revealed_count = row['p2_number_of_pokemon_revealed']

            # Skip if all Pokemon already revealed
            if current_revealed_count >= MAX_REVEALS:
                continue

            # Find index of last revealed Pokemon in the sequence
            revealed_pokemon = []
            for i in range(1, current_revealed_count + 1):
                pokemon_name = row.get(f'p2_pokemon{i}_name')
                if pd.notna(pokemon_name):
                    revealed_pokemon.append(pokemon_name)

            # Add next Pokemon (future reveals) to this row
            for future_idx in range(1, MAX_REVEALS - current_revealed_count + 1):
                seq_idx = current_revealed_count - 1 + future_idx

                # Ensure the index is valid for the sequence
                if seq_idx < len(sequence):
                    enriched_df.at[idx, f'p2_next_reveal_{future_idx}'] = sequence[seq_idx]

    # Verify the new columns match next_pokemon where applicable
    check_mask = enriched_df['next_pokemon'].notna()
    if check_mask.any():
        match_mask = (enriched_df.loc[check_mask, 'p2_next_reveal_1'] == enriched_df.loc[check_mask, 'next_pokemon'])
        match_percentage = match_mask.mean() * 100
        print(f"Validation: p2_next_reveal_1 matches next_pokemon in {match_percentage:.2f}% of non-NA rows")

        if match_percentage < 99.9:
            mismatches = enriched_df[check_mask & (~match_mask)].head(5)
            print("\nSample mismatch examples:")
            for _, mismatch in mismatches.iterrows():
                print(f"Game {mismatch['game_id']}, Turn {mismatch['turn_id']}: "
                      f"next_pokemon={mismatch['next_pokemon']}, "
                      f"p2_next_reveal_1={mismatch['p2_next_reveal_1']}")

    return enriched_df


def analyze_sequence_coverage(df: pd.DataFrame) -> None:
    """Analyze the coverage of sequence columns to validate data quality."""
    reveal_cols = [f'p2_next_reveal_{i}' for i in range(1, MAX_REVEALS + 1)]

    # Count non-null values in each column
    non_null_counts = {col: df[col].notna().sum() for col in reveal_cols}
    total_rows = len(df)

    print("\nSequence column coverage:")
    for col, count in non_null_counts.items():
        print(f"  {col}: {count} non-null values ({count/total_rows*100:.2f}% of rows)")

    # Sample a few games to visually verify sequence consistency
    sample_games = df['game_id'].sample(min(5, df['game_id'].nunique())).unique()

    print("\nSample game sequences:")
    for game_id in sample_games:
        game_df = df[df['game_id'] == game_id].sort_values('turn_id')

        # Get the first and last row for this game
        first_row = game_df.iloc[0]
        last_row = game_df.iloc[-1]

        print(f"\nGame: {game_id}")
        print(f"  First turn ({first_row['turn_id']}): {first_row['p2_number_of_pokemon_revealed']} Pokemon revealed")

        # Show all Pokemon in the sequence
        sequence = []

        # First, get already revealed Pokemon
        for i in range(1, first_row['p2_number_of_pokemon_revealed'] + 1):
            pokemon = first_row.get(f'p2_pokemon{i}_name')
            if pd.notna(pokemon):
                sequence.append(f"{pokemon} (already revealed)")

        # Then, get future reveals
        for i in range(1, MAX_REVEALS + 1):
            col = f'p2_next_reveal_{i}'
            if col in first_row and pd.notna(first_row[col]):
                sequence.append(f"{first_row[col]} (next_reveal_{i})")

        print(f"  Sequence: {' -> '.join(sequence)}")

        print(f"  Last turn ({last_row['turn_id']}): {last_row['p2_number_of_pokemon_revealed']} Pokemon revealed")


def get_input_path(config):
    """Get the input CSV path from config."""
    if 'test_data_path' in config.get('data', {}):
        return Path(config['data']['test_data_path'])
    # Default to all_pokemon_moves.csv in same directory as processed_path
    processed_path = Path(config['data']['processed_path'])
    return processed_path.parent / 'all_pokemon_moves.csv'


def main():
    config = load_config()
    input_csv = get_input_path(config)
    output_csv = Path(config['data']['processed_path'])

    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    if 'game_id' not in df.columns:
        raise ValueError(f"Input CSV {input_csv} must contain a 'game_id' column")

    # Extract the complete ordered sequence of P2 Pokemon reveals for each game
    reveal_sequences = extract_pokemon_reveal_sequences(df)
    print(f"Extracted reveal sequences for {len(reveal_sequences)} games")

    # Print a few sample sequences to verify extraction
    sample_game_ids = list(reveal_sequences.keys())[:5]
    print("\nSample game sequences:")
    for game_id in sample_game_ids:
        sequence = reveal_sequences[game_id]
        print(f"Game {game_id}: {' -> '.join(sequence)}")

    # Add columns showing future Pokemon reveals
    enriched_df = add_future_reveal_columns(df, reveal_sequences)

    # Analyze the coverage and quality of sequence data
    analyze_sequence_coverage(enriched_df)

    # Save the enriched dataset
    print(f"\nSaving enriched dataset to {output_csv}...")
    enriched_df.to_csv(output_csv, index=False)
    print(f"Saved {len(enriched_df)} rows to {output_csv}")


if __name__ == "__main__":
    main()
