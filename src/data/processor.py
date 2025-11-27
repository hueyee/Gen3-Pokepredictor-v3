"""Parquet data processing utilities."""
from pathlib import Path
from typing import List, TypedDict

import pandas as pd
import numpy as np
import yaml


PokemonDict = TypedDict('PokemonDict', {'name': str, 'moves': List[str]})


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_parquet(file_path: Path) -> pd.DataFrame:
    """Load a parquet file into a DataFrame."""
    return pd.read_parquet(file_path)


def main():
    config = load_config()
    parquet_folder = Path("Parquets/")
    file_path = parquet_folder / "all_pokemon_showdown_replays.parquet"
    
    df = load_parquet(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
