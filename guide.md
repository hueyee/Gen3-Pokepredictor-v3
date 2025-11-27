# Phase 1: Data Acquisition
`dataScraper.py`
    - Downloads replays from showdown as raw HTML (just the text log).

`game_parser.py`
    - Turns the HTML game logs into a structured dataset.


# Phase 2: Preprocessing
`ParquetProcessing.py` / `ParquetSplit.py`
    - Helpers for creating the main parquet file

`CreateRevealedDatasetFromTurnDataset.py`
    - Splits games into snapshots

`SequenceCSV.py`
    - Flattens data into the final dataset

# Training
 `train_pokemon_models.py`
    - Trains the datasets

# Helpers
`one_hot_encoder.py`

