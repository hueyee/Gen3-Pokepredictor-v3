# Scripts That Matter
## Getting Replays
Run `dataScraper`

## Parsing Replays
Run `DataParserForAllPokemon.py`
Then run `CreateRevealedDatasetForAllPokemon.py`
Then run `SequenceCSV.py`

## Training
Run `train_pokemon_models.py`

## To run predictions
Run `predict.py`

## To run server
Run `server.py`

## Analytics
Two entrypoints separate multi-game stats from per-game debugging views.

Overall (multi-game) analytics:

```
python src/models/analytics_overall.py --max-turn 20 --sample-games 10
```

Per-game analytics (pick a game or auto-select):

```
python src/models/analytics_per_game.py --game-id <GAME_ID> --top-false 3
```

Outputs are saved under `data/analytics/`:
- discovery_curve.png
- log_loss_over_turns.png
- signal_vs_noise.png
- suspects_<game_id>.png
- rank_race_<game_id>.png

Options:

```
# All games, higher turn cap
python src/models/analytics_overall.py --max-turn 50

# Limit to first N games for speed
python src/models/analytics_overall.py --sample-games 50 --max-turn 30

# Focus suspects/rank views on a specific game
python src/models/analytics_per_game.py --game-id <GAME_ID> --top-false 3
```