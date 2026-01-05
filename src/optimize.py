"""
Hyperparameter optimization using Optuna for Pokemon predictor model.
This script searches for optimal alpha, gamma, learning_rate, dropout, and batch_size.
"""

import sys
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna

# Ensure src is in path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import existing classes and functions
from src.models.neural_network import (
    StreamingPokemonDataset,
    PokemonPredictor,
    FocalLoss,
    load_streaming_splits,
    calculate_batch_recall,
    CONFIG
)
from src.data.pokedex import NUM_POKEMON


def objective(trial):
    """
    Optuna will run this function multiple times with different hyperparameters.
    """
    
    # 1. Suggest Hyperparameters for this Trial
    # We search for the "sweet spot" to fix underconfidence
    params = {
        'gamma': trial.suggest_float('gamma', 0.5, 3.0),      # Try loosening the suppression (e.g., 1.0)
        'alpha': trial.suggest_float('alpha', 0.2, 0.8),      # Try increasing positive weight (e.g., 0.5)
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'dropout': trial.suggest_float('dropout', 0.2, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128])
    }
    
    # 2. Setup Data (Load only once ideally, but here for safety)
    # Using a smaller subset or just the validation set for speed is common,
    # but for full accuracy, use your standard train/val split.
    train_ids, val_ids = load_streaming_splits(CONFIG['data_path'])
    
    # Speed Tip: Use a smaller sample_rate for hyperparam search to run trials faster
    train_dataset = StreamingPokemonDataset(CONFIG['data_path'], train_ids, sample_rate=0.1) 
    val_dataset = StreamingPokemonDataset(CONFIG['data_path'], val_ids, sample_rate=0.2)
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], num_workers=0)

    # 3. Initialize Model & Loss
    input_dim = 2 + (NUM_POKEMON * 4)
    model = PokemonPredictor(input_dim, CONFIG['hidden_dim'], NUM_POKEMON, dropout=params['dropout']).to(CONFIG['device'])
    
    # HERE IS THE MAGIC: injecting the suggested alpha/gamma
    criterion = FocalLoss(alpha=params['alpha'], gamma=params['gamma'], reduction='sum').to(CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    # 4. Quick Training Loop (Fewer Epochs for Search)
    # You usually only need 3-5 epochs to tell if a config is good
    n_search_epochs = 5 
    
    for epoch in range(n_search_epochs):
        model.train()
        for feats, targets in train_loader:
            feats, targets = feats.to(CONFIG['device']), targets.to(CONFIG['device'])
            optimizer.zero_grad()
            output = model(feats)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
        # 5. Validation & Pruning
        model.eval()
        val_recall_sum = 0
        val_samples = 0
        with torch.no_grad():
            for feats, targets in val_loader:
                feats, targets = feats.to(CONFIG['device']), targets.to(CONFIG['device'])
                outputs = model(feats)
                # Reuse your existing metric function
                b_recall, b_n = calculate_batch_recall(outputs, targets)
                val_recall_sum += b_recall
                val_samples += b_n
        
        avg_recall = val_recall_sum / max(val_samples, 1)
        
        # Report intermediate result to Optuna
        trial.report(avg_recall, epoch)

        # Pruning: Stop bad trials early (if recall is 0.0 after epoch 1, kill it)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Return the metric we want to MAXIMIZE
    return avg_recall


if __name__ == "__main__":
    # Create the study
    study = optuna.create_study(direction="maximize")
    
    print("Starting Hyperparameter Search...")
    # Run 20 trials (should be enough given your fast training)
    study.optimize(objective, n_trials=20)

    print("\n--- Best Hyperparameters ---")
    print(study.best_params)
    print(f"\nBest Recall@6: {study.best_value:.4f}")
    
    # Visualization (Optional)
    # Uncomment these lines if you want to see plots:
    # optuna.visualization.plot_optimization_history(study).show()
    # optuna.visualization.plot_param_importances(study).show()
