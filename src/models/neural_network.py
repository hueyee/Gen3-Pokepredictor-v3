import json
import logging
import sys
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, IterableDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src is in path to import pokedex
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.pokedex import ALL_POKEMON, NAME_TO_IDX, NUM_POKEMON

# Configuration
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = {
    'data_path': _PROJECT_ROOT / 'data' / 'parsed' / 'replays.jsonl',
    'model_save_path': _PROJECT_ROOT / 'models' / 'pokemon_predictor.pth',
    'batch_size': 128,
    'learning_rate': 0.0001,
    'epochs': 30,
    'hidden_dim': 512,
    'dropout': 0.24,
    'device': 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'),
    'seed': 42
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StreamingPokemonDataset(IterableDataset):
    def __init__(self, file_path, split_ids, training=True, sample_rate=1.0):
        super().__init__()
        self.file_path = file_path
        self.split_ids = split_ids
        self.training = training
        self.sample_rate = sample_rate
        self.all_pokemon = ALL_POKEMON
        self.name_to_idx = NAME_TO_IDX

    def _encode_pokemon(self, pokemon_name):
        idx = self.name_to_idx.get(pokemon_name, -1)
        one_hot = np.zeros(NUM_POKEMON, dtype=np.float32)
        if idx != -1:
            one_hot[idx] = 1.0
        return one_hot

    def _encode_team(self, team_list):
        multi_hot = np.zeros(NUM_POKEMON, dtype=np.float32)
        for mon in team_list:
            name = mon['species'] if isinstance(mon, dict) else mon
            idx = self.name_to_idx.get(name, -1)
            if idx != -1:
                multi_hot[idx] = 1.0
        return multi_hot

    def __iter__(self):
        with open(self.file_path, 'r') as f:
            for line in f:
                if self.sample_rate < 1.0 and np.random.random() > self.sample_rate:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                game_id = row.get('meta', {}).get('game_id')
                # Fast string check
                if game_id not in self.split_ids:
                    continue

                meta = row.get('meta', {})
                rating = float(meta.get('rating', 1500)) / 2000.0
                turn = float(meta.get('turn_number', 0)) / 50.0

                obs_state = row.get('observer_state', {})
                obs_active = self._encode_pokemon(obs_state.get('active_pokemon'))
                obs_team = self._encode_team(obs_state.get('revealed_team', []))

                opp_state = row.get('opponent_state', {})
                opp_active = self._encode_pokemon(opp_state.get('active_pokemon'))
                opp_revealed = self._encode_team(opp_state.get('revealed_team', []))

                features = np.concatenate([
                    [rating, turn],
                    obs_active,
                    obs_team,
                    opp_active,
                    opp_revealed
                ])

                target_list = row.get('target', [])
                target_vector = self._encode_team(target_list)

                yield torch.FloatTensor(features), torch.FloatTensor(target_vector)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.47, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): Weighting factor for the positive class (0 < alpha < 1).
                           Helps balance the large number of negative classes (unpicked mons).
            gamma (float): Focusing parameter (gamma >= 0). 
                           Higher gamma reduces loss for "easy" examples (confident predictions).
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. Calculate standard BCE (Binary Cross Entropy)
        # using functional API for numerical stability with logits
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # 2. Get the probabilities (pt) for the class that is true
        # p_t = p if y=1, else (1-p)
        # We can calculate pt using exp(-bce_loss) because BCE = -log(pt)
        pt = torch.exp(-bce_loss)
        
        # 3. Calculate the Focal Loss
        # Formula: -alpha * (1-pt)^gamma * log(pt)
        # Note: We apply alpha to positive examples and (1-alpha) to negatives
        
        # Create alpha factor tensor matching target shape
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_factor * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class PokemonPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(PokemonPredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

def load_streaming_splits(file_path, regen=False):
    """Create or load cached train/val game_id splits."""
    file_path = Path(file_path)
    split_dir = _PROJECT_ROOT / 'data' / 'splits'
    train_file = split_dir / 'train_game_ids.txt'
    val_file = split_dir / 'val_game_ids.txt'

    if not regen and train_file.exists() and val_file.exists():
        logger.info("Loading cached game_id splits...")
        train_ids = {line.strip() for line in train_file.read_text().splitlines() if line.strip()}
        val_ids = {line.strip() for line in val_file.read_text().splitlines() if line.strip()}
        if train_ids and val_ids:
            logger.info(f"Cached Train Games: {len(train_ids)}, Val Games: {len(val_ids)}")
            return train_ids, val_ids

    logger.info(f"Scanning game_ids from {file_path} for splits (regen={regen})...")
    if not file_path.exists():
        logger.error(f"Data file not found: {file_path}")
        sys.exit(1)

    total_bytes = file_path.stat().st_size
    game_ids = set()
    with open(file_path, 'r') as f:
        # Using a smaller unit scale for better visualization of huge files
        pbar = tqdm(total=total_bytes, unit='B', unit_scale=True, desc='Scanning game_ids')
        for line in f:
            pbar.update(len(line))
            # Optimization: simple string parsing instead of full json.loads if possible
            # But json.loads is safer. We stick to safety.
            try:
                # We only need the meta part, maybe optimize later
                row = json.loads(line)
                gid = row.get('meta', {}).get('game_id')
                if gid:
                    game_ids.add(gid)
            except:
                continue
        pbar.close()

    game_ids_list = list(game_ids)
    train_games, val_games = train_test_split(game_ids_list, test_size=0.2, random_state=CONFIG['seed'])
    logger.info(f"Train Games: {len(train_games)}, Val Games: {len(val_games)}")

    split_dir.mkdir(parents=True, exist_ok=True)
    train_file.write_text('\n'.join(train_games))
    val_file.write_text('\n'.join(val_games))
    return set(train_games), set(val_games)

def calculate_batch_recall(outputs, targets, k=6):
    """Calculates recall sum and count for a single batch."""
    # Move to CPU for metric calculation to save GPU memory
    probs = torch.sigmoid(outputs).cpu()
    targets = targets.cpu()
    
    # Get indices of top K probabilities
    _, top_k_indices = torch.topk(probs, k=k, dim=1)
    
    batch_recall_sum = 0
    valid_samples = 0
    
    targets_binary = (targets > 0.5)
    
    # Vectorized calculation where possible would be faster, but loop is safe
    for i in range(len(targets)):
        true_indices = torch.nonzero(targets_binary[i]).squeeze(dim=-1)
        # Handle scalar output for single items
        if true_indices.ndim == 0 and true_indices.numel() == 1:
             true_indices = true_indices.unsqueeze(0)
             
        if true_indices.numel() == 0:
            continue
            
        predicted = top_k_indices[i]
        
        # Intersection using boolean masks
        mask = torch.isin(predicted, true_indices)
        found = mask.sum().item()
        
        recall = found / true_indices.numel()
        batch_recall_sum += recall
        valid_samples += 1
        
    return batch_recall_sum, valid_samples

def save_checkpoint(epoch, model, optimizer, best_val_recall):
    ckpt_dir = _PROJECT_ROOT / 'models' / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f'epoch_{epoch}.pth'
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_val_recall': best_val_recall,
        'timestamp': time.time()
    }, ckpt_path)
    logger.info(f"Checkpoint saved: {ckpt_path}")

def load_latest_checkpoint(model, optimizer):
    ckpt_dir = _PROJECT_ROOT / 'models' / 'checkpoints'
    if not ckpt_dir.exists(): return 0, 0.0
    ckpts = sorted(ckpt_dir.glob('epoch_*.pth'), key=lambda p: p.stat().st_mtime)
    if not ckpts: return 0, 0.0
    
    latest = ckpts[-1]
    data = torch.load(latest, map_location=CONFIG['device'])
    model.load_state_dict(data['model_state'])
    optimizer.load_state_dict(data['optimizer_state'])
    logger.info(f"Resuming from {latest.name} (Epoch {data['epoch']})")
    return data['epoch'] + 1, data.get('best_val_recall', 0.0)

def train_model(regen_splits=False, checkpoint_every=0, resume=False):
    torch.manual_seed(CONFIG['seed'])

    # 1. Data Preparation
    train_ids, val_ids = load_streaming_splits(CONFIG['data_path'], regen=regen_splits)
    train_dataset = StreamingPokemonDataset(CONFIG['data_path'], train_ids, sample_rate=0.2)
    val_dataset = StreamingPokemonDataset(CONFIG['data_path'], val_ids, sample_rate=0.2)

    # Workers
    num_workers = 2 # Reduced for safety on Mac, can try 4 if memory holds
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        num_workers=num_workers,
        pin_memory=True, # MPS doesn't use it but it's harmless
        persistent_workers=True
    )
    # Validation loader doesn't need persistent workers as it runs once per epoch
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        num_workers=num_workers
    )

    # 2. Model Initialization
    input_dim = 2 + (NUM_POKEMON * 4)
    model = PokemonPredictor(input_dim, CONFIG['hidden_dim'], NUM_POKEMON, CONFIG['dropout']).to(CONFIG['device'])
    
    # Focal Loss with alpha=0.25 and gamma=2.0
    # alpha=0.25: Balances the large number of negative classes (~150:1 ratio)
    # gamma=2.0: Focuses on hard examples, down-weighting easy negatives
    criterion = FocalLoss(alpha=0.25, gamma=2.0).to(CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    start_epoch = 0
    best_val_recall = 0.0
    if resume:
        start_epoch, best_val_recall = load_latest_checkpoint(model, optimizer)

    logger.info(f"Starting training on {CONFIG['device']}...")
    
    for epoch in range(start_epoch, CONFIG['epochs']):
        # --- Training ---
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        
        # TQDM wrapper
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for features, targets in pbar:
            features, targets = features.to(CONFIG['device']), targets.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_batches += 1
            
            # Update progress bar occasionally
            if train_batches % 10 == 0:
                pbar.set_postfix(loss=f"{train_loss_sum/train_batches:.4f}")
                
        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # --- Validation (Memory Safe) ---
        model.eval()
        val_loss_sum = 0.0
        val_recall_sum = 0.0
        val_batches = 0
        val_samples = 0
        
        logger.info("Running validation...")
        with torch.no_grad():
            for features, targets in tqdm(val_loader, desc="Validating", leave=False):
                features, targets = features.to(CONFIG['device']), targets.to(CONFIG['device'])
                
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss_sum += loss.item()
                
                # Calculate metrics immediately, do not store outputs
                batch_recall, batch_n = calculate_batch_recall(outputs, targets)
                val_recall_sum += batch_recall
                val_samples += batch_n
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        avg_val_recall = val_recall_sum / max(val_samples, 1)

        logger.info(f"Epoch {epoch+1} Results: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Team Recall={avg_val_recall:.4f}")

        # Save Best
        if avg_val_recall > best_val_recall:
            best_val_recall = avg_val_recall
            Path(CONFIG['model_save_path']).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CONFIG['model_save_path'])
            logger.info(f"New best model saved! ({best_val_recall:.4f})")

        # Checkpoint
        if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(epoch, model, optimizer, best_val_recall)

def get_analytics(regen_splits=False):
    """Memory-safe analytics generation."""
    logger.info("Generating Analytics (Streaming Mode)...")

    _, val_ids = load_streaming_splits(CONFIG['data_path'], regen=regen_splits)
    val_dataset = StreamingPokemonDataset(CONFIG['data_path'], val_ids)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])

    input_dim = 2 + (NUM_POKEMON * 4)
    model = PokemonPredictor(input_dim, CONFIG['hidden_dim'], NUM_POKEMON).to(CONFIG['device'])
    try:
        model.load_state_dict(torch.load(CONFIG['model_save_path'], map_location=CONFIG['device']))
    except FileNotFoundError:
        logger.error("Model file not found. Train first.")
        return
        
    model.eval()

    # Accumulators for global stats
    total_samples = 0
    correct_bits = 0
    total_bits = 0
    
    # Per-Pokemon Accumulators (Counts, not raw data)
    pokemon_true_counts = np.zeros(NUM_POKEMON)
    pokemon_recall_hits = np.zeros(NUM_POKEMON)

    with torch.no_grad():
        for features, targets in tqdm(val_loader, desc="Analyzing"):
            features = features.to(CONFIG['device'])
            targets_np = targets.numpy() # Targets are binary
            
            # Predict
            logits = model(features)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds_binary = (probs > 0.5).astype(int)
            targets_binary = (targets_np > 0.5).astype(int)
            
            # 1. Global Hamming Accuracy (Bitwise accuracy)
            correct_bits += np.sum(preds_binary == targets_binary)
            total_bits += preds_binary.size
            total_samples += preds_binary.shape[0]
            
            # 2. Per Pokemon Stats
            # Add up true occurrences per pokemon
            batch_true_counts = np.sum(targets_binary, axis=0)
            pokemon_true_counts += batch_true_counts
            
            # Add up recall hits (Predicted=1 AND True=1)
            # We want Recall: TP / (TP + FN)
            # hits = (Pred==1) & (True==1)
            hits = (preds_binary == 1) & (targets_binary == 1)
            pokemon_recall_hits += np.sum(hits, axis=0)

    # --- Final Report ---
    logger.info("\n--- Analytics Report ---")
    
    global_acc = correct_bits / total_bits
    logger.info(f"Global Label Accuracy (Hamming): {global_acc:.4f}")
    
    # Calculate Per-Pokemon Recall
    # Avoid divide by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        pokemon_recalls = pokemon_recall_hits / pokemon_true_counts
        pokemon_recalls = np.nan_to_num(pokemon_recalls) # 0/0 -> 0

    pokemon_scores = {}
    for idx, name in enumerate(ALL_POKEMON):
        if idx >= len(pokemon_recalls): continue
        if pokemon_true_counts[idx] > 50: # Only count heavily represented pokemon
            pokemon_scores[name] = pokemon_recalls[idx]
            
    sorted_scores = sorted(pokemon_scores.items(), key=lambda x: x[1], reverse=True)

    logger.info("\nEasiest Pokemon to Predict (Top 10 Recall):")
    for name, score in sorted_scores[:10]:
        logger.info(f"  {name}: {score:.2f} ({int(pokemon_true_counts[NAME_TO_IDX[name]])} samples)")
        
    logger.info("\nHardest Pokemon to Predict (Bottom 10 Recall):")
    for name, score in sorted_scores[-10:]:
        logger.info(f"  {name}: {score:.2f} ({int(pokemon_true_counts[NAME_TO_IDX[name]])} samples)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--analyze', action='store_true', help='Run analytics')
    parser.add_argument('--regen-splits', action='store_true', help='Force regeneration of cached train/val game_id splits')
    parser.add_argument('--checkpoint-every', type=int, default=0, help='Save checkpoint every N epochs (0 disables)')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    args = parser.parse_args()
    
    if args.train:
        train_model(regen_splits=args.regen_splits, checkpoint_every=args.checkpoint_every, resume=args.resume)
    
    if args.analyze:
        get_analytics(regen_splits=args.regen_splits)
        
    if not args.train and not args.analyze:
        print("Please run with --train or --analyze")