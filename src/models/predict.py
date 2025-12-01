import torch
import numpy as np
import sys
import json
from pathlib import Path

# Add project root to path so we can import src
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.pokedex import ALL_POKEMON, NAME_TO_IDX, NUM_POKEMON
from src.models.neural_network import PokemonPredictor, CONFIG

class LivePredictor:
    def __init__(self, model_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.all_pokemon = ALL_POKEMON
        self.name_to_idx = NAME_TO_IDX
        
        # Initialize Model Structure
        # Input dim must match training: 2 (meta) + 4 * NUM_POKEMON (one-hot vectors)
        input_dim = 2 + (NUM_POKEMON * 4)
        self.model = PokemonPredictor(input_dim, CONFIG['hidden_dim'], NUM_POKEMON)
        
        # Load Weights
        path = model_path if model_path else CONFIG['model_save_path']
        print(f"Loading model from {path}...")
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        except FileNotFoundError:
            print("Error: Model file not found. Please train the model first.")
            sys.exit(1)
            
        self.model.to(self.device)
        self.model.eval()

    def _encode_pokemon(self, pokemon_name):
        """Converts a single pokemon name to a one-hot vector."""
        one_hot = np.zeros(NUM_POKEMON, dtype=np.float32)
        idx = self.name_to_idx.get(pokemon_name, -1)
        if idx != -1:
            one_hot[idx] = 1.0
        return one_hot

    def _encode_team(self, team_list):
        """Converts a list of pokemon (strings or dicts) to a multi-hot vector."""
        multi_hot = np.zeros(NUM_POKEMON, dtype=np.float32)
        for item in team_list:
            # Handle standard string list or the dict structure from logs
            name = item['species'] if isinstance(item, dict) else item
            idx = self.name_to_idx.get(name, -1)
            if idx != -1:
                multi_hot[idx] = 1.0
        return multi_hot

    def predict(self, game_state):
        """
        Runs prediction on a dictionary representing the game state.
        
        Expected format of game_state:
        {
            'rating': 1500,
            'turn': 5,
            'my_active': 'Swampert',
            'my_team': ['Swampert', 'Skarmory', 'Tyranitar', ...],
            'opp_active': 'Zapdos',
            'opp_revealed': ['Zapdos', 'Celebi']
        }
        """
        # 1. Feature Engineering
        rating = float(game_state.get('rating', 1500)) / 2000.0
        turn = float(game_state.get('turn', 0)) / 50.0
        
        obs_active = self._encode_pokemon(game_state.get('my_active'))
        obs_team = self._encode_team(game_state.get('my_team', []))
        opp_active = self._encode_pokemon(game_state.get('opp_active'))
        opp_revealed = self._encode_team(game_state.get('opp_revealed', []))
        
        # 2. Build Tensor
        features = np.concatenate([
            [rating, turn],
            obs_active,
            obs_team,
            opp_active,
            opp_revealed
        ])
        
        tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device) # Add batch dim
        
        # 3. Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            
        # 4. Decode Results
        results = []
        already_revealed = set(game_state.get('opp_revealed', []))
        
        # Create list of (Name, Probability)
        for idx, prob in enumerate(probs):
            name = self.all_pokemon[idx]
            # Optional: Filter out pokemon we already know exist
            if name not in already_revealed:
                results.append((name, prob))
        
        # Sort by high probability
        results.sort(key=lambda x: x[1], reverse=True)
        return results

# --- Example Usage ---
if __name__ == "__main__":
    predictor = LivePredictor()

    # Mock Game State: Late game Gen 3 scenario
    sample_game = {
        'rating': 1650,
        'turn': 15,
        'my_active': 'Metagross',
        'my_team': ['Metagross', 'Swampert', 'Snorlax', 'Gengar', 'Zapdos', 'Dugtrio'],
        
        # We see they have a Skarmory out, and we've seen Blissey earlier
        'opp_active': 'Skarmory',
        'opp_revealed': ['Skarmory', 'Blissey', 'Tyranitar'] 
    }

    print("\n--- Running Prediction ---")
    print(f"Opponent Visible: {sample_game['opp_revealed']}")
    
    predictions = predictor.predict(sample_game)
    
    print("\nTop 5 Most Likely Unrevealed Pokemon:")
    for name, prob in predictions[:5]:
        print(f"{name}: {prob*100:.1f}%")