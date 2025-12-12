import sys
from pathlib import Path
import torch
import onnx

# Ensure project root is on sys.path for src imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.data.pokedex import NUM_POKEMON
from src.models.neural_network import PokemonPredictor

CONFIG = {
    'hidden_dim': 512,
    'dropout': 0.3,
    'model_save_path': PROJECT_ROOT / 'models' / 'pokemon_predictor.pth',
    'onnx_save_path': PROJECT_ROOT / 'models' / 'pokemon_predictor.onnx',
}

device = 'cpu'

def main():
    input_dim = 2 + (NUM_POKEMON * 4)

    model = PokemonPredictor(input_dim, CONFIG['hidden_dim'], NUM_POKEMON, CONFIG['dropout'])
    model.load_state_dict(torch.load(CONFIG['model_save_path'], map_location=device))
    model.to(device)
    model.eval()

    # Create a dummy input matching the training feature vector shape
    dummy_input = torch.randn(1, input_dim, device=device)

    # Ensure output directory exists
    CONFIG['onnx_save_path'].parent.mkdir(parents=True, exist_ok=True)

    # Export base ONNX file
    torch.onnx.export(
        model,
        dummy_input,
        str(CONFIG['onnx_save_path']),
        input_names=["features"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={
            "features": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    # Re-save with external data format to produce .onnx.data
    onnx_model = onnx.load(str(CONFIG['onnx_save_path']))
    external_data_path = str(CONFIG['onnx_save_path']) + ".data"
    onnx.save_model(
        onnx_model,
        str(CONFIG['onnx_save_path']),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=Path(external_data_path).name,
        size_threshold=1024,
        convert_attribute=False,
    )

    print(f"ONNX model exported to {CONFIG['onnx_save_path']}")
    print(f"External data saved to {external_data_path}")


if __name__ == "__main__":
    main()
