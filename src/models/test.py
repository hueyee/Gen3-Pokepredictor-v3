"""Test and evaluate Pokemon prediction models."""
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import tqdm
import logging
import warnings
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import yaml

from src.features.encoder import CustomOneHotEncoder

logging.getLogger('joblib').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return {
        'file_path': Path(cfg['data']['test_data_path']),
        'models_dir': Path(cfg['models']['output_dir']),
        'confidence_threshold': cfg['testing']['confidence_threshold'],
        'max_propagations': cfg['testing']['max_propagations'],
        'sample_size': cfg['testing']['sample_size'],
        'debug': False,
    }

def load_latest_models(config):
    """Load the latest trained models from the models directory."""
    models_dir = Path(config['models_dir'])
    model_dirs = sorted(glob.glob(str(models_dir / "*")))
    if not model_dirs:
        raise ValueError(f"No model directories found in {models_dir}")

    latest_dir = Path(model_dirs[-1])
    print(f"Loading models from directory: {latest_dir}")

    models = {}

    for pokemon_idx in tqdm.tqdm(range(2, 7), desc="Loading models"):
        model_path = latest_dir / f"pokemon_prediction_model_{pokemon_idx}.joblib"
        if model_path.exists():
            models[pokemon_idx] = joblib.load(model_path)
        else:
            raise ValueError(f"Model for Pokemon {pokemon_idx} not found in {latest_dir}")

    return models

def prepare_features(row, pokemon_idx, model_info):
    features = {}

    categorical_features = model_info['categorical_features']
    numerical_features = model_info['numerical_features']

    for feature in categorical_features + numerical_features:
        if feature in row:
            features[feature] = row[feature]

    return features

def encode_features(features, model_info):
    categorical_features = model_info['categorical_features']
    numerical_features = model_info['numerical_features']
    encoder = model_info['encoder']

    X_cat = np.zeros((1, encoder.n_features))

    current_idx = 0
    for feature in categorical_features:
        if feature in features:
            value = features[feature]
            if feature in encoder.encoders and value in encoder.encoders[feature]:
                X_cat[0, current_idx + encoder.encoders[feature][value]] = 1
        current_idx += len(encoder.encoders[feature])

    X_num = np.zeros((1, len(numerical_features)))
    for i, feature in enumerate(numerical_features):
        if feature in features:
            X_num[0, i] = features[feature]

    X_combined = np.hstack([X_cat, X_num])
    return X_combined

def predict_with_model(features, model_info):
    X_combined = encode_features(features, model_info)
    model = model_info['model']
    proba = model.predict_proba(X_combined)[0]
    classes = model.classes_

    result = {cls: prob for cls, prob in zip(classes, proba)}
    return result

def predict_next_pokemon(row, pokemon_idx, model_info):
    """Predict the next Pokemon based on features and model."""
    features = prepare_features(row, pokemon_idx, model_info)
    predictions = predict_with_model(features, model_info)
    return predictions


def propagate_predictions(row, models, start_idx=2, confidence_threshold=0.1):
    """Propagate predictions through the Pokemon positions."""
    original_row = row.copy()
    results = []

    all_state_history = {}
    visualization_data = {
        'nodes': [],
        'edges': [],
        'game_states': []
    }

    root_id = 0
    starting_pokemon = row.get('p2_pokemon1_name', 'Unknown')
    visualization_data['nodes'].append({
        'id': root_id,
        'label': f"Start\n{starting_pokemon}",
        'type': 'start',
        'pokemon': starting_pokemon,
        'level': 0,
        'order': 0
    })

    visualization_data['game_states'].append({
        'node_id': root_id,
        'state': original_row.copy(),
        'level': 0
    })

    current_node_id = 1
    initial_turn_id = original_row.get('turn_id', 0)

    for position_idx in range(start_idx, 7):
        print(f"\nPredicting Pokemon at position {position_idx}...")

        if position_idx == start_idx:
            direct_predictions = predict_next_pokemon(original_row, position_idx, models[position_idx])
            top_predictions = sorted(direct_predictions.items(), key=lambda x: x[1], reverse=True)

            candidate_predictions = [(pokemon, prob) for pokemon, prob in top_predictions
                                     if prob >= confidence_threshold]
            if not candidate_predictions:
                candidate_predictions = [top_predictions[0]]

            result = {
                'pokemon_idx': position_idx,
                'predictions': candidate_predictions,
                'propagation_order': 0
            }
            results.append(result)

            prediction_paths = []
            for pokemon, prob in candidate_predictions:
                new_row = original_row.copy()

                new_row[f'p2_pokemon{position_idx}_name'] = pokemon
                new_row['p2_current_pokemon'] = pokemon
                new_row['p2_number_of_pokemon_revealed'] = position_idx


                new_row['turn_id'] = initial_turn_id + (position_idx - start_idx + 1) * 3

                path = {
                    'pokemon': pokemon,
                    'probability': prob,
                    'state': new_row,
                    'node_id': current_node_id,
                    'order': 0
                }

                visualization_data['nodes'].append({
                    'id': current_node_id,
                    'label': f"{pokemon}\n{prob:.3f}",
                    'type': 'prediction',
                    'pokemon': pokemon,
                    'level': position_idx - 1,
                    'order': 0,
                    'probability': prob
                })

                visualization_data['edges'].append({
                    'source': root_id,
                    'target': current_node_id,
                    'type': 'prediction'
                })

                visualization_data['game_states'].append({
                    'node_id': current_node_id,
                    'state': new_row.copy(),
                    'level': position_idx - 1
                })

                prediction_paths.append(path)
                current_node_id += 1

            all_state_history[position_idx] = prediction_paths

        else:
            previous_position = position_idx - 1
            previous_predictions = all_state_history[previous_position]

            current_predictions = []

            for prev_path in previous_predictions:
                previous_state = prev_path['state']
                previous_pokemon = prev_path['pokemon']
                previous_prob = prev_path['probability']
                previous_node_id = prev_path['node_id']
                previous_order = prev_path['order']

                new_predictions = predict_next_pokemon(previous_state, position_idx, models[position_idx])
                top_new_predictions = sorted(new_predictions.items(), key=lambda x: x[1], reverse=True)

                candidate_new_predictions = [(pokemon, prob) for pokemon, prob in top_new_predictions
                                             if prob >= confidence_threshold]
                if not candidate_new_predictions:
                    candidate_new_predictions = [top_new_predictions[0]]

                for pokemon, prob in candidate_new_predictions:
                    new_state = previous_state.copy()

                    new_state[f'p2_pokemon{position_idx}_name'] = pokemon
                    new_state['p2_current_pokemon'] = pokemon
                    new_state['p2_number_of_pokemon_revealed'] = position_idx


                    new_state['turn_id'] = previous_state.get('turn_id', initial_turn_id) + 3

                    weighted_prob = prob * previous_prob

                    new_path = {
                        'pokemon': pokemon,
                        'probability': weighted_prob,
                        'original_prob': prob,
                        'parent_prob': previous_prob,
                        'state': new_state,
                        'node_id': current_node_id,
                        'parent_id': previous_node_id,
                        'order': previous_order + 1
                    }

                    visualization_data['nodes'].append({
                        'id': current_node_id,
                        'label': f"{pokemon}\n{prob:.3f}",
                        'type': 'prediction',
                        'pokemon': pokemon,
                        'level': position_idx - 1,
                        'order': previous_order + 1,
                        'probability': prob,
                        'weighted_prob': weighted_prob
                    })

                    visualization_data['edges'].append({
                        'source': previous_node_id,
                        'target': current_node_id,
                        'type': 'prediction'
                    })

                    visualization_data['game_states'].append({
                        'node_id': current_node_id,
                        'state': new_state.copy(),
                        'level': position_idx - 1
                    })

                    current_predictions.append(new_path)
                    current_node_id += 1

            weighted_predictions = defaultdict(float)
            total_parent_probability = defaultdict(float)
            pokemon_to_paths = defaultdict(list)

            for path in current_predictions:
                pokemon = path['pokemon']
                weighted_predictions[pokemon] += path['probability']
                total_parent_probability[pokemon] += path['parent_prob']
                pokemon_to_paths[pokemon].append(path)

            normalized_predictions = {}
            for pokemon, weighted_sum in weighted_predictions.items():
                if total_parent_probability[pokemon] > 0:
                    normalized_predictions[pokemon] = weighted_sum / total_parent_probability[pokemon]

            top_normalized = sorted(normalized_predictions.items(), key=lambda x: x[1], reverse=True)

            for pokemon, avg_prob in top_normalized:
                avg_node_id = current_node_id

                visualization_data['nodes'].append({
                    'id': avg_node_id,
                    'label': f"AVG {pokemon}\n{avg_prob:.3f}",
                    'type': 'average',
                    'pokemon': pokemon,
                    'level': position_idx - 1,
                    'probability': avg_prob
                })

                for path in pokemon_to_paths[pokemon]:
                    visualization_data['edges'].append({
                        'source': path['node_id'],
                        'target': avg_node_id,
                        'type': 'average'
                    })

                current_node_id += 1

            normalized_candidates = [(pokemon, prob) for pokemon, prob in top_normalized
                                     if prob >= confidence_threshold]
            if not normalized_candidates:
                normalized_candidates = [top_normalized[0]]

            max_order = max(path['order'] for path in current_predictions) if current_predictions else 0
            result = {
                'pokemon_idx': position_idx,
                'predictions': normalized_candidates,
                'propagation_order': max_order
            }
            results.append(result)

            next_level_paths = []
            for pokemon, avg_prob in normalized_candidates:
                best_path = max(pokemon_to_paths[pokemon], key=lambda p: p['probability'])

                new_path = {
                    'pokemon': pokemon,
                    'probability': avg_prob,
                    'state': best_path['state'],
                    'node_id': next(node['id'] for node in visualization_data['nodes']
                                    if node['type'] == 'average' and node['pokemon'] == pokemon
                                    and node['level'] == position_idx - 1),
                    'order': max_order
                }

                next_level_paths.append(new_path)

            all_state_history[position_idx] = next_level_paths

    return results, visualization_data

def create_prediction_visualization(visualization_data, output_path="prediction_visualization.png"):
    G = nx.DiGraph()

    node_colors = []
    node_sizes = []
    node_labels = {}

    for node in visualization_data['nodes']:
        G.add_node(node['id'])

        node_labels[node['id']] = node['label']

        if node['type'] == 'start':
            node_colors.append('lightblue')
            node_sizes.append(2000)
        elif node['type'] == 'average':
            node_colors.append('gold')
            node_sizes.append(1800)
        elif node['type'] == 'prediction':
            if node['order'] == 0:
                node_colors.append('lightgreen')
            elif node['order'] == 1:
                node_colors.append('lightcoral')
            elif node['order'] == 2:
                node_colors.append('lightsalmon')
            elif node['order'] == 3:
                node_colors.append('khaki')
            else:
                node_colors.append('plum')

            node_sizes.append(1500 - node['order'] * 200)

    for edge in visualization_data['edges']:
        if edge['type'] == 'prediction':
            G.add_edge(edge['source'], edge['target'], style='solid')
        else:
            G.add_edge(edge['source'], edge['target'], style='dashed')

    pos = {}

    nodes_by_level = defaultdict(list)
    for node in visualization_data['nodes']:
        nodes_by_level[node['level']].append(node['id'])

    max_level = max(nodes_by_level.keys())
    y_step = 0.8 / (max_level + 1)

    for level, level_nodes in nodes_by_level.items():
        y_pos = 0.9 - level * y_step

        nodes_count = len(level_nodes)
        if nodes_count == 1:
            pos[level_nodes[0]] = (0.5, y_pos)
        else:
            x_step = 0.8 / nodes_count
            for i, node_id in enumerate(level_nodes):
                x_pos = 0.1 + i * x_step
                pos[node_id] = (x_pos, y_pos)

    plt.figure(figsize=(20, 16))

    solid_edges = [(u, v) for u, v in G.edges() if G.get_edge_data(u, v).get('style') == 'solid']
    dashed_edges = [(u, v) for u, v in G.edges() if G.get_edge_data(u, v).get('style') == 'dashed']

    nx.draw_networkx_edges(G, pos, edgelist=solid_edges, width=1.0)
    nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, width=1.0, style='dashed')

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)

    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='black', label='Start'),
        Patch(facecolor='lightgreen', edgecolor='black', label='Order 0 (Direct)'),
        Patch(facecolor='lightcoral', edgecolor='black', label='Order 1'),
        Patch(facecolor='lightsalmon', edgecolor='black', label='Order 2'),
        Patch(facecolor='khaki', edgecolor='black', label='Order 3'),
        Patch(facecolor='plum', edgecolor='black', label='Order 4'),
        Patch(facecolor='gold', edgecolor='black', label='Weighted Average'),
        Line2D([0], [0], color='black', lw=1, label='Prediction Flow'),
        Line2D([0], [0], color='black', lw=1, linestyle='dashed', label='Aggregation')
    ]

    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    plt.axis('off')
    plt.title(f"Pokemon Prediction Tree with Weighted Propagation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Prediction visualization saved to {output_path}")

    return G


def evaluate_predictions(df, models, config):
    """Evaluate predictions on the dataset."""
    confidence_threshold = config['confidence_threshold']
    results = {
        'overall': {
            'correct': 0,
            'total': 0,
            'accuracy': 0
        }
    }

    for pokemon_idx in range(2, 7):
        results[f'pokemon_{pokemon_idx}'] = {
            'correct': 0,
            'total': 0,
            'accuracy': 0,
            'by_propagation_order': defaultdict(lambda: {'correct': 0, 'total': 0, 'accuracy': 0})
        }

    all_predictions = []

    valid_rows = df[df['p2_number_of_pokemon_revealed'] < 6].copy()
    total_rows = len(valid_rows)

    print(f"Processing {total_rows} game states for prediction...")

    for i, row in tqdm.tqdm(valid_rows.iterrows(), total=total_rows, desc="Evaluating predictions"):
        revealed_count = row['p2_number_of_pokemon_revealed']
        start_idx = revealed_count + 1
        if start_idx < 2:
            start_idx = 2

        row_dict = row.to_dict()
        prediction_results, _ = propagate_predictions(
            row_dict, models, start_idx, confidence_threshold=confidence_threshold
        )

        for result in prediction_results:
            pokemon_idx = result['pokemon_idx']
            propagation_order = result['propagation_order']
            top_prediction = result['predictions'][0][0]
            target = row['next_pokemon']

            is_correct = top_prediction == target

            results['overall']['total'] += 1
            results['overall']['correct'] += int(is_correct)

            results[f'pokemon_{pokemon_idx}']['total'] += 1
            results[f'pokemon_{pokemon_idx}']['correct'] += int(is_correct)

            results[f'pokemon_{pokemon_idx}']['by_propagation_order'][propagation_order]['total'] += 1
            results[f'pokemon_{pokemon_idx}']['by_propagation_order'][propagation_order]['correct'] += int(is_correct)

            all_predictions.append({
                'pokemon_idx': pokemon_idx,
                'propagation_order': propagation_order,
                'predicted': top_prediction,
                'actual': target,
                'correct': is_correct,
                'confidence': result['predictions'][0][1],
                'starting_position': revealed_count
            })

    results['overall']['accuracy'] = results['overall']['correct'] / results['overall']['total'] if results['overall']['total'] > 0 else 0

    for pokemon_idx in range(2, 7):
        key = f'pokemon_{pokemon_idx}'
        results[key]['accuracy'] = results[key]['correct'] / results[key]['total'] if results[key]['total'] > 0 else 0

        for order in results[key]['by_propagation_order']:
            order_results = results[key]['by_propagation_order'][order]
            order_results['accuracy'] = order_results['correct'] / order_results['total'] if order_results['total'] > 0 else 0

    return results, all_predictions


def visualize_results(results, all_predictions):
    plt.figure(figsize=(12, 6))

    pokemon_accuracies = [results[f'pokemon_{idx}']['accuracy'] for idx in range(2, 7)]
    pokemon_labels = [f'Pokemon {idx}' for idx in range(2, 7)]

    plt.bar(pokemon_labels, pokemon_accuracies)
    plt.title('Prediction Accuracy by Pokemon Position')
    plt.xlabel('Pokemon Position')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, acc in enumerate(pokemon_accuracies):
        plt.text(i, acc + 0.02, f'{acc:.2f}', ha='center')

    plt.tight_layout()
    plt.savefig('accuracy_by_pokemon_position.png')
    plt.close()

    prediction_df = pd.DataFrame(all_predictions)

    starting_positions = prediction_df.groupby(['starting_position', 'pokemon_idx'])['correct'].agg(['mean', 'count']).reset_index()
    starting_positions.columns = ['starting_position', 'pokemon_idx', 'accuracy', 'count']

    plt.figure(figsize=(14, 8))
    pivot_table = starting_positions.pivot_table(
        index='starting_position',
        columns='pokemon_idx',
        values='accuracy',
        aggfunc='mean'
    )
    sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.2f',
                cbar_kws={'label': 'Accuracy'})
    plt.title('Prediction Accuracy by Starting Position and Pokemon Position')
    plt.xlabel('Pokemon Position Being Predicted')
    plt.ylabel('Starting Position (Pokemon Revealed)')
    plt.tight_layout()
    plt.savefig('accuracy_by_starting_position.png')
    plt.close()

    plt.figure(figsize=(14, 8))
    count_pivot = starting_positions.pivot_table(
        index='starting_position',
        columns='pokemon_idx',
        values='count',
        aggfunc='sum'
    )
    sns.heatmap(count_pivot, annot=True, cmap='Greens', fmt='g',
                cbar_kws={'label': 'Number of Predictions'})
    plt.title('Number of Predictions by Starting Position and Pokemon Position')
    plt.xlabel('Pokemon Position Being Predicted')
    plt.ylabel('Starting Position (Pokemon Revealed)')
    plt.tight_layout()
    plt.savefig('counts_by_starting_position.png')
    plt.close()

    plt.figure(figsize=(14, 8))
    order_accuracy = prediction_df.groupby(['pokemon_idx', 'propagation_order'])['correct'].mean().reset_index()
    order_accuracy = order_accuracy.pivot(index='propagation_order', columns='pokemon_idx', values='correct')

    sns.heatmap(order_accuracy, annot=True, cmap='YlGnBu', fmt='.2f', cbar_kws={'label': 'Accuracy'})
    plt.title('Prediction Accuracy by Pokemon Position and Propagation Order')
    plt.xlabel('Pokemon Position')
    plt.ylabel('Propagation Order')
    plt.tight_layout()
    plt.savefig('accuracy_by_propagation_order.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    confidence_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    prediction_df['confidence_bin'] = pd.cut(prediction_df['confidence'], bins=confidence_bins)

    conf_accuracy = prediction_df.groupby('confidence_bin')['correct'].mean()
    conf_counts = prediction_df.groupby('confidence_bin').size()

    ax = conf_accuracy.plot(kind='bar', color='skyblue')
    plt.title('Accuracy by Confidence Level')
    plt.xlabel('Confidence Range')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    for i, acc in enumerate(conf_accuracy):
        plt.text(i, acc + 0.02, f'{acc:.2f}\n(n={conf_counts.iloc[i]})', ha='center')

    plt.tight_layout()
    plt.savefig('accuracy_by_confidence.png')
    plt.close()

    print("\nOverall Accuracy:", results['overall']['accuracy'])

    print("\nAccuracy by Pokemon Position:")
    for idx in range(2, 7):
        print(f"Pokemon {idx}: {results[f'pokemon_{idx}']['accuracy']:.4f}")

    print("\nAccuracy by Pokemon Position and Propagation Order:")
    for idx in range(2, 7):
        print(f"\nPokemon {idx}:")
        for order, order_results in sorted(results[f'pokemon_{idx}']['by_propagation_order'].items()):
            print(f"  Order {order}: {order_results['accuracy']:.4f} ({order_results['correct']}/{order_results['total']})")

    print("\nAccuracy by Starting Position:")
    for start_pos, group_data in starting_positions.groupby('starting_position'):
        avg_accuracy = group_data['accuracy'].mean()
        total_count = group_data['count'].sum()
        print(f"Starting at position {start_pos}: {avg_accuracy:.4f} (n={total_count})")

def find_first_pokemon_rows(df, n=1):
    mask = df['p2_number_of_pokemon_revealed'] == 1
    filtered = df[mask]

    if len(filtered) == 0:
        print("No rows found with only first Pokemon revealed. Using a row with minimal Pokemon revealed.")
        filtered = df.sort_values('p2_number_of_pokemon_revealed').head(n)

    return filtered.head(n)


def debug_prediction_process(df, models, config):
    """Debug a full prediction chain on a sample row to identify issues."""
    print("\n===== DEBUGGING PREDICTION PROCESS =====")

    mask = df['p2_number_of_pokemon_revealed'] == 1
    if mask.sum() > 0:
        sample_row = df[mask].iloc[0].to_dict()
        print(f"Selected a sample row with 1 Pokemon revealed: {sample_row['p2_pokemon1_name']}")
    else:
        sample_row = df.iloc[0].to_dict()
        print(f"No rows with exactly 1 Pokemon, using first row with {sample_row['p2_number_of_pokemon_revealed']} revealed")

    current_state = sample_row.copy()

    print("\nKey columns in initial state:")
    for idx in range(1, 7):
        print(f"p2_pokemon{idx}_name: {current_state.get(f'p2_pokemon{idx}_name', 'MISSING')}")

    print(f"p2_current_pokemon: {current_state.get('p2_current_pokemon', 'MISSING')}")
    print(f"p2_number_of_pokemon_revealed: {current_state.get('p2_number_of_pokemon_revealed', 'MISSING')}")
    print(f"turn_id: {current_state.get('turn_id', 'MISSING')}")

    for pokemon_idx in range(2, 7):
        model_info = models[pokemon_idx]
        print(f"\nModel for Pokemon {pokemon_idx} uses these features:")
        print(f"Categorical features: {model_info['categorical_features']}")
        print(f"Numerical features: {model_info['numerical_features']}")

    for position_idx in range(2, 7):
        print(f"\n--- Predicting Pokemon at position {position_idx} ---")

        print("\nCurrent state before prediction:")
        for feature in models[position_idx]['categorical_features'] + models[position_idx]['numerical_features']:
            if feature in current_state:
                print(f"{feature}: {current_state[feature]}")
            else:
                print(f"{feature}: MISSING")

        predictions = predict_next_pokemon(current_state, position_idx, models[position_idx])
        top_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_pokemon, confidence = top_predictions[0]

        print(f"\nTop prediction: {top_pokemon} with {confidence:.4f} confidence")

        print("\nUpdating state with prediction:")
        prev_state = current_state.copy()

        current_state[f'p2_pokemon{position_idx}_name'] = top_pokemon
        current_state['p2_current_pokemon'] = top_pokemon
        current_state['p2_number_of_pokemon_revealed'] = position_idx
        current_state['turn_id'] = current_state.get('turn_id', 0) + 3

        print("\nState changes:")
        for key, new_value in current_state.items():
            if key not in prev_state or prev_state[key] != new_value:
                print(f"{key}: {prev_state.get(key, 'None')} -> {new_value}")

    print("\n===== DEBUG COMPLETE =====")
    return current_state


def main():
    """Main entry point for testing Pokemon prediction models."""
    config = load_config()

    file_path = Path(config['file_path'])
    sample_size = config['sample_size']
    confidence_threshold = config['confidence_threshold']

    print("Loading data...")
    df = pd.read_csv(file_path)

    if sample_size is not None:
        print(f"Sampling {sample_size} rows from {len(df)} total rows")

        strat_col = 'p2_number_of_pokemon_revealed'

        counts_by_strata = df.groupby(strat_col).size()
        print(f"Distribution of revealed Pokemon in dataset:")
        for count, num_rows in counts_by_strata.items():
            print(f"  {count} Pokemon revealed: {num_rows} rows")

        samples_per_stratum = {}
        total_weight = sum([(i+1)**2 for i in range(6)])

        for i in range(6):
            weight = (i+1)**2 / total_weight
            samples_per_stratum[i] = max(int(sample_size * weight), 50)

        if sum(samples_per_stratum.values()) > sample_size:
            scale = sample_size / sum(samples_per_stratum.values())
            samples_per_stratum = {k: max(int(v * scale), 10) for k, v in samples_per_stratum.items()}

        print("Sampling strategy:")
        for strata, sample_count in samples_per_stratum.items():
            print(f"  {strata} Pokemon revealed: {sample_count} samples")

        sampled_dfs = []
        for strata_value, sample_count in samples_per_stratum.items():
            strata_df = df[df[strat_col] == strata_value]
            if len(strata_df) > 0:
                sampled = strata_df.sample(min(sample_count, len(strata_df)), random_state=42)
                sampled_dfs.append(sampled)

        df = pd.concat(sampled_dfs)
        print(f"Final sampled data shape: {df.shape}")
        print(f"Distribution of revealed Pokemon in sampled data:")
        for count, num_rows in df.groupby(strat_col).size().items():
            print(f"  {count} Pokemon revealed: {num_rows} rows")

    print("Loading models...")
    models = load_latest_models(config)

    if config['debug']:
        debug_prediction_process(df, models, config)

    for revealed_count in range(0, 5):
        mask = df['p2_number_of_pokemon_revealed'] == revealed_count
        example_rows = df[mask]

        if len(example_rows) > 0:
            example_row = example_rows.iloc[0].to_dict()
            start_idx = revealed_count + 1
            if start_idx < 2:
                start_idx = 2

            print(f"\nCreating prediction visualization for start at position {start_idx}")
            print(f"Pokemon revealed: {', '.join([example_row.get(f'p2_pokemon{i}_name', 'None') for i in range(1, start_idx)])}")

            try:
                results, viz_data = propagate_predictions(
                    example_row, models, start_idx, confidence_threshold=confidence_threshold
                )
                G = create_prediction_visualization(viz_data, f"prediction_viz_from_pos{start_idx}.png")
            except Exception as e:
                print(f"Failed to create visualization for start position {start_idx}: {e}")

    print("Evaluating models with propagation...")
    evaluation_results, all_predictions = evaluate_predictions(df, models, config)

    print("Visualizing results...")
    visualize_results(evaluation_results, all_predictions)

    print("\nEvaluation complete. Check the output files for visualizations.")


if __name__ == "__main__":
    main()