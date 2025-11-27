import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import datetime
import warnings
warnings.filterwarnings('ignore')

from one_hot_encoder import CustomOneHotEncoder


def get_default_config():
    """Return default configuration values."""
    return {
        'file_path': Path("./data/processed/Parquets/all_pokemon_sequences.csv"),
        'models_dir': Path("./data/models/Models"),
        'use_rating_features': True,
        'use_current_pokemon': True,
        'use_previous_pokemon': True,
        'use_pokemon_count': True,
        'use_moves': False,
        'use_turn_info': True,
        'use_player_1': False,
        'n_estimators': 400,
        'max_depth': 50,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 1/3,
        'bootstrap': True,
        'class_weight': 'balanced_subsample',
        'random_state': 42,
        'criterion': 'entropy',
        'n_jobs': -1,
        'warm_start': True,
        'oob_score': True,
        'test_size': 0.2,
        'validation_size': 0.2,
        'show_feature_importance': True,
        'top_n_features': 20,
    }


def parse_args():
    """Parse command-line arguments."""
    defaults = get_default_config()
    parser = argparse.ArgumentParser(description="Train Pokemon prediction models")
    parser.add_argument('--data_path', type=Path, default=defaults['file_path'],
                        help='Path to the training data CSV file')
    parser.add_argument('--models_dir', type=Path, default=defaults['models_dir'],
                        help='Directory to save trained models')
    parser.add_argument('--n_estimators', type=int, default=defaults['n_estimators'],
                        help='Number of trees in the random forest')
    parser.add_argument('--max_depth', type=int, default=defaults['max_depth'],
                        help='Maximum depth of trees')
    parser.add_argument('--test_size', type=float, default=defaults['test_size'],
                        help='Proportion of data for testing')
    parser.add_argument('--validation_size', type=float, default=defaults['validation_size'],
                        help='Proportion of data for validation')
    parser.add_argument('--random_state', type=int, default=defaults['random_state'],
                        help='Random seed for reproducibility')
    parser.add_argument('--no_rating_features', action='store_true',
                        help='Disable rating features')
    parser.add_argument('--use_moves', action='store_true',
                        help='Include move features')
    parser.add_argument('--use_player_1', action='store_true',
                        help='Include player 1 features')
    parser.add_argument('--no_feature_importance', action='store_true',
                        help='Disable feature importance display')
    return parser.parse_args()


def args_to_config(args):
    """Convert parsed arguments to configuration dictionary."""
    defaults = get_default_config()
    return {
        'file_path': args.data_path,
        'models_dir': args.models_dir,
        'use_rating_features': not args.no_rating_features,
        'use_current_pokemon': defaults['use_current_pokemon'],
        'use_previous_pokemon': defaults['use_previous_pokemon'],
        'use_pokemon_count': defaults['use_pokemon_count'],
        'use_moves': args.use_moves,
        'use_turn_info': defaults['use_turn_info'],
        'use_player_1': args.use_player_1,
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': defaults['min_samples_split'],
        'min_samples_leaf': defaults['min_samples_leaf'],
        'max_features': defaults['max_features'],
        'bootstrap': defaults['bootstrap'],
        'class_weight': defaults['class_weight'],
        'random_state': args.random_state,
        'criterion': defaults['criterion'],
        'n_jobs': defaults['n_jobs'],
        'warm_start': defaults['warm_start'],
        'oob_score': defaults['oob_score'],
        'test_size': args.test_size,
        'validation_size': args.validation_size,
        'show_feature_importance': not args.no_feature_importance,
        'top_n_features': defaults['top_n_features'],
    }

def load_data(file_path):
    """Load data from a CSV file."""
    file_path = Path(file_path)
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def split_data_with_validation(df, config):
    """Split data into train/test/validation sets."""
    print("Creating train/test/validation split...")

    test_size = config['test_size']
    validation_size = config['validation_size']
    random_state = config['random_state']
    file_path = Path(config['file_path'])

    class_counts = df['next_pokemon'].value_counts()
    rare_classes = class_counts[class_counts < 3].index

    if len(rare_classes) > 0:
        print(f"Removing {len(rare_classes)} rare Pokemon classes with fewer than 3 occurrences before splitting")
        df = df[~df['next_pokemon'].isin(rare_classes)]
        print(f"Data shape after removing rare classes: {df.shape}")

    train_df, temp_df = train_test_split(
        df, test_size=(test_size + validation_size), random_state=random_state, stratify=df['next_pokemon']
    )

    temp_class_counts = temp_df['next_pokemon'].value_counts()
    temp_rare_classes = temp_class_counts[temp_class_counts < 2].index

    if len(temp_rare_classes) > 0:
        print(f"Removing {len(temp_rare_classes)} additional rare classes from temp split")
        temp_df = temp_df[~temp_df['next_pokemon'].isin(temp_rare_classes)]

    test_size_adjusted = test_size / (test_size + validation_size)
    test_df, validation_df = train_test_split(
        temp_df, test_size=(1 - test_size_adjusted), random_state=random_state, stratify=temp_df['next_pokemon']
    )

    print(f"Train set size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test set size: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"Validation set size: {len(validation_df)} ({len(validation_df)/len(df)*100:.1f}%)")

    validation_path = file_path.parent / "validation_pokemon_moves.csv"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_df.to_csv(validation_path, index=False)
    print(f"Validation set saved to {validation_path}")

    return pd.concat([train_df, test_df])

def preprocess_data(df, pokemon_idx, config):
    """Preprocess data for a specific Pokemon index."""
    print(f"Preprocessing data for Pokemon {pokemon_idx}...")
    processed_df = df.copy()

    target_column = f"next_pokemon"

    features = []

    if config['use_rating_features']:
        features.append('p2_rating')
        features.append('p1_rating')

    if config['use_current_pokemon']:
        features.append('p2_current_pokemon')
        if config['use_player_1']:
            features.append('p1_current_pokemon')

    if config['use_previous_pokemon']:
        if config['use_player_1']:
            features.extend([
                'p1_pokemon1_name', 'p1_pokemon2_name', 'p1_pokemon3_name', 'p1_pokemon4_name', 'p1_pokemon5_name',
                'p1_pokemon6_name',
            ])

        for i in range(1, pokemon_idx):
            features.append(f'p2_pokemon{i}_name')

    if config['use_pokemon_count']:
        features.append('p2_number_of_pokemon_revealed')
        if config['use_player_1']:
            features.append('p1_number_of_pokemon_revealed')

    if config['use_moves']:
        if config['use_player_1']:
            for i in range(1, 7):
                for j in range(1, 5):
                    features.append(f'p1_pokemon{i}_move{j}')

        for i in range(1, pokemon_idx):
            for j in range(1, 5):
                features.append(f'p2_pokemon{i}_move{j}')

    if config['use_turn_info']:
        features.append('turn_id')

    mask = processed_df['p2_number_of_pokemon_revealed'] >= pokemon_idx - 1
    processed_df = processed_df[mask]

    processed_df = processed_df.dropna(subset=[target_column])

    categorical_features = []
    numerical_features = []

    for feature in features:
        if feature in processed_df.columns:
            if processed_df[feature].dtype == 'object':
                categorical_features.append(feature)
            else:
                numerical_features.append(feature)

    for col in categorical_features:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].fillna('Unknown')

    for col in numerical_features:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())

    X = processed_df[features]
    y = processed_df[target_column]

    print(f"Final preprocessed data shape: {X.shape}")
    print(f"Number of categorical features: {len(categorical_features)}")
    print(f"Number of numerical features: {len(numerical_features)}")
    print(f"Target distribution sample:")
    print(y.value_counts().head(10))

    return X, y, categorical_features, numerical_features, processed_df


def get_model(config):
    """Create and return an untrained RandomForestClassifier with specified parameters."""
    return RandomForestClassifier(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        min_samples_split=config['min_samples_split'],
        min_samples_leaf=config['min_samples_leaf'],
        max_features=config['max_features'],
        bootstrap=config['bootstrap'],
        class_weight=config['class_weight'],
        criterion=config['criterion'],
        random_state=config['random_state'],
        n_jobs=config['n_jobs'],
        warm_start=config['warm_start'],
        oob_score=config['oob_score'],
        verbose=1
    )


def encode_features(X, categorical_features, numerical_features, encoder=None, fit=False):
    """Encode features using the CustomOneHotEncoder.
    
    Args:
        X: DataFrame with features
        categorical_features: List of categorical feature names
        numerical_features: List of numerical feature names
        encoder: Existing encoder to use (required if fit=False)
        fit: Whether to fit the encoder on X
        
    Returns:
        Tuple of (encoded_features, encoder, feature_names)
        
    Raises:
        ValueError: If fit=False and encoder is None
    """
    if not fit and encoder is None:
        raise ValueError('encoder must be provided when fit=False')
    
    if fit:
        encoder = CustomOneHotEncoder().fit(X, categorical_features)
    
    X_cat = encoder.transform(X, categorical_features)
    X_num = X[numerical_features].values if numerical_features else np.zeros((X.shape[0], 0))
    X_combined = np.hstack([X_cat, X_num])
    feature_names = encoder.get_feature_names() + numerical_features
    
    return X_combined, encoder, feature_names


def train(model, X, y, config):
    """Train a model with batch training for warm start."""
    print(f"Creating and training random forest with {config['n_estimators']} trees...")
    start_time = time.time()

    trees_per_batch = 200
    for i in range(0, config['n_estimators'], trees_per_batch):
        batch_end = min(i + trees_per_batch, config['n_estimators'])
        print(f"Training trees {i+1} to {batch_end}...")
        model.n_estimators = batch_end
        model.fit(X, y)
        elapsed_time = time.time() - start_time
        print(f"  Trained {batch_end} trees in {elapsed_time:.2f} seconds.")
        if config['oob_score']:
            print(f"  Current out-of-bag score: {model.oob_score_:.4f}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")
    return model


def train_model(X, y, categorical_features, numerical_features, pokemon_idx, config):
    """Orchestrate the training pipeline for a single Pokemon position model."""
    print(f"Training model for Pokemon {pokemon_idx}...")
    print("Splitting data into train and test sets...")

    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < 2].index
    if len(rare_classes) > 0:
        print(f"Removing {len(rare_classes)} rare Pokemon classes with only 1 occurrence")
        keep_mask = ~y.isin(rare_classes)
        X = X[keep_mask]
        y = y[keep_mask]
        print(f"Data shape after removing rare classes: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=config['random_state'], stratify=y
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    print("Creating and fitting custom one-hot encoder...")
    X_train_combined, encoder, feature_names = encode_features(
        X_train, categorical_features, numerical_features, fit=True
    )

    print("Transforming test features...")
    X_test_combined, _, _ = encode_features(
        X_test, categorical_features, numerical_features, encoder=encoder, fit=False
    )

    model = get_model(config)
    model = train(model, X_train_combined, y_train, config)

    print("Making predictions on test set...")
    y_pred = model.predict(X_test_combined)

    model_info = {
        'model': model,
        'encoder': encoder,
        'feature_names': feature_names,
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'X_train_combined': X_train_combined,
        'X_test_combined': X_test_combined
    }

    return model_info


def evaluate_model(model_info, pokemon_idx, config):
    """Evaluate model performance and generate metrics/plots."""
    print(f"\nModel Evaluation for Pokemon {pokemon_idx}:")
    model = model_info['model']
    y_test = model_info['y_test']
    y_pred = model_info['y_pred']

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    if hasattr(model, 'oob_score_'):
        print(f"Out-of-bag score: {model.oob_score_:.4f}")

    print("\nTop 10 class prediction counts:")
    print(pd.Series(y_pred).value_counts().head(10))

    top_classes = y_test.value_counts().head(15).index.tolist()
    print("\nClassification Report (top 15 classes):")
    print(classification_report(y_test, y_pred, labels=top_classes, zero_division=0))

    all_classes = sorted(list(set(y_test.unique()).union(set(y_pred))))
    print(f"\nConfusion Matrix (all {len(all_classes)} classes):")
    cm = confusion_matrix(
        y_test, y_pred,
        labels=all_classes
    )
    cm_df = pd.DataFrame(cm, index=all_classes, columns=all_classes)

    try:
        plt.figure(figsize=(2560/300, 1440/300), dpi=300)
        sns.heatmap(cm_df, annot=False, fmt='d', cmap='RdBu_r', xticklabels=False, yticklabels=False)
        plt.title(f'Confusion Matrix - Pokemon {pokemon_idx}')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_pokemon_{pokemon_idx}.png', dpi=300)
        plt.close()
        print(f"\nConfusion matrix plot saved as 'confusion_matrix_pokemon_{pokemon_idx}.png'")
    except Exception as e:
        print(f"Could not create confusion matrix plot: {e}")

    if config['show_feature_importance']:
        show_feature_importance(model_info, pokemon_idx, config)


def show_feature_importance(model_info, pokemon_idx, config):
    """Display and plot feature importance analysis."""
    top_n_features = config['top_n_features']
    print(f"\nFeature Importance Analysis for Pokemon {pokemon_idx}:")
    model = model_info['model']
    feature_names = model_info['feature_names']

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    importance_df['Base_Feature'] = importance_df['Feature'].apply(
        lambda x: x.split('_')[0] if '_' in x else x
    )

    base_importance = importance_df.groupby('Base_Feature')['Importance'].sum().reset_index()
    base_importance = base_importance.sort_values('Importance', ascending=False).reset_index(drop=True)

    print("\nAggregate Feature Importance (by original column):")
    for i, row in base_importance.head(top_n_features).iterrows():
        print(f"{i+1}. {row['Base_Feature']}: {row['Importance']:.4f}")

    print(f"\nTop {top_n_features} Individual Feature Values:")
    for i, row in importance_df.head(top_n_features).iterrows():
        print(f"{i+1}. {row['Feature']}: {row['Importance']:.4f}")

    try:
        plt.figure(figsize=(12, 10))
        plt.title(f'Aggregate Feature Importance - Pokemon {pokemon_idx}')
        top_n = min(15, len(base_importance))
        sns.barplot(x='Importance', y='Base_Feature', data=base_importance.head(top_n))
        plt.tight_layout()
        plt.savefig(f'aggregate_feature_importance_pokemon_{pokemon_idx}.png')
        plt.close()
        print(f"\nAggregate feature importance plot saved as 'aggregate_feature_importance_pokemon_{pokemon_idx}.png'")
    except Exception as e:
        print(f"Could not create aggregate feature importance plot: {e}")

    try:
        plt.figure(figsize=(14, 20))
        plt.title(f'Top {top_n_features} Individual Feature Importances - Pokemon {pokemon_idx}')
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n_features))
        plt.tight_layout()
        plt.savefig(f'individual_feature_importance_pokemon_{pokemon_idx}.png')
        plt.close()
        print(f"Top {top_n_features} individual feature importance plot saved as 'individual_feature_importance_pokemon_{pokemon_idx}.png'")
    except Exception as e:
        print(f"Could not create individual feature importance plot: {e}")


def save_model_package(model_info, output_dir, pokemon_idx):
    """Save the trained model package to disk."""
    output_dir = Path(output_dir)
    model_filename = output_dir / f"pokemon_prediction_model_{pokemon_idx}.joblib"
    print(f"\nSaving model package to '{model_filename}'...")

    model_package = {
        'model': model_info['model'],
        'encoder': model_info['encoder'],
        'feature_names': model_info['feature_names'],
        'categorical_features': model_info['categorical_features'],
        'numerical_features': model_info['numerical_features'],
        'pokemon_idx': pokemon_idx
    }

    joblib.dump(model_package, model_filename)
    print(f"Model package saved to '{model_filename}'")


def main():
    """Main entry point for training Pokemon prediction models."""
    args = parse_args()
    config = args_to_config(args)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config['models_dir']) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Models will be saved to: {output_dir}")

    df = load_data(config['file_path'])
    df = split_data_with_validation(df, config)

    for pokemon_idx in range(2, 7):
        print(f"\n{'='*80}")
        print(f"Processing Pokemon {pokemon_idx}")
        print(f"{'='*80}\n")

        X, y, categorical_features, numerical_features, processed_df = preprocess_data(df, pokemon_idx, config)
        model_info = train_model(X, y, categorical_features, numerical_features, pokemon_idx, config)
        evaluate_model(model_info, pokemon_idx, config)
        save_model_package(model_info, output_dir, pokemon_idx)


if __name__ == "__main__":
    main()