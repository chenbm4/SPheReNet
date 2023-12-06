import optuna
from optuna.pruners import MedianPruner
import tensorflow as tf
from train_modularized import TrainData, build_model, train_model, load_config, load_weight_map

def objective(trial):
    # Load hyperparameters from configuration or define them here
    config = load_config('model_config/config.json')  # Assuming you have a load_config function

    # Define hyperparameters using Optuna
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    config['learning_rate'] = lr

    # Load data
    data = TrainData(config['train_data_file'], config['validation_split'])

    # Build model
    model = build_model()  # Pass any model hyperparameters if needed

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Load weight map
    weight_map = load_weight_map(config['weight_map_path'])

    # Train the model and return the metric to be optimized
    try:
        metric = train_model(model, optimizer, weight_map, data, config, trial)
        return metric
    except optuna.exceptions.TrialPruned as e:
        raise e

if __name__ == "__main__":
    pruner = MedianPruner()
    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(objective, n_trials=100)
    print(study.best_params)