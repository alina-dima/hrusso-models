"""
This script uses Optuna to perform hyperparameter optimization on the
decoder-only transformer model.
"""
import os
import subprocess
import optuna


def objective(trial):
    """
    The objective function used for Optuna hyperparameter optimization.

    Args:
        trial: The current trial.

    Returns:
        The validation loss of the model.
    """
    # Define the hyperparameter search space
    dropout = trial.suggest_float('dropout', 0.2, 0.4, step=0.1)
    lr = trial.suggest_categorical('lr', [1e-5, 1e-4, 1e-3])
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    decoder_layers = trial.suggest_int('decoder_layers', 2, 4, step=2)
    decoder_attention_heads = trial.suggest_int('decoder_attention_heads', 2, 4, step=2)
    decoder_embed_dim = trial.suggest_categorical('decoder_embed_dim', [64, 128])
    decoder_ffn_embed_dim = trial.suggest_categorical('decoder_ffn_embed_dim', [256, 512, 1024])
    
    # Create one checkpoint directories per trial
    checkpoint_dir = f'checkpoints/optuna_trial_{trial.number}'
    log_file = os.path.join(checkpoint_dir, 'train.log')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # The fairseq command for the current hyperparameter setting
    command = [
        'fairseq-train', '../data-bin/hrusso_dataset',
        '--task', 'language_modeling',
        '--arch', 'transformer_lm', '--share-decoder-input-output-embed',
        '--dropout', str(dropout),
        '--decoder-layers', str(decoder_layers), '--decoder-embed-dim', str(decoder_embed_dim),
        '--decoder-ffn-embed-dim', str(decoder_ffn_embed_dim), '--decoder-attention-heads', str(decoder_attention_heads),
        '--max-tokens', '2048', '--update-freq', '4',
        '--optimizer', 'adam', '--adam-betas', '(0.9, 0.98)', '--clip-norm', '0.1',
        '--lr', str(lr),
        '--max-update', '10000', '--log-format', 'simple', '--log-interval', '100', '--patience', '5',
        '--save-dir', checkpoint_dir, '--batch-size', str(batch_size), '--criterion', 'cross_entropy',
        '--keep-last-epochs', '5'
    ]

    # Redirect training output to the log file
    with open(log_file, 'w') as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)

    # Evaluate the model and return the validation loss
    val_loss = get_validation_loss(checkpoint_dir)
    clean_up_checkpoints(checkpoint_dir)

    return val_loss


def get_validation_loss(checkpoint_dir):
    """
    Extract the validation loss from the training logs.

    Args:
        checkpoint_dir: The directory containing the training logs.

    Returns:
        The validation loss of the model
    """
    log_file = os.path.join(checkpoint_dir, 'train.log')
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in reversed(lines):
            if 'valid' in line and 'loss' in line:
                parts = line.strip().split()
                for i, part in enumerate(parts):
                    if part == 'loss':
                        return float(parts[i + 1])
    return float('inf')


def clean_up_checkpoints(checkpoint_dir):
    """
    Keep only the best checkpoint and remove all others.

    Args:
        checkpoint_dir: The directory containing the checkpoints.
    """
    # Keep only the best checkpoint
    best_checkpoint = None
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('checkpoint_best'):
            best_checkpoint = filename
            break

    # Remove all other checkpoint files
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('checkpoint') and filename != best_checkpoint:
            file_path = os.path.join(checkpoint_dir, filename)
            os.remove(file_path)


# Run the Optuna optimization
study = optuna.create_study(
    storage="sqlite:///db.sqlite3",
    study_name="hrusso_transformer",
    direction="minimize"
)
study.optimize(objective, n_trials=50)

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)
