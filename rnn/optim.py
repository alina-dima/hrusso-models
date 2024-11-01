"""
This file illustrates the Optuna hyperparameter optimization for RNN model.
"""

import random
import optuna
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from preprocessing import preprocess_data
from custom_callback import LoggingCallback


def build_model(total_words, emb_dim, max_sequence_len, units, rnn_type):
    """Build and return a TensorFlow sequential model.

    Args:
        total_words: Total number of words in the vocabulary.
        emb_dim: Embedding dimension.
        max_sequence_len: Maximum length of the input sequence.
        units: Number of units in the RNN layer.
        rnn_type: Type of RNN layer to use (i.e., LSTM or GRU).

    Returns:
            model: The sequential model.
    """
    model = Sequential()
    model.add(Embedding(total_words, emb_dim, input_length=max_sequence_len - 1))
    if rnn_type == 'LSTM':
        model.add(LSTM(units))
    else:
        model.add(GRU(units))
    model.add(Dense(total_words, activation='softmax'))
    return model


def random_seed(seed):
    """Sets random seeds.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def objective(trial, seed=42):
    """Objective function for hyperparameter optimization.

    Args:
        trial: The current Optuna trial.
        seed: Random seed value.
    """
    random_seed(seed)

    emb_dim = trial.suggest_categorical('emb_dim', [50, 100])
    units = trial.suggest_categorical('units', [128, 256, 512])
    lr = trial.suggest_categorical('lr', [1e-3, 1e-4])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    rnn_type = trial.suggest_categorical('rnn_type', ['LSTM', 'GRU'])
    optimizer_type = trial.suggest_categorical('optimiser', ['sgd', 'adam'])

    X_train, y_train, X_val, y_val, max_len, n_words = preprocess_data(
        '../data/train_data.txt', '../data/val_data.txt')

    model = build_model(n_words, emb_dim, max_len, units, rnn_type)

    if optimizer_type == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy', TopKCategoricalAccuracy(k=3, name='top_3_acc')])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint_path = f'checkpoints/optuna_trial_{trial.number}/'
    model_checkpoint = ModelCheckpoint(f'{checkpoint_path}best_model.h5', monitor='val_loss', save_best_only=True)
    logging_callback = LoggingCallback(checkpoint_path)

    model.fit(X_train, y_train,
              epochs=100,
              batch_size=batch_size,
              validation_data=(X_val, y_val),
              callbacks=[early_stopping, model_checkpoint, logging_callback],
              verbose=2)

    # Get and return the validation loss
    score = model.evaluate(X_val, y_val, verbose=0)
    return score[0]


# Run Optuna hyperparameter optimization
study = optuna.create_study(
    storage="sqlite:///db.sqlite3",
    study_name="rnn_hyperparameter_optimization",
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective)

print("Best hyperparameters: ", study.best_params)
