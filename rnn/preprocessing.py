"""
This script contains functions used to preprocess the data for training
and evaluating the RNN models.
"""

import pickle as pkl
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def file_to_sentence_list(filename):
    """Read a file and return a list of sentences."""
    with open(filename, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    return [sentence.strip() for sentence in sentences]


def create_sequences(data, tokenizer, max_sequence_len, total_words):
    """Creates sequences.

    Args:
        data: List of sentences.
        tokenizer: Tokenizer object.
        max_sequence_len: Maximum sequence length.
        total_words: Total number of words in the vocabulary.

    Returns:
        X: Input sequences.
        y: Target sequences.
    """
    input_sequences = []
    for line in data:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
    input_sequences = np.array(pad_sequences(input_sequences,
                                             maxlen=max_sequence_len,
                                             padding='pre'))
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)
    return X, y


def preprocess_data(train_file, val_file):
    """Preprocesses the data and returns sequences for training and validation.

    Args:
        train_file: Path to the training file.
        val_file: Path to the validation file.

    Returns:
        X_train: Input sequences for training.
        y_train: Target sequences for training.
        X_val: Input sequences for validation.
        y_val: Target sequences for validation.
        max_sequence_len: Maximum sequence length.
        total_words: Total number of words in the vocabulary.
    """
    train_data = file_to_sentence_list(train_file)
    validation_data = file_to_sentence_list(val_file)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data)
    with open('tokenizer.pkl', 'wb') as file:
        pkl.dump(tokenizer, file, protocol=pkl.HIGHEST_PROTOCOL)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in train_data:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max([len(seq) for seq in input_sequences])
    X_train, y_train = create_sequences(train_data, tokenizer,
                                        max_sequence_len, total_words)
    X_val, y_val = create_sequences(validation_data, tokenizer,
                                    max_sequence_len, total_words)

    return X_train, y_train, X_val, y_val, max_sequence_len, total_words
