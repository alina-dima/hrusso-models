"""
This script evaluates the best RNN model on the test data.
"""

import time
import pickle as pkl
import tensorflow as tf
from preprocessing import file_to_sentence_list
from preprocessing import create_sequences


def get_max_seq_length(train_data_path, tokenizer):
    """
    Get the maximum sequence length from the training data.

    Args:
        train_data_path: Path to the training data file.
        tokenizer: Tokenizer object from training.
    """
    train_data = file_to_sentence_list(train_data_path)
    train_input_sequences = []
    for line in train_data:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            train_input_sequences.append(n_gram_sequence)

    return max([len(seq) for seq in train_input_sequences])


def main():
    """
    Main function to evaluate the best model on the test data.
    """
    test_data = file_to_sentence_list('../data/test_data.txt')

    # Loads the tokenizer obtained from training.
    file = open('tokenizer.pkl', 'rb')
    tokenizer = pkl.load(file)
    total_words = len(tokenizer.word_index) + 1
    max_sequence_len = get_max_seq_length('../data/train_data.txt', tokenizer)
    X_test, y_test = create_sequences(test_data, tokenizer, max_sequence_len, total_words)

    best_model = tf.keras.models.load_model('best_model.h5', compile=False)
    best_model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')])
    start = time.time()
    test_loss, test_accuracy, test_top_3_accuracy = best_model.evaluate(X_test,
                                                                        y_test,
                                                                        verbose=0)
    end = time.time()

    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')
    print(f'Test Top-3 Accuracy: {test_top_3_accuracy}')
    print(f'Test Inference Time: {(end - start) / len(X_test)}')


if __name__ == '__main__':
    main()
