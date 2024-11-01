"""
This script trains and saves n-gram models of different orders and using
different smoothing techniques.

The perplexity, accuracy and inference time are also calculated
for each model on the validation data.
"""

import dill as pkl
from utils import read_and_tokenize_data, evaluate
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, Laplace, Lidstone, StupidBackoff


def initialize_model(smoothing, n):
    """
    Initializes the n-gram model with the given smoothing type and order.

    Args:
        type: The type of the model to initialize.
        n: The order of the n-gram model.

    Returns:
        The initialized n-gram model.
    """
    if smoothing == 'MLE':
        model = MLE(order=n)
    elif smoothing == 'Laplace':
        model = Laplace(order=n)
    elif smoothing == 'Lidstone':
        model = Lidstone(gamma=0.5, order=n)
    else:
        model = StupidBackoff(alpha=0.4, order=n)
    return model


def main():
    tokenized_train = read_and_tokenize_data('../data/train_data.txt')
    tokenized_validation = read_and_tokenize_data('../data/val_data.txt')
    models = ['MLE', 'Laplace', 'Lidstone', 'StupidBackoff']
    orders = [2, 3, 4, 5, 6]

    for model_type in models:
        for n in orders:
            model = initialize_model(model_type, n)

            # Train the model on the training data
            train_everygrams, padded_train_sents = padded_everygram_pipeline(n, tokenized_train)
            model.fit(train_everygrams, padded_train_sents)

            # Evaluate the model on the validation data
            ppl, acc, it = evaluate(model, tokenized_validation, n)
            print(f'Perplexity: {ppl} for {model_type} and n={n}')
            print(f'Top-3 Accuracy: {acc:.3f}% for {model_type} and n={n}')
            print(f'Inference Time: {it}s for {model_type} and n={n}\n')

            # Save the model
            with open('n_grams_' + model_type + '_' + str(n) + '.pkl', 'wb') as f:
                pkl.dump(model, f)


if __name__ == "__main__":
    main()
