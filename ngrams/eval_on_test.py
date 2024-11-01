"""
This script evaluates the performance of a trained n-gram model
on the test data.
"""
import dill as pkl
from utils import read_and_tokenize_data, evaluate


def main():
    model_path = 'n_grams_StupidBackoff_4.pkl'
    test_data_path = '../data/test_data.txt'
    n = 4  # Set the order of the n-gram model

    with open(model_path, 'rb') as file:
        model = pkl.load(file)
    tokenized_test = read_and_tokenize_data(test_data_path)
    ppl, acc, it = evaluate(model, tokenized_test, n)

    print(f'Perplexity: {ppl}')
    print(f'Top-3 Accuracy: {acc:.3f}%')
    print(f'Inference Time: {it}s')


if __name__ == "__main__":
    main()
