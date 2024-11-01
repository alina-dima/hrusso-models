"""
This module contains utility functions used in the training
and evaluation of the n-gram models.
"""

import time
import nltk
from nltk.tokenize import word_tokenize
from nltk import ngrams
nltk.download('punkt_tab')


def read_and_tokenize_data(file_path):
    """
    Reads the data from the file and tokenizes it.

    Args:
        file_path: The path to the file to read.

    Returns:
        The tokenized data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    data = [sentence.strip() for sentence in data]
    tokenized_data = [word_tokenize(sentence) for sentence in data]
    return tokenized_data


def pad_left(sent, pad_token='<s>', n=3):
    """
    Pads the left side of the sentence with pad_token up to the length n.

    Args:
    sent: The tokenized sentence to pad.
    pad_token: The token to use for padding.
    n: The length of n-gram context.

    Returns:
    The padded sentence.
    """
    padding = [pad_token] * (n - len(sent) + 1)
    return padding + sent


def evaluate(model, tokenized_data, n):
    """
    Evaluates the model on a given tokenized dataset.

    Args:
        model: The n-gram model to evaluate.
        tokenized_data: The tokenized data.
        n: The order of the n-gram model.

    Returns:
        The perplexity, accuracy, inference time and list of n-grams.
    """
    padded_data = [pad_left(sent, n=n) for sent in tokenized_data]
    ngrams_list = [ngram for sentence_tokens in padded_data for ngram in ngrams(sentence_tokens, n)]
    valid_ngrams_list = [ngram for ngram in ngrams_list if ngram[-1] in model.vocab]

    perplexity = model.perplexity(valid_ngrams_list)

    count_correct_pred = 0
    counts = 0
    times = []
    for ngram in valid_ngrams_list:
        counts += 1
        context = ngram[:-1]
        start = time.time()
        next_word = model.generate(3, text_seed=context)
        end = time.time()
        times.append(end-start)
        if ngram[-1] in next_word:
            count_correct_pred += 1

    accuracy = 100*count_correct_pred/len(valid_ngrams_list)
    return perplexity, accuracy, sum(times) / len(times)
