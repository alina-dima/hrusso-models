"""
This script contains utility functions for generating sequences of words
from a list of sentences, to be used for testing the pre-trained transformer model.
"""

from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences


def read_sentences(file_path):
    """
    Reads the sentences from a file.

    Args:
        file_path: The path to the file containing the sentences
                   separated by newlines.

    Returns:
        The list of sentences.
    """
    with open(file_path, 'r') as file:
        sentences = file.readlines()
    return [sentence.strip() for sentence in sentences]


def tokenize_sentences(sentences):
    """
    Tokenizes a list of sentences using the NLTK library.

    Args:
        sentences: A list of sentences.

    Returns:
        The list of tokenized sentences.
    """
    return [word_tokenize(sentence) for sentence in sentences]


def generate_sequences(tokenized_sentences, sequence_length):
    """
    Generates sequences of words from the tokenized sentences.

    Args:
        tokenized_sentences: A list of tokenized sentences.
        sequence_length: The length of the input sequences.

    Returns:
        A list of input-output sequence pairs.
    """
    sequences = []
    for sentence in tokenized_sentences:
        for i in range(len(sentence) - sequence_length):
            sequences.append((sentence[i:i+sequence_length],
                              sentence[i+sequence_length]))
    return sequences


def pad_sequences_custom(sequences, sequence_length):
    """
    Pads the input sequences to the specified length (e.g.,
    if the sequences were generated at the beginning of a sentence,
    or if the source sentence is shorter than the desired length).

    Args:
        sequences: A list of input-output sequence pairs.
        sequence_length: The length to which the sequences should be padded.

    Returns:
        A list of padded input-output sequence pairs.
    """
    padded_sequences = []
    for input_seq, next_word in sequences:
        if len(input_seq) < sequence_length:
            input_seq = pad_sequences([input_seq], maxlen=sequence_length,
                                      padding='pre')[0]
        padded_sequences.append((input_seq, next_word))
    return padded_sequences


def get_seq_nextword(file_path, seq_len=3):
    """
    Reads the sentences from a file, tokenizes them, generates sequences
    of words, and pads the sequences to the specified length.

    Args:
        file_path: The path to the file containing the sentences.
        seq_len: The length of the input sequences.

    Returns:
        A tuple containing the input sequences and the target next words.
    """
    sentences = read_sentences(file_path)
    tokenized_sentences = tokenize_sentences(sentences)
    sequences = generate_sequences(tokenized_sentences, seq_len)
    padded_sequences = pad_sequences_custom(sequences, seq_len)

    inputs, targets = [], []
    for seq, next_word in padded_sequences:
        inputs.append(' '.join(seq))
        targets.append(next_word)

    return inputs, targets
