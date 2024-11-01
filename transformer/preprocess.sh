#!/bin/bash
# Preprocesses the data for the fairseq model.
# The data is tokenized (using the space as a delimiter) and binarized.

fairseq-preprocess --only-source \
    --trainpref ../data/train_data.txt \
    --validpref ../data/val_data.txt \
    --testpref ../data/test_data.txt \
    --destdir ../data-bin/hrusso_dataset \
    --workers 4