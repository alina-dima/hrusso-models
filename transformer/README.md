# Transformer Architecture

This folder contains the files relating to training and optimising the transformer models. Below you can find a description of each file.

## Description of Contents

- **`optuna_trial_34/`** (Folder)
  - Contains the training log of the 35th trial. This folder contains the best pre-trained transformer model saved as `checkpoint_best.pt` and the associated training log `train.log` containing performance measurement metrics for the training data and for the validation data.

- **`db.sqlite3`**
  - A SQLite3 database file which stores information about the trials in the Optuna study. The file can be uploaded on https://optuna.github.io/optuna-dashboard/ to consult the optimisation history. There one can observe the validation loss as a function of trial and the importance of each hyperparameter.

- **`evaluate.py`**
  - A Python script containing the functions used to evaluate the best model on the test data in terms of accuracy and inference time.

- **`hp_optimisation.py`**
  - A Python script used to define the optimisation process for the transformer models. This file contains the code used to set up the transformer model alongside the code for setting up the Optuna optimisation process.

- **`preproces.sh`**
  - This script handles data preprocessing by tokenizing and binarizing the data to make it suitable for a fairseq model.

- **`sequence_utils.py`**
  - Provides an implementation of sequence generating utility functions to be used on potentially non-fairseq unprocessed data such as user generated data or even the unprocessed testing data.
