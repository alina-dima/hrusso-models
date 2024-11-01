# RNN Architecture

This folder contains the files relating to training and optimising the RNN models. Below you can find a description of each file.

## Description of Contents

- **`best_model.h5`**
  - This file stores the best model resulting from the Optuna optimisation. This model can be loaded and directly used.

- **`custom_callback.py`**
  - Contains custom callback functions for the model's training process, such as early stopping, model checkpointing, or logging metrics. These functions allow more control and customization over the training loop.

- **`db.sqlite3`**
  - A SQLite3 database file which stores information about the trials in the Optuna study. The file can be uploaded on https://optuna.github.io/optuna-dashboard/ to consult the optimisation history. There one can observe the validation loss as a function of trial and the importance of each hyperparameter.

- **`evaluate.py`**
  - A Python script containing the functions used to evaluate the best model in terms of loss, accuracy and inference time.

- **`optim.py`**
  - A Python script used to define the optimisation process for the RNN models. This file contains the code used to set up the RNN models alongside the code for setting up the Optuna optimisation process.

- **`preprocessing.py`**
  - This Python script handles data preprocessing tasks ensuring consistency between training and testing.

- **`tokenizer.pkl`**
  - A Pickle file that contains the saved tokenizer. Use this tokenizer for consistent preprocessing during inference during testing or after deployment.

- **`training_log_optuna_trial_40.csv`**
  - Contains the training log of the 41st trial, where the loss and accuracy were calculated on the training and validation data on each epoch.
