"""
This file contains a custom callback class that logs the loss,
accuracy and validation at the end of each epoch.
"""

import os
import tensorflow as tf


class LoggingCallback(tf.keras.callbacks.Callback):
    """
    Callback to log training metrics to a CSV file.
    """
    def __init__(self, checkpoint_file):
        super(LoggingCallback, self).__init__()
        self.log_file = checkpoint_file + 'training_log.csv'
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write('epoch,loss,accuracy,top_3_acc,val_loss,val_accuracy,val_top_3_acc\n')

    def on_epoch_end(self, epoch, logs=None):
        """Log the metrics at the end of each epoch."""
        log_data = [
            epoch,
            logs.get('loss'),
            logs.get('accuracy'),
            logs.get('top_3_acc'),
            logs.get('val_loss'),
            logs.get('val_accuracy'),
            logs.get('val_top_3_acc')
        ]
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(','.join(map(str, log_data)) + '\n')
