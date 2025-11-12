# -*- coding: utf-8 -*-
"""
@author: sikdara

from custom_validate_callback import ValCallback

"""
import keras, os
import keras.backend as K
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
import wandb

class ValCallback(keras.callbacks.Callback):
    def __init__(self, test_generator, test_steps, model_name, wandb_log = False, save_model = False, checkpoint_path=None, best_only=False, checkpoint_freq=1):
        self.test_generator = test_generator
        self.test_steps = test_steps
        self.model_name = model_name
        self.wandb_log = wandb_log
        self.model_save = save_model
        self.checkpoint_path = checkpoint_path
        self.checkpoint_freq = checkpoint_freq
        self.model_path = checkpoint_path
        self.best_only = best_only
        self.val_acc = 0.0
        self.best_val_acc = 0.0

    def _implements_test_batch_hooks(self):
        return False

    def _implements_predict_batch_hooks(self):
        return False

    def on_epoch_end(self, epoch, logs={}):

        if tf.executing_eagerly():
            lr = self.model.optimizer.lr.numpy()
        else:
            lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)

        print(' - lr : ', lr)

        if self.wandb_log:
            wandb.log({'epoch' : epoch})
            wandb.log({'loss': logs['loss'], 'acc': logs['acc']})
            wandb.log({'lr': lr})
            
        # Check if we should validate this epoch
        if (epoch + 1) % self.test_steps == 0 and epoch != 0:
            
            loss, acc = self.model.evaluate(self.test_generator)
            self.val_acc = acc
            
            # Determine if we should save
            should_save = False
            
            if self.model_save and (epoch + 1) % self.checkpoint_freq == 0:
                if self.best_only:
                    # Only save if this is the best model so far
                    if acc > self.best_val_acc:
                        should_save = True
                        # Delete previous best model
                        if os.path.exists(self.model_path) and self.best_val_acc > 0.0:
                            try:
                                os.remove(self.model_path)
                                print(f'Removed previous checkpoint: {self.model_path}')
                            except Exception as e:
                                print(f'Could not remove old checkpoint: {e}')
                else:
                    # Save every checkpoint_freq epochs
                    should_save = True
                
                # Save the model if we should
                if should_save:
                    self.model_path = self.checkpoint_path.format(epoch + 1, lr, acc)
                    
                    # Create directory if it doesn't exist
                    model_dir = os.path.dirname(self.model_path)
                    if model_dir:
                        os.makedirs(model_dir, exist_ok=True)
                    
                    try:
                        self.model.save(self.model_path)
                        print(f'\n✓ Saved checkpoint: {self.model_path}')
                    except Exception as e:
                        print(f'\n✗ Failed to save checkpoint: {e}')
            
            # Update best validation accuracy
            self.best_val_acc = max(self.best_val_acc, self.val_acc)
               
            # Log validation accuracy and loss
            if self.wandb_log:
                wandb.log({'val_loss': loss, 'val_acc': acc})