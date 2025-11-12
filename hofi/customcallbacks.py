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
        # print('++++++++++++++ best only +++++++++++++++', self.best_only)

    # New added -------------------------------------------------------------------

    def _implements_test_batch_hooks(self):
        # Keras checks this; return False if your callback does not implement
        # per-test-batch hooks (most custom callbacks don't).
        return False

    def _implements_predict_batch_hooks(self):
        # Same for predict-time batch hooks.
        return False


    def _implements_test_batch_hooks(self):
        return False
    
    # ---------------------------------------------------------------------------------------

    def on_epoch_end(self, epoch, logs={}):

        if tf.executing_eagerly():
            lr = self.model.optimizer.lr.numpy()
        else:
            lr = self.model.optimizer.learning_rate.numpy()
        print(' - lr : ', lr)

        if self.wandb_log:
            # Log epoch, training accuracy and loss
            wandb.log({'epoch' : epoch})
            wandb.log({'loss': logs['loss'], 'acc': logs['acc']})
            wandb.log({'lr': lr})
            

        if (epoch + 1) % self.test_steps == 0 and epoch != 0:
            
            loss, acc = self.model.evaluate(self.test_generator) # change to model.evaluate()

            if self.model_save:
                # print('++++++++++++++ Saving model from customcallback +++++++++++++++', self.checkpoint_path)
                if self.best_only:
                    if acc > self.best_val_acc and self.val_acc> 0.0: # delete previous file and replace with new one
                        # Check if the file exists before trying to delete it
                        if os.path.exists(self.model_path):
                            os.remove(self.model_path)
                            
                        self.model_path = self.checkpoint_path.format(epoch, lr, acc)
                        
                elif (epoch + 1) % self.checkpoint_freq == 0 and epoch != 0: 
                    # if best_only == False, then save based on checkpoint freq
                    self.model_path = self.checkpoint_path.format(epoch, lr, acc)
                
                if self.best_val_acc > 0.0:
                    self.model.save(self.model_path)

            self.val_acc = acc
            self.best_val_acc = max(self.best_val_acc, self.val_acc)
               
            # Log validation accuracy and loss
            if self.wandb_log:
                wandb.log({'val_loss': loss, 'val_acc': acc})