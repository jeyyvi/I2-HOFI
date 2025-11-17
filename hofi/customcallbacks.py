# # -*- coding: utf-8 -*-
# """
# @author: sikdara

# from custom_validate_callback import ValCallback

# """
# import keras, os
# import keras.backend as K
# from sklearn.metrics import accuracy_score
# import numpy as np
# import tensorflow as tf
# import wandb

# class ValCallback(keras.callbacks.Callback):
#     def __init__(self, test_generator, test_steps, model_name, wandb_log = False, save_model = False, checkpoint_path=None, best_only=False, checkpoint_freq=1):
#         self.test_generator = test_generator
#         self.test_steps = test_steps
#         self.model_name = model_name
#         self.wandb_log = wandb_log
#         self.model_save = save_model
#         self.checkpoint_path = checkpoint_path
#         self.checkpoint_freq = checkpoint_freq
#         self.model_path = checkpoint_path
#         self.best_only = best_only
#         self.val_acc = 0.0
#         self.best_val_acc = 0.0

#     def _implements_test_batch_hooks(self):
#         return False

#     def _implements_predict_batch_hooks(self):
#         return False

#     def on_epoch_end(self, epoch, logs={}):

#         if tf.executing_eagerly():
#             lr = self.model.optimizer.lr.numpy()
#         else:
#             lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)

#         print(' - lr : ', lr)

#         if self.wandb_log:
#             wandb.log({'epoch' : epoch})
#             wandb.log({'loss': logs['loss'], 'acc': logs['acc']})
#             wandb.log({'lr': lr})
            
#         # Check if we should validate this epoch
#         if (epoch + 1) % self.test_steps == 0 and epoch != 0:
            
#             loss, acc = self.model.evaluate(self.test_generator)
#             self.val_acc = acc
            
#             # Determine if we should save
#             should_save = False
            
#             if self.model_save and (epoch + 1) % self.checkpoint_freq == 0:
#                 if self.best_only:
#                     # Only save if this is the best model so far
#                     if acc > self.best_val_acc:
#                         should_save = True
#                         # Delete previous best model
#                         if os.path.exists(self.model_path) and self.best_val_acc > 0.0:
#                             try:
#                                 os.remove(self.model_path)
#                                 print(f'Removed previous checkpoint: {self.model_path}')
#                             except Exception as e:
#                                 print(f'Could not remove old checkpoint: {e}')
#                 else:
#                     # Save every checkpoint_freq epochs
#                     should_save = True
                
#                 # Save the model if we should
#                 if should_save:
#                     self.model_path = self.checkpoint_path.format(epoch + 1, lr, acc)
                    
#                     # Create directory if it doesn't exist
#                     model_dir = os.path.dirname(self.model_path)
#                     if model_dir:
#                         os.makedirs(model_dir, exist_ok=True)
                    
#                     try:
#                         self.model.save(self.model_path)
#                         print(f'\n✓ Saved checkpoint: {self.model_path}')
#                     except Exception as e:
#                         print(f'\n✗ Failed to save checkpoint: {e}')
            
#             # Update best validation accuracy
#             self.best_val_acc = max(self.best_val_acc, self.val_acc)
               
#             # Log validation accuracy and loss
#             if self.wandb_log:
#                 wandb.log({'val_loss': loss, 'val_acc': acc})

# -*- coding: utf-8 -*-
"""
Enhanced callbacks for saving all checkpoints and tracking best model
"""
import keras, os
import keras.backend as K
import numpy as np
import tensorflow as tf
import wandb
import json

class ValCallback(keras.callbacks.Callback):
    def __init__(self, test_generator, test_steps, model_name, wandb_log=False, 
                 save_model=False, checkpoint_path=None, best_only=False, 
                 checkpoint_freq=1, save_all=True):
        """
        Args:
            save_all (bool): If True, saves every checkpoint regardless of best_only setting
        """
        self.test_generator = test_generator
        self.test_steps = test_steps
        self.model_name = model_name
        self.wandb_log = wandb_log
        self.model_save = save_model
        self.checkpoint_path = checkpoint_path
        self.checkpoint_freq = checkpoint_freq
        self.model_path = checkpoint_path
        self.best_only = best_only
        self.save_all = save_all
        self.val_acc = 0.0
        self.best_val_acc = 0.0
        self.best_model_path = None
        
        # Create history tracking
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # Create directory for saving history
        if self.checkpoint_path:
            history_dir = os.path.dirname(self.checkpoint_path)
            if history_dir:
                os.makedirs(history_dir, exist_ok=True)

    def _implements_test_batch_hooks(self):
        return False

    def _implements_predict_batch_hooks(self):
        return False

    def on_epoch_end(self, epoch, logs={}):
        # Get learning rate
        if tf.executing_eagerly():
            lr = self.model.optimizer.lr.numpy()
        else:
            lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)

        print(' - lr : ', lr)
        
        # Store training metrics
        train_loss = logs.get('loss', 0)
        train_acc = logs.get('acc', 0)
        
        # Log to WandB
        if self.wandb_log:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'lr': lr
            })
        
        # Run validation if it's time
        if (epoch + 1) % self.test_steps == 0 and epoch != 0:
            loss, acc = self.model.evaluate(self.test_generator, verbose=0)
            self.val_acc = acc
            val_loss = loss
            
            print(f' - val_loss: {val_loss:.4f} - val_acc: {acc:.4f}')
            
            # Update history
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(float(train_loss))
            self.history['train_acc'].append(float(train_acc))
            self.history['val_loss'].append(float(val_loss))
            self.history['val_acc'].append(float(acc))
            self.history['lr'].append(float(lr))
            
            # Save history to JSON
            history_path = os.path.join(
                os.path.dirname(self.checkpoint_path), 
                'training_history.json'
            )
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            
            # Log validation metrics to WandB
            if self.wandb_log:
                wandb.log({
                    'val_loss': val_loss,
                    'val_acc': acc
                })
            
            # Determine if we should save
            is_best = acc > self.best_val_acc
            should_save_regular = (epoch + 1) % self.checkpoint_freq == 0
            
            if self.model_save:
                # Save regular checkpoint (every checkpoint_freq epochs)
                if self.save_all and should_save_regular:
                    regular_path = self.checkpoint_path.format(epoch + 1, lr, acc)
                    model_dir = os.path.dirname(regular_path)
                    if model_dir:
                        os.makedirs(model_dir, exist_ok=True)
                    
                    try:
                        self.model.save(regular_path)
                        print(f'\n✓ Saved checkpoint: {regular_path}')
                    except Exception as e:
                        print(f'\n✗ Failed to save checkpoint: {e}')
                
                # Save best model separately
                if is_best:
                    # Remove previous best model
                    if self.best_model_path and os.path.exists(self.best_model_path):
                        try:
                            os.remove(self.best_model_path)
                            print(f'Removed previous best: {self.best_model_path}')
                        except Exception as e:
                            print(f'Could not remove old best model: {e}')
                    
                    # Save new best model with special naming
                    best_path = self.checkpoint_path.replace(
                        '|epoch:{:03d}', 
                        '|BEST|epoch:{:03d}'
                    ).format(epoch + 1, lr, acc)
                    
                    model_dir = os.path.dirname(best_path)
                    if model_dir:
                        os.makedirs(model_dir, exist_ok=True)
                    
                    try:
                        self.model.save(best_path)
                        self.best_model_path = best_path
                        print(f'\n🏆 Saved BEST model: {best_path}')
                    except Exception as e:
                        print(f'\n✗ Failed to save best model: {e}')
            
            # Update best validation accuracy
            if is_best:
                self.best_val_acc = acc
                print(f'New best validation accuracy: {acc:.4f}')


class MetricsCallback(keras.callbacks.Callback):
    """Additional callback for tracking detailed metrics"""
    
    def __init__(self, log_dir='./logs'):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
    def on_epoch_end(self, epoch, logs=None):
        # Save epoch metrics
        metrics_file = os.path.join(self.log_dir, 'epoch_metrics.json')
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []
        
        epoch_data = {
            'epoch': epoch + 1,
            'loss': float(logs.get('loss', 0)),
            'acc': float(logs.get('acc', 0)),
        }
        
        if 'val_loss' in logs:
            epoch_data['val_loss'] = float(logs.get('val_loss', 0))
            epoch_data['val_acc'] = float(logs.get('val_acc', 0))
        
        all_metrics.append(epoch_data)
        
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)