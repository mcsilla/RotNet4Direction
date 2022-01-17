"""Module providing Trainer class"""

import os
import tensorflow as tf
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataloader import GenericDataLoader
from model.model import RotNet4

class Trainer:
    """Class for managing training.

    Args:
        config:
            python dictionary containing training configuration
    """
    def __init__(self, config):
        self.config = config
        # self._assert_config()

        # Train Dataset
        train_dataloader = GenericDataLoader(self.config[
            'train_dataset_config'])
        self.train_dataset = train_dataloader.get_dataset(self.config['image_size'])

        # Validation Dataset
        val_dataloader = GenericDataLoader(self.config[
            'val_dataset_config'])
        self.val_dataset = val_dataloader.get_dataset(self.config['image_size'])

        self._model = None
        self.initial_epoch = 0

    def continue_running(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.config['checkpoint_dir'])
        print('Latest checkpoint: ', latest_checkpoint)
        if latest_checkpoint is not None:
            self.initial_epoch = int(latest_checkpoint.split("_")[1])
        return latest_checkpoint

    @property
    def model(self):
        """Property returning model being trained."""

        if self._model is not None:
            return self._model

        with self.config['strategy'].scope():
            decay_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=self.config['initial_learning_rate'],
                end_learning_rate=self.config['end_learning_rate'],
                decay_steps=self.config['decay_steps'],
                power=self.config['power'])
            self._model = RotNet4(self.config['image_size']).model

            self._model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=decay_schedule
                ),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )
            latest_ckpt = self.continue_running()
            if latest_ckpt is not None:
                self._model.load_weights(latest_ckpt)

            return self._model



    def _get_checkpoint_filename_format(self):
        return os.path.join(self.config['checkpoint_dir'],
                            self.config['checkpoint_file_prefix'] +
                            "{epoch:04d}")

    def _get_logger_callback(self):
        return tf.keras.callbacks.TensorBoard(log_dir=self.config['log_dir'], update_freq='batch')



    def train(self):
        """Trainer entry point.

        Runs .fit() on loaded model.
        """


        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self._get_checkpoint_filename_format(),
                monitor='val_loss',
                # save_best_only=True,
                # mode='min',
                # save_weights_only=True,
                # save_freq=1000,
            ),

            self._get_logger_callback(),
            # tf.keras.callbacks.LearningRateScheduler(self.learning_rate_scheduler)
        ]

        history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=self.config['epochs'],
            callbacks=callbacks,
            initial_epoch=self.initial_epoch
        )

        return history