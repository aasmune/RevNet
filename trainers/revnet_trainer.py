from base.base_trainer import BaseTrain
import os
from itertools import accumulate
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LambdaCallback


class RevNetTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(RevNetTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.generators = None
        self.create_generator()
        self.init_callbacks()
        self.end_of_each_generator_step = []

    def init_callbacks(self):

        # self.callbacks.append(
        #     EarlyStopping(
        #         monitor='val_loss',
        #         patience=self.config.callbacks.early_stopping_patience,
        #         restore_best_weights=self.config.callbacks.early_stopping_restore_best_weights
        #     )
        # )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

        # Create custom callback resetting LSTM state when each dataset (generator) is finished
        size_of_each_generator = [len(generator) for generator in self.generators]
        end_of_each_generator_step = accumulate(size_of_each_generator)
        self.callbacks.append(
            LambdaCallback(on_batch_end=lambda idx: self.model.reset_state() if idx in end_of_each_generator_step else None)
        )


    def create_generator(self):
        X_trains = self.data[0]
        Y_trains = self.data[1]
        
        generators = [TimeseriesGenerator(
            X, 
            Y, 
            length=self.config.model.recursive_depth,
            batch_size=self.config.trainer.batch_size) for (X, Y) in zip(X_trains, Y_trains)]

        self.generators = generators

    def train(self):
        history = self.model.fit_generator(
            self.generators,
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            callbacks=self.callbacks,
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])
