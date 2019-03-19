from base.base_trainer import BaseTrain
import os
from itertools import accumulate
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LambdaCallback

def custom_generator(generators):
    generator_lengths = [len(generator) for generator in generators]
    current_index = [0 for _ in range(len(generators))]
    while True:
        for i in range(len(generators)):
            gen = generators[i]
            length = generator_lengths[i]
            index = current_index[i]
            if index < length:
                yield gen[i]



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
            LambdaCallback(on_batch_end=lambda idx, _: self.model.reset_state() if idx in end_of_each_generator_step else None)
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

        # Hyper parameters
        print("\nHyper parameters:")
        print(f"Architecture: {self.config.model.architecture}")
        print(f"Epochs: {self.config.trainer.num_epochs}")
        print(f"Batch size: {self.config.trainer.batch_size}")
        print(f"Training sets: {list(self.config.runs.keys())}\n")
        history = self.model.fit_generator(
            custom_generator(self.generators),
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            callbacks=self.callbacks,
            steps_per_epoch=sum(x.data.shape[0] for x in self.generators)/self.config.trainer.batch_size
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])
