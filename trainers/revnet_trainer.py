from base.base_trainer import BaseTrain
import os
import numpy as np
from itertools import accumulate
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LambdaCallback

def custom_generator(generators):
    generator_lengths = [len(generator) for generator in generators]
    # current_index = [0 for _ in range(len(generators))]
    while True:
        for i in range(len(generators)):
            gen = generators[i]
            length = generator_lengths[i]
            
            for j in range(length):
                yield gen[j]
                
                



class RevNetTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(RevNetTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.generators = None
        self.end_of_each_generator_step = []
        self.size_of_each_generator = []
        self.create_generator()
        self.init_callbacks()
        

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
        self.size_of_each_generator = [len(generator) for generator in self.generators]
        end_of_each_generator_step = list(accumulate(self.size_of_each_generator))
        self.end_of_each_generator_step = end_of_each_generator_step
        # self.callbacks.append(
        #     LambdaCallback(on_batch_end=lambda idx, _: self.model.reset_states() if idx in end_of_each_generator_step else None)
        # )


    def create_generator(self):
        X_trains = self.data[0]
        Y_trains = self.data[1]
        
        generators = TimeseriesGenerator(
            X_trains, 
            Y_trains,
            length=self.config.model.recursive_depth,
            batch_size=self.config.trainer.batch_size)

        self.generators = generators

    def train(self):

        # Hyper parameters
        print("\nHyper parameters:")
        print(f"Architecture: {self.config.model.architecture}")
        print(f"Epochs: {self.config.trainer.num_epochs}")
        print(f"Batch size: {self.config.trainer.batch_size}")
        print(f"Training sets: {list(key for key in self.config.runs.keys() if not self.config.runs[key].use_for_testing)}\n")
        print(f"Len generators {len(self.generators)}")
        # print(f"Size of each generator: {self.size_of_each_generator}")
        # print(f"End of generator steps: {self.end_of_each_generator_step}")
        
        history = self.model.fit_generator(
            self.generators,
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            callbacks=self.callbacks
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        # self.val_loss.extend(history.history['val_loss'])
        # self.val_acc.extend(history.history['val_acc'])

        # Save weights
        save_path = os.path.join(os.path.dirname(__file__), "..", "saved_models")
        if not os.path.exists(save_path):
            print("Creating directory for save trained model.")
            os.makedirs(save_path)

        print("Save trained model.")
        self.model.save_weights(os.path.join(save_path, "model_weights_" + self.config.model.architecture + "_" + str(self.config.trainer.num_epochs) + "_epochs_" + self.config.trainer.name.split(".")[-1] + ".h5"))

        print("Save history from during training")
        np.savetxt(os.path.join(save_path, "loss_history_" + self.config.model.architecture + ".txt"), self.loss, delimiter=",")
        np.savetxt(os.path.join(save_path, "accuracy_history_" + self.config.model.architecture + ".txt"), self.loss, delimiter=",")
        # numpy.savetxt("loss_history.txt", self.loss, delimiter=",")
        # numpy.savetxt("loss_history.txt", self.loss, delimiter=",")



class RevNetTrainer2(BaseTrain):
    def __init__(self, model, data, config):
        super(RevNetTrainer2, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.generators = []
        self.end_of_each_generator_step = []
        self.size_of_each_generator = []
        self.create_generator()
        self.init_callbacks()
        

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
        self.size_of_each_generator = [len(generator) for generator in self.generators]
        end_of_each_generator_step = list(accumulate(self.size_of_each_generator))
        self.end_of_each_generator_step = end_of_each_generator_step
        # self.callbacks.append(
        #     LambdaCallback(on_batch_end=lambda idx, _: self.model.reset_states() if idx in end_of_each_generator_step else None)
        # )


    def create_generator(self):
        X_trains = self.data[0]
        Y_trains = self.data[1]

        for X, Y in zip(X_trains, Y_trains):
            self.generators.append(TimeseriesGenerator(
            np.array(X), 
            np.array(Y),
            length=self.config.model.recursive_depth,
            batch_size=self.config.trainer.batch_size))


    def train(self):

        # Hyper parameters
        print("\nHyper parameters:")
        print(f"Architecture: {self.config.model.architecture}")
        print(f"Epochs: {self.config.trainer.num_epochs}")
        print(f"Batch size: {self.config.trainer.batch_size}")
        print(f"Training sets: {list(key for key in self.config.runs.keys() if not self.config.runs[key].use_for_testing)}\n")
        print(f"Len generators {len(self.generators)}")
        # print(f"Size of each generator: {self.size_of_each_generator}")
        # print(f"End of generator steps: {self.end_of_each_generator_step}")
        for generator in self.generators:
            history = self.model.fit_generator(
                generator,
                epochs=self.config.trainer.num_epochs,
                verbose=self.config.trainer.verbose_training,
                callbacks=self.callbacks
            )
            self.loss.extend(history.history['loss'])
            self.acc.extend(history.history['acc'])
            # self.val_loss.extend(history.history['val_loss'])
            # self.val_acc.extend(history.history['val_acc'])

        # Save weights
        save_path = os.path.join(os.path.dirname(__file__), "..", "saved_models")
        if not os.path.exists(save_path):
            print("Creating directory for save trained model.")
            os.makedirs(save_path)

        print("Save trained model.")
        self.model.save_weights(os.path.join(save_path, "model_weights_" + self.config.model.architecture + "_" + str(self.config.trainer.num_epochs) + "_epochs_" + self.config.trainer.name.split(".")[-1] + ".h5"))

        print("Save history from during training")
        np.savetxt(os.path.join(save_path, "loss_history_" + self.config.model.architecture + ".txt"), self.loss, delimiter=",")
        np.savetxt(os.path.join(save_path, "accuracy_history_" + self.config.model.architecture + ".txt"), self.loss, delimiter=",")
        # numpy.savetxt("loss_history.txt", self.loss, delimiter=",")
        # numpy.savetxt("loss_history.txt", self.loss, delimiter=",")