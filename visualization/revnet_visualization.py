import os
import numpy as np
from base.base_visualization import BaseVisualization
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.datasets import mnist
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt   

class RevNetVisualization(BaseVisualization):
    
    
    def __init__(self, model, data, config):
        super(RevNetVisualization, self).__init__(model, data, config)

        self.load_weights()
        self.create_generator()

    
    def load_weights(self):
        print("Start load model")
        save_path = os.path.join(os.path.dirname(__file__), "..", "model_weights_" + self.config.model.architecture + "_" + str(self.config.trainer.num_epochs) + "_epochs_" + str(self.config.trainer.name.split(".")[-1]) + ".h5")
        

        if os.path.exists(save_path):
            self.model.load_weights(save_path)

        print("Finished loading model")

    def create_generator(self):
        X_test = self.data[0]
        Y_test = self.data[1]
        
        self.generator = TimeseriesGenerator(
            X_test, 
            Y_test, 
            length=self.config.model.recursive_depth,
            batch_size=self.config.trainer.batch_size)

    def visualize(self, data_loader):

        print("Starting to visualize")

        X = self.data[0]
        Y = self.data[1]

        predicted_output = self.model.predict_generator(
            self.generator
        ) 

        print("Prediction finished")
        # Visualize
        output_path = os.path.join(os.path.dirname(__file__), "..", "output")

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for i in range(self.config.data.number_of_cells):
            fig = plt.figure(figsize=(12, 8))
            plt.title(f"Accuracy training, cell {i}")
            plt.xlabel("Number of training steps")
            plt.plot(Y[:,i], label="Measured", linewidth=0.7)
            plt.plot(predicted_output[:,i], label="Predicted", linewidth=0.7)
            plt.ylabel("Voltage")
            plt.grid()
            plt.legend(loc="best")
            plt.tick_params(axis='y')
            plt.tight_layout()
            np.savetxt(os.path.join(output_path, self.config.model.architecture + "_" + str(self.config.trainer.num_epochs) + "_epochs_" + self.config.trainer.name.split(".")[-1] + f"_measured_{i}.csv"), Y[:,i], delimiter=",")
            np.savetxt(os.path.join(output_path, self.config.model.architecture + "_" + str(self.config.trainer.num_epochs) + "_epochs_" + self.config.trainer.name.split(".")[-1] + f"_predicted_{i}.csv"), predicted_output[:,i], delimiter=",")

            plt.savefig(os.path.join(output_path, self.config.model.architecture + "_" + str(self.config.trainer.num_epochs) + "_epochs_" + self.config.trainer.name.split(".")[-1] + f"_cell_{i}.pdf"))
        plt.show()

    # def plot_loss(self):
    #     # return 

    # def plot_accuracy(self):
    #     raise NotImplementedError


class RevNetVisualization2(BaseVisualization):
    
    
    def __init__(self, model, data, config):
        super(RevNetVisualization2, self).__init__(model, data, config)

        self.load_weights()
        self.create_generator()

    
    def load_weights(self):
        print("Start load model")
        save_path = os.path.join(os.path.dirname(__file__), "..", "model_weights_" + self.config.model.architecture + "_" + str(self.config.trainer.num_epochs) + "_epochs_" + str(self.config.trainer.name.split(".")[-1]) + ".h5")
        

        if os.path.exists(save_path):
            self.model.load_weights(save_path)

        print("Finished loading model")

    def create_generator(self):
        X_test = self.data[0][0]
        Y_test = self.data[1][0]
        
        self.generator = TimeseriesGenerator(
            np.array(X_test), 
            np.array(Y_test), 
            length=self.config.model.recursive_depth,
            batch_size=self.config.trainer.batch_size)

    def visualize(self, data_loader):

        print("Starting to visualize")

        X = np.array(self.data[0][0])
        Y = np.array(self.data[1][0])

        predicted_output = self.model.predict_generator(
            self.generator
        ) 

        print("Prediction finished")
        # Visualize
        output_path = os.path.join(os.path.dirname(__file__), "..", "output")

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for i in range(self.config.data.number_of_cells):
            fig = plt.figure(figsize=(12, 8))
            plt.title(f"Accuracy training, cell {i}")
            plt.xlabel("Number of training steps")
            plt.plot(Y[:,i], label="Measured", linewidth=0.7)
            plt.plot(predicted_output[:,i], label="Predicted", linewidth=0.7)
            plt.ylabel("Voltage")
            plt.grid()
            plt.legend(loc="best")
            plt.tick_params(axis='y')
            plt.tight_layout()
            np.savetxt(os.path.join(output_path, self.config.model.architecture + "_" + str(self.config.trainer.num_epochs) + "_epochs_" + self.config.trainer.name.split(".")[-1] + f"_measured_{i}.csv"), Y[:,i], delimiter=",")
            np.savetxt(os.path.join(output_path, self.config.model.architecture + "_" + str(self.config.trainer.num_epochs) + "_epochs_" + self.config.trainer.name.split(".")[-1] + f"_predicted_{i}.csv"), predicted_output[:,i], delimiter=",")

            plt.savefig(os.path.join(output_path, self.config.model.architecture + "_" + str(self.config.trainer.num_epochs) + "_epochs_" + self.config.trainer.name.split(".")[-1] + f"_cell_{i}.pdf"))
        plt.show()

    # def plot_loss(self):
    #     # return 

    # def plot_accuracy(self):
    #     raise NotImplementedError