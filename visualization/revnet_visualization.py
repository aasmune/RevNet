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
        save_path = os.path.join(os.path.dirname(__file__), "..", "saved_models", "model_weights_" + self.config.model.architecture + ".h5")

        if os.path.exists(save_path):
            self.model.load_weights(save_path)

    def create_generator(self):
        X_test = self.data[0][0]
        Y_test = self.data[1][0]
        
        self.generator = TimeseriesGenerator(
            X_test, 
            Y_test, 
            length=self.config.model.recursive_depth,
            batch_size=self.config.trainer.batch_size)

    def visualize(self):

        print("Starting to visualize")

        X = self.data[0][0]
        Y = self.data[1][0]

        predicted_output = self.model.predict_generator(
            self.generator
        ) 

        # Visualize
        output_path = os.path.join(os.path.dirname(__file__), "..", "output")

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for i in range(self.config.data.number_of_cells):

            fig = plt.figure(figsize=(12, 8))
            plt.title(f"Accuracy training, cell {i}")
            plt.xlabel("Number of training steps")
            plt.plot(Y[:,i], label="Measured", linewidth=0.7)
            plt.plot(np.append(np.zeros(self.config.model.recursive_depth), predicted_output[:,i]), label="Predicted", linewidth=0.7)
            plt.ylabel("Voltage")
            plt.grid()
            plt.legend(loc="best")
            plt.tick_params(axis='y')
            plt.tight_layout()

            plt.savefig(os.path.join(output_path, f"cell_{i}.pdf"))

    # def plot_loss(self):
    #     # return 

    # def plot_accuracy(self):
    #     raise NotImplementedError
        