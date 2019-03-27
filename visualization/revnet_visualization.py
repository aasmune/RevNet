import os
import numpy as np
from base.base_visualization import BaseVisualization
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.datasets import mnist
from sklearn.preprocessing import normalize

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colors
from matplotlib import ticker 

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
        sns.set(style="darkgrid")  # makes plots pretty

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
            # plt.plot(np.append(np.zeros(self.config.model.recursive_depth), predicted_output[:,i]), label="Predicted", linewidth=0.7)
            plt.plot(predicted_output[:,i], label="Predicted", linewidth=0.7)

            plt.axhline(y=3.0, linewidth=0.5, color="k", linestyle='--', label="Min. safe voltage")
            plt.axhline(y=4.2, linewidth=0.5, color="k", linestyle='--', label="Max. safe voltage")

            plt.ylabel("Voltage")
            plt.grid()
            plt.legend(loc="best")
            plt.tick_params(axis='y')
            plt.tight_layout()

            plt.savefig(os.path.join(output_path, f"cell_{i}.pdf"))

    def make_animation(self):
        """
        Animates based on a voltage object, giving the next voltages through get_next()
        """
        # Create figure for plotting
        fig = plt.figure()

        ax1 = fig.add_subplot(4, 2, 1)
        ax2 = fig.add_subplot(4, 2, 2)
        ax3 = fig.add_subplot(4, 2, 3)
        ax4 = fig.add_subplot(4, 2, 4)
        ax5 = fig.add_subplot(4, 2, 5)
        ax6 = fig.add_subplot(4, 2, 6)
        ax7 = fig.add_subplot(4, 2, 7)

        # Get data to animate
        predicted_output = self.model.predict_generator(
            self.generator
        )

        # Set up plot to call animate() function periodically
        # Might have to change interval and add no. of frames we want in the animation.
        ani = animation.FuncAnimation(fig, self.animate,
            fargs=(predicted_output, ax1, ax2, ax3, ax4, ax5, ax6, ax7, fig), interval=50)

        # Can try to save the animation as an mp4-file
        # ani.save('voltage_change.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
        plt.show()

    def animate(self, i, predicted_data, ax1, ax2, ax3, ax4, ax5, ax6, ax7, fig):
        # This function is called periodically from FuncAnimation

        # get voltage data from specific time
        volt = np.reshape(predicted_data[i, :], ((2, 10, 7)))
        
        # Clear previous data from axes
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax5.clear()
        ax6.clear()
        ax7.clear()

        # Remove ticks on axes
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.set_xticklabels([])
        ax4.set_yticklabels([])
        ax4.set_xticklabels([])
        ax5.set_yticklabels([])
        ax5.set_xticklabels([])
        ax6.set_yticklabels([])
        ax6.set_xticklabels([])
        ax7.set_yticklabels([])
        ax7.set_xticklabels([])

        # Create 7 heatmaps, with colors scaled to min and max voltages
        min_safe = 3.0
        max_safe = 4.2

        heatmap1 = ax1.pcolor(volt[:, :, 0], cmap='inferno',
                            vmin=min_safe, vmax=max_safe, edgecolors='k', linewidths=0.7)
        heatmap2 = ax2.pcolor(volt[:, :, 1], cmap='inferno',
                            vmin=min_safe, vmax=max_safe, edgecolors='k', linewidths=0.7)
        heatmap3 = ax3.pcolor(volt[:, :, 2], cmap='inferno',
                            vmin=min_safe, vmax=max_safe, edgecolors='k', linewidths=0.7)
        heatmap4 = ax4.pcolor(volt[:, :, 3], cmap='inferno', 
                            vmin=min_safe, vmax=max_safe, edgecolors='k', linewidths=0.7)
        heatmap5 = ax5.pcolor(volt[:, :, 4], cmap='inferno',
                            vmin=min_safe, vmax=max_safe, edgecolors='k', linewidths=0.7)
        heatmap6 = ax6.pcolor(volt[:, :, 5], cmap='inferno',
                            vmin=min_safe, vmax=max_safe, edgecolors='k', linewidths=0.7)
        heatmap7 = ax7.pcolor(volt[:, :, 6], cmap='inferno',
                            vmin=min_safe, vmax=max_safe, edgecolors='k', linewidths=0.7)

        # Format text in the heatmaps to show voltage value for each cell
        data = volt[:, :, 0]
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                ax1.text(x + 0.5, y + 0.5, '%.3f' % data[y, x],
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=8)
        
        data = volt[:, :, 1]
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                ax2.text(x + 0.5, y + 0.5, '%.3f' % data[y, x],
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=8)

        data = volt[:, :, 2]
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                ax3.text(x + 0.5, y + 0.5, '%.3f' % data[y, x],
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=8)

        data = volt[:, :, 3]
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                ax4.text(x + 0.5, y + 0.5, '%.3f' % data[y, x],
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=8)
        
        data = volt[:, :, 4]
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                ax5.text(x + 0.5, y + 0.5, '%.3f' % data[y, x],
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=8)
        
        data = volt[:, :, 5]
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                ax6.text(x + 0.5, y + 0.5, '%.3f' % data[y, x],
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=8)

        data = volt[:, :, 6]
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                ax7.text(x + 0.5, y + 0.5, '%.3f' % data[y, x],
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=8)

    def plot_cell_volt_grid(self, save_plot=True, show_plot=False, index=0, title="cell_volt_start"):
        """
        Plots cell voltages at certain time-index
        """
        sns.set(style="darkgrid")  # makes plots pretty
        
        # Create figure and axes for the plot
        fig = plt.figure(figsize=(12, 8))

        ax1 = fig.add_subplot(4, 2, 1)
        ax2 = fig.add_subplot(4, 2, 2)
        ax3 = fig.add_subplot(4, 2, 3)
        ax4 = fig.add_subplot(4, 2, 4)
        ax5 = fig.add_subplot(4, 2, 5)
        ax6 = fig.add_subplot(4, 2, 6)
        ax7 = fig.add_subplot(4, 2, 7)

        # Get data
        predicted_output = self.model.predict_generator(
            self.generator
        )

        # Create heatmap of cell voltages, code from animation reused
        self.animate(0, predicted_output, ax1, ax2, ax3, ax4, ax5, ax6, ax7, fig)

        if save_plot:
            # create output path
            output_path = os.path.join(os.path.dirname(__file__), "..", "output")
            if not output_path.exists():
                os.makedirs(output_path)
            # save figure
            plt.savefig(output_path.joinpath(title))

        if show_plot:
            plt.show()

    # def plot_loss(self):
    #     # return 

    # def plot_accuracy(self):
    #     raise NotImplementedError
   