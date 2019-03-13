import matplotlib.pyplot as plt
import numpy as np

def netAccuracy(*, true_output, model_output, n_cells, recursive_depth):
    """
    Plots true output vs neural net model output.
    """
    
    for i in range(n_cells):

        fig = plt.figure(figsize=(12, 8))
        plt.title("Accuracy training")
        plt.xlabel("Number of training steps")
        plt.plot(true_output[:,i], label="Measured", linewidth=0.7)
        plt.plot(np.append(np.zeros(recursive_depth), model_output[:,i]), label="Predicted", linewidth=0.7)
        plt.ylabel("Voltage")
        plt.grid()
        plt.legend(loc="best")
        plt.tick_params(axis='y')
        plt.tight_layout()

        plt.show()