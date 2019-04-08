import numpy as np
import os
import matplotlib.pyplot as plt 

Y = np.genfromtxt(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "SmallMemory_10_epochs_RevNetTrainer_measured_0.csv")), delimiter=",")
predicted = np.genfromtxt(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "SmallMemory_10_epochs_RevNetTrainer_predicted_0.csv")), delimiter=",")

fig = plt.figure(figsize=(12, 8))
plt.title(f"Accuracy training, cell {0}")
plt.xlabel("Number of training steps")
plt.plot(Y, label="Measured", linewidth=0.7)
plt.plot(predicted, label="Predicted", linewidth=0.7)
plt.ylabel("Voltage")
plt.grid()
plt.legend(loc="best")
plt.tick_params(axis='y')
plt.tight_layout()

plt.show()