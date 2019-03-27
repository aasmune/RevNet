import os
import pickle
import pathlib

import analyze_csv as csv_import
# import visualization as visualize
import matplotlib.pyplot as plt                             #for plotting
from matplotlib import animation
from matplotlib import colors
from matplotlib import ticker 
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM, Input, Dropout, Flatten
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import seaborn as sns
import datetime as dt

from sklearn.preprocessing import normalize
cwd = os.path.dirname(__file__)


class VoltObj:
    """
    This can be changed to take in predicted values from the network for example.
    """
    def __init__(self, times, voltage_prediction):
        self.voltages = voltage_prediction
        self.time = times
        self.val = 0
        self.i = 0
    
    def get_next(self):
        time = self.time[self.i]
        cell_voltages = np.reshape(self.voltages[self.i, :], ((2, 10, 7)))
        self.i += 20
        return time, cell_voltages
    
    def get_next_test(self):
        self.i += 1
        self.val = np.sin(np.pi*self.i/15)
        self.voltages[1, :, :] = self.val
        return self.voltages

channels = [
    # File name, scaling factor
    ["AMK_FL_Setpoint_negative_torque_limit", 21], #0
    ["AMK_FR_Setpoint_negative_torque_limit", 21], #1
    ["AMK_RL_Setpoint_negative_torque_limit", 21], #2
    ["AMK_RR_Setpoint_negative_torque_limit", 21], #3
    ["AMK_FL_Setpoint_positive_torque_limit", 21], #4
    ["AMK_FR_Setpoint_positive_torque_limit", 21], #5
    ["AMK_RL_Setpoint_positive_torque_limit", 21], #6
    ["AMK_RR_Setpoint_positive_torque_limit", 21], #7
    ["AMK_FL_Actual_velocity", 20000], #8
    ["AMK_FR_Actual_velocity", 20000], #9
    ["AMK_RL_Actual_velocity", 20000], #10
    ["AMK_RR_Actual_velocity", 20000], #11
    ["AMK_FL_Torque_current", 50000], #12
    ["AMK_FR_Torque_current", 50000], #13
    ["AMK_RL_Torque_current", 50000], #14
    ["AMK_RR_Torque_current", 50000], #15
    ["AMK_FL_Temp_IGBT", 80], #16      #inverter temp
    ["AMK_FR_Temp_IGBT", 80], #17
    ["AMK_RL_Temp_IGBT", 80], #18
    ["AMK_RR_Temp_IGBT", 80], #19
    ["BMS_Tractive_System_Current_Transient", 140], #20
    # ["BMS_SOC_from_lut", 96], #21
    ["INS_Vx", 30], #22      #long vel
    ["INS_Vy", 10], #23      #lat vel
    ["INS_Ax", 15], #24      #long acc
    ["INS_Ay", 25], #25      #lat acc
    ["INS_Yaw_rate", 3], #26
    ["SBS_F1_APPS1_Sensor", 105], #27      #Acceleration pedal position sensor
    ["SBS_F1_APPS2_Sensor", 105], #28
    ["SBS_F1_brakePressure1_Sensor", 40], #29
    ["SBS_F1_brakePressure2_Sensor", 40], #30
    ["SBS_F2_Damper_pos_FL", 40], #31
    ["SBS_F2_Damper_pos_FR", 40], #32
    ["SBS_R1_Damper_pos_RL", 40], #33
    ["SBS_R1_Damper_pos_RR", 40], #34
    ["SBS_F1_KERS_Sensor", 170]#35
]

def import_log(folder):
    
    filenames = [os.path.join(folder, channel[0]) + ".csv" for channel in channels]
        
    raw_data = csv_import.read_csv_files(filenames)

    data, time = csv_import.create_single_table(raw_data)

    return data, time


def netAccuracy(save_plot=True, show_plot=False,*, true_x, true_output, model_x, model_output, n_cells, recursive_depth):
    """
    Plots true output vs neural net model output.
    """

    sns.set(style="darkgrid")  # makes plots pretty
    output_path = pathlib.Path(os.path.dirname(__file__), "output")

    if not output_path.exists():
        os.makedirs(output_path)
    for i in range(2):

        fig = plt.figure(figsize=(12, 8))
        plt.title(f"Predicted output vs. measured for cell{i}")
        plt.xlabel("Timestep")
        plt.plot(true_x, true_output[:,i], label="Measured", linewidth=0.7)
        plt.plot(model_x, model_output[:,i], label="Predicted", linewidth=0.7)
        plt.ylabel("Voltage")

        plt.axhline(y=3, linewidth=0.5, color="k", linestyle='--', label="Min. safe voltage")
        plt.axhline(y=4.2, linewidth=0.5, color="k", linestyle='--', label="Max. safe voltage")

        plt.legend(loc="best")
        plt.tick_params(axis='y')
        plt.tight_layout()

        if save_plot:
            plt.savefig(output_path.joinpath(f"cell_{i}.pdf"))
        if show_plot:
            plt.show()

def make_animation(*, time, voltage_prediction):
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

    #predicted_voltages = np.ones((2, 10, 7))
    voltObj = VoltObj(time, voltage_prediction)

    # Set up plot to call animate() function periodically
    ani = animation.FuncAnimation(fig, animate,
          fargs=(voltObj, ax1, ax2, ax3, ax4, ax5, ax6, ax7, fig), interval=50)
    # ani.save('voltage_change.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()

def animate(i, voltObj, ax1, ax2, ax3, ax4, ax5, ax6, ax7, fig):
    # This function is called periodically from FuncAnimation
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

    time, volt = voltObj.get_next()
    min_safe = 0.875  # 3.0
    max_safe = 0.975  # 4.2

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

    # Format plot
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

    # title = 'Voltage over time ' + str(dt.datetime.now().strftime('%H:%M:%S')) #for real-time
    title = 'Voltage at time, t = ' + str(time)
    fig.suptitle(title)

def plot_cell_voltages(save_plot=True, show_plot=False, index=0, title="cell_volt_start",
                       *, time, voltage_prediction):
    """
    Plots cell voltages at certain time, time[index].
    """
    sns.set(style="darkgrid")  # makes plots pretty
    
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(4, 2, 1)
    ax2 = fig.add_subplot(4, 2, 2)
    ax3 = fig.add_subplot(4, 2, 3)
    ax4 = fig.add_subplot(4, 2, 4)
    ax5 = fig.add_subplot(4, 2, 5)
    ax6 = fig.add_subplot(4, 2, 6)
    ax7 = fig.add_subplot(4, 2, 7)
    
    voltObj = VoltObj(time, voltage_prediction)
    voltObj.i = index
    # create heatmap of cell voltages, code from animation reused
    animate(0, voltObj, ax1, ax2, ax3, ax4, ax5, ax6, ax7, fig)

    if save_plot:
        # create output path
        output_path = pathlib.Path(os.path.dirname(__file__), "output")
        if not output_path.exists():
            os.makedirs(output_path)
        # save figure
        plt.savefig(output_path.joinpath(title))

    if show_plot:
        plt.show()


if __name__ == "__main__":
    """
    test on fsg-data
    """
    
    folder_fsg = os.path.join(cwd, "data", "FSG_endurance")
    data,time = import_log(folder_fsg)
    time = time[20000:30000]

    NUM_CELLS = 140
    recursive_depth=(10)
    pred_time = time[recursive_depth-1:-1:1]

    Y_fsg = np.loadtxt("real_fsg.csv")
    y_fsg = np.loadtxt("pred_fsg.csv")
 
    # Visualize
    #netAccuracy(true_x=time, true_output=Y_fsg,model_x=pred_time,  model_output=y_fsg,
    #n_cells=NUM_CELLS, recursive_depth=recursive_depth, save_plot=False, show_plot=True)
    
    # make_animation(time=pred_time, voltage_prediction=y_fsg)
    plot_cell_voltages(time=pred_time, voltage_prediction=y_fsg)