import matplotlib.pyplot as plt
import numpy as np

import os
import pickle
import pathlib

import analyze_csv as csv_import
import visualization as visualize
import matplotlib.pyplot as plt                             #for plotting
from matplotlib import animation
from matplotlib import colors
from matplotlib import ticker 
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM, Input, Dropout, Flatten
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import seaborn as sns

from sklearn.preprocessing import normalize
cwd = os.path.dirname(__file__)

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
        plt.title(f"Predicted output vs measured, cell{i}")
        plt.xlabel("Timestep")
        plt.plot(true_x, true_output[:,i], label="Measured", linewidth=0.7)
        plt.plot(model_x, model_output[:,i], label="Predicted", linewidth=0.7)
        plt.ylabel("Voltage")

        plt.axhline(y=1, linewidth=0.5, color="r", linestyle='--', label="max allowed voltage")
        plt.axhline(y=0.87, linewidth=0.5, color="r", linestyle='--', label="min allowed voltage")

        plt.legend(loc="best")
        plt.tick_params(axis='y')
        plt.tight_layout()

        if save_plot:
            plt.savefig(output_path.joinpath(f"cell_{i}.pdf"))
        if show_plot:
            plt.show()
        

def animate_cell_voltage(*, model_x, model_output, n_cells, recursive_depth):
    sns.set()
    # grid = np.zeros(n_cells/2, n_cells/2)
    # grid_matrix = np.zeros(())
    for i in range(len(model_output[:, 1])):
        a = 1
    cell1 = np.append(np.zeros(recursive_depth), model_output[:,1])
    print(len(cell1))
    print(len(model_output[:, 1]))
    print(recursive_depth)

    #fig = plt.figure()
    #f, ax = plt.subplots(figsize=(9, 6))
    #sns.heatmap(cell1, annot=True, fmt="d", linewidths=.5, ax=ax)
    return


def init():
    for i in range(7):
        plt.subplot(4, 2, i+1)
        plt.title(str(i+1))
        sns.heatmap(data)

def animate(i):
    data = np.ones((2, 10))*i
    for j in range(7):
        plt.subplot(4, 2, j+1)
        plt.title(str(j+1))
        sns.heatmap(data)

def test():
    """
    HERE, SOMETHING'S WRONG
    """
    sns.set()
    data = np.zeros((2, 10))
    fig = plt.figure(1)

    """
    for i in range(7):
        plt.subplot(4, 2, i+1)
        plt.title(str(i+1))
        sns.heatmap(data)
    plt.show()
    """

    anim = animation.FuncAnimation(fig, animate, np.arange(1, 100), init_func=init, interval=25, repeat = False)
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
    netAccuracy(true_x=time, true_output=Y_fsg,model_x=pred_time,  model_output=y_fsg,
    n_cells=NUM_CELLS, recursive_depth=recursive_depth, save_plot=False, show_plot=True)
    # test()