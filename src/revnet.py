import os
import pickle
import pathlib

import analyze_csv as csv_import
import visualization as visualize
import matplotlib.pyplot as plt                             #for plotting   
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Input, Dropout, Flatten
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

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

NUM_CELLS = 140
NUM_NON_BATTERY_CHANNELS = len(channels)
input_indices=range(0,NUM_NON_BATTERY_CHANNELS + 2 * NUM_CELLS)
output_indices=range(NUM_NON_BATTERY_CHANNELS + NUM_CELLS, NUM_NON_BATTERY_CHANNELS + 2 * NUM_CELLS)
recursive_depth=(10)

TEMPERATURE_CHANNEL_TEMPLATE = "BMS_Cell_Temperature_"
VOLTAGE_CHANNEL_TEMPLATE = "BMS_Cell_Voltage_"

temperature_channels = [[TEMPERATURE_CHANNEL_TEMPLATE + str(i), 70] for i in range(NUM_CELLS)]
voltage_channels = [[VOLTAGE_CHANNEL_TEMPLATE + str(i), 4.5] for i in range(NUM_CELLS)]

channels.extend(temperature_channels)
channels.extend(voltage_channels)


def import_log(folder):
    
    filenames = [os.path.join(folder, channel[0]) + ".csv" for channel in channels]
        
    raw_data = csv_import.read_csv_files(filenames)

    data = csv_import.create_single_table(raw_data)

    return data

def import_fss():
    folder_fss = os.path.join(cwd, "data", "FSS_endurance")
    data_fss = import_log(folder_fss)

    start_time_fss = 80000
    driver_change_start_fss = 142000
    driver_change_finish_fss = 174000
    finish_time_fss = 234000

    X_FSS = data_fss[:,input_indices]
    Y_FSS = data_fss[:,output_indices]


    X_fss = X_FSS[start_time_fss:finish_time_fss]
    Y_fss = Y_FSS[start_time_fss:finish_time_fss]

    X_fss=normalize(X_fss, norm='max', axis=0)
    Y_fss=normalize(Y_fss, norm='max', axis=0)

    return X_fss, Y_fss

def import_fsg():
    folder_fsg = os.path.join(cwd, "data", "FSG_endurance")
    data_fsg = import_log(folder_fsg)

#    start_time_fsg = 80000
#   driver_change_start_fsg = 142000
#    driver_change_finish_fsg = 174000
#    finish_time_fsg = 234000

    X_FSG = data_fsg[:,input_indices]
    Y_FSG = data_fsg[:,output_indices]


#    X_fsg = X_FSG[start_time_fsg:finish_time_fsg]
#    Y_fsg = Y_FSG[start_time_fsg:finish_time_fsg]

    X_fsg=normalize(X_FSG, norm='max', axis=0)
    Y_fsg=normalize(Y_FSG, norm='max', axis=0)

    return X_fsg, Y_fsg

def import_nr3():
    folder_nr3 = os.path.join(cwd, "data", "testing_endurance_3")
    data_nr3 = import_log(folder_nr3)

    X_nr3 = data_nr3[:,input_indices]
    Y_nr3 = data_nr3[:,output_indices]

    X_nr3=normalize(X_nr3, norm='max', axis=0)
    Y_nr3=normalize(Y_nr3, norm='max', axis=0)

    return X_nr3, Y_nr3

def main():
    #----------------------------------------------------------------------------
    # Import endurance FSS
    #----------------------------------------------------------------------------

    X_fss, Y_fss = import_fss()
    X_nr3, Y_nr3 = import_nr3()
    X_fsg, Y_fsg = import_fsg()

    
    # Create the model for the network
    model = Sequential([        
        LSTM(1000, input_shape=(recursive_depth, len(input_indices)), return_sequences=True),
        Dropout(0.2),

        LSTM(100, return_sequences=True),
        LSTM(100, return_sequences=True),
        Dropout(0.2),

        LSTM(100, return_sequences=True),
        LSTM(100, return_sequences=True),
        Dropout(0.2),

        Flatten(),

        Dense(NUM_CELLS, activation="linear")
    ])

    #Compile model
    model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

    model.summary()

    
    batch_size = 200
    data_gen_fss = TimeseriesGenerator(X_fss, Y_fss,
                                length=recursive_depth,
                                batch_size=batch_size)

    data_gen_nr3 = TimeseriesGenerator(X_nr3, Y_nr3,
                                length=recursive_depth,
                                batch_size=batch_size) 

    data_gen_fsg = TimeseriesGenerator(X_fsg, Y_fsg,
                                length=recursive_depth,
                                batch_size=batch_size)  

    model.fit_generator(
        data_gen_fss, 
        epochs=50)

    model.fit_generator(
        data_gen_nr3,
        epochs=50
    )

    model.save("saved_model.h5")


    y_fsg = model.predict_generator(data_gen_fsg) 

    


    # Visualize
    output_path = pathlib.Path(os.path.dirname(__file__), "output")

    if not output_path.exists():
        os.makedirs(output_path)
    
    visualize.netAccuracy(true_output=Y_train, model_output=y_train,
    n_cells=NUM_CELLS, recursive_depth=recursive_depth)
        
        

    
    


if __name__ == "__main__":
    main()



    
