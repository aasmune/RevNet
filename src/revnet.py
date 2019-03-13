import os
import pickle

import analyze_csv as csv_import
import visualization as visualize
import matplotlib.pyplot as plt                             #for plotting   
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Input, Dropout, Flatten
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
cwd = os.path.dirname(__file__)

NUM_CELLS = 1
input_indices=range(0,37 + 2 * NUM_CELLS)
output_indices=range(37 + NUM_CELLS, 37 + 2 * NUM_CELLS)
recursive_depth=(2)


def import_log(folder):
    TEMPERATURE_CHANNEL_TEMPLATE = "BMS_Cell_Temperature_"
    VOLTAGE_CHANNEL_TEMPLATE = "BMS_Cell_Voltage_"

    temperature_channels = [TEMPERATURE_CHANNEL_TEMPLATE + str(i) for i in range(NUM_CELLS)]
    voltage_channels = [VOLTAGE_CHANNEL_TEMPLATE + str(i) for i in range(NUM_CELLS)]
    channels = [
        "AMK_FL_Setpoint_negative_torque_limit", 
        "AMK_FR_Setpoint_negative_torque_limit",
        "AMK_RL_Setpoint_negative_torque_limit",
        "AMK_RR_Setpoint_negative_torque_limit",
        "AMK_FL_Setpoint_positive_torque_limit", 
        "AMK_FR_Setpoint_positive_torque_limit",
        "AMK_RL_Setpoint_positive_torque_limit",
        "AMK_RR_Setpoint_positive_torque_limit",
        "AMK_FL_Actual_velocity", 
        "AMK_FR_Actual_velocity",
        "AMK_RL_Actual_velocity",
        "AMK_RR_Actual_velocity",
        "AMK_FL_Torque_current",
        "AMK_FR_Torque_current",
        "AMK_RL_Torque_current",
        "AMK_RR_Torque_current",
        "AMK_FL_Temp_IGBT",                             #inverter temp
        "AMK_FR_Temp_IGBT",
        "AMK_RL_Temp_IGBT",
        "AMK_RR_Temp_IGBT",
        "BMS_Tractive_System_Current_Transient",
        "BMS_SOC_from_lut",
        "INS_Vx",                                       #long vel
        "INS_Vy",                                       #lat vel
        "INS_Ax",                                       #long acc
        "INS_Ay",                                       #lat acc
        "INS_Yaw_rate",
        "SBS_F1_APPS1_Sensor",                          #Acceleration pedal position sensor
        "SBS_F1_APPS2_Sensor",
        "SBS_F1_brakePressure1_Sensor",
        "SBS_F1_brakePressure2_Sensor", 
        "SBS_F2_Damper_pos_FL",
        "SBS_F2_Damper_pos_FR",
        "SBS_R1_Damper_pos_RL",
        "SBS_R1_Damper_pos_RR",
        "SBS_F1_KERS_Sensor"
        
        ]
    channels.extend(temperature_channels)
    channels.extend(voltage_channels)

    filenames = [os.path.join(folder, channel) + ".csv" for channel in channels]
        
    raw_data = csv_import.read_csv_files(filenames)

    data = csv_import.create_single_table(raw_data)

    return data


def main():
    
    # Import endurance FSS
    folder = os.path.join(cwd, "data", "FSS_endurance")
    data = import_log(folder)

    X = data[:,input_indices]
    Y = data[:,output_indices]

    X_train = X[69000:107000]
    Y_train = Y[69000:107000]

    X_test = X[132000:180000]
    Y_test = Y[132000:180000]

    

    # Create the model for the network
    model = Sequential([        
        LSTM(100, input_shape=(recursive_depth, len(input_indices)), return_sequences=True),
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
    data_gen_train = TimeseriesGenerator(X_train, Y_train,
                                length=recursive_depth,
                                batch_size=batch_size)

    data_gen_test = TimeseriesGenerator(X_test, Y_test,
                                length=recursive_depth,
                                batch_size=batch_size)                            


    model.fit_generator(data_gen_train, epochs=7)

    y_train = model.predict_generator(data_gen_train) 

    visualize.netAccuracy(true_output=Y_train, model_output=y_train,
    n_cells=NUM_CELLS, recursive_depth=recursive_depth)


if __name__ == "__main__":
    main()



    