import os
import pickle

import analyze_csv as csv_import
import matplotlib.pyplot as plt                             #for plotting   
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Input, Dropout, Flatten
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
cwd = os.path.dirname(__file__)

NUM_CELLS = 2
input_indices=range(0,36 + 2 * NUM_CELLS)
output_indices=range(36 + NUM_CELLS, 36 + 2 * NUM_CELLS)
recursive_depth=(2)


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
    ["BMS_SOC_from_lut", 96], #21
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
    TEMPERATURE_CHANNEL_TEMPLATE = "BMS_Cell_Temperature_"
    VOLTAGE_CHANNEL_TEMPLATE = "BMS_Cell_Voltage_"

    temperature_channels = [[TEMPERATURE_CHANNEL_TEMPLATE + str(i), 70] for i in range(NUM_CELLS)]
    voltage_channels = [[VOLTAGE_CHANNEL_TEMPLATE + str(i), 4.5] for i in range(NUM_CELLS)]
    
    channels.extend(temperature_channels)
    channels.extend(voltage_channels)

    filenames = [os.path.join(folder, channel[0]) + ".csv" for channel in channels]
        
    raw_data = csv_import.read_csv_files(filenames)

    data = csv_import.create_single_table(raw_data)

    return data


def main():
    #----------------------------------------------------------------------------
    # Import endurance FSS
    #----------------------------------------------------------------------------
    folder = os.path.join(cwd, "data", "FSS_endurance")
    data = import_log(folder)

    X = data[:,input_indices]
    Y = data[:,output_indices]

    X_train = X[69000:107000]
    Y_train = Y[69000:107000]

    X_test = X[132000:180000]
    Y_test = Y[132000:180000]
    #----------------------------------------------------------------------------
    #-----------------nomrmalization of input - data(make prittier)---------------------
    #----------------------------------------------------------------------------
    maxVoltage=4.5
    ScalingVector = np.array([channel[1] for channel in channels]) #fetches the scaling vectors
    X_test=X_test/ScalingVector
    X_train=X_train/ScalingVector
    Y_train=Y_train/maxVoltage
    Y_test=Y_test/maxVoltage
    
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



    for i in range(NUM_CELLS):

        fig = plt.figure(figsize=(12, 8))
        plt.title("Accuracy training")
        plt.xlabel("Number of training steps")
        plt.plot(Y_train[:,i], label="Measured", linewidth=0.7)
        plt.plot(np.append(np.zeros(recursive_depth), y_train[:,i]), label="Predicted", linewidth=0.7)
        plt.ylabel("Voltage")
        plt.grid()
        plt.legend(loc="best")
        plt.tick_params(axis='y')
        plt.tight_layout()

        plt.show()
        
        

    
    


if __name__ == "__main__":
    main()



    