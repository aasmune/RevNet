import os
from base.base_data_loader import BaseDataLoader
from keras.datasets import mnist
from sklearn.preprocessing import normalize

from utils.import_csv import read_csv_files, create_single_table

class RevNetDataLoader(BaseDataLoader):
    DATA_SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    
    def __init__(self, config):
        super(RevNetDataLoader, self).__init__(config)

        # Number of cells used
        self.number_of_cells = self.config.data.number_of_cells

        self.channels, self.number_of_non_battery_channels = self._create_channels()

        self.X_train, self.Y_train = [], []
        self.X_test, self.Y_test =   [], []

        for run, run_config in self.config.runs.items():
            X, Y = self._import_run(run)

            if run_config.use_for_testing:
                self.X_test.append(X)
                self.Y_test.append(Y)
            else:
                self.X_train.append(X)
                self.Y_train.append(Y)

    def get_train_data(self):
        return self.X_train, self.Y_train

    def get_test_data(self):
        return self.X_test, self.Y_test

    def _create_channels(self):
        # Load all channels not related to temperature and voltage 
        # measurements for cells
        channels = self.config.data.other_channels
        number_of_non_battery_channels = len(channels)

        # Load temperature and voltage channels
        
        TEMPERATURE_CHANNEL_TEMPLATE = "BMS_Cell_Temperature_"
        VOLTAGE_CHANNEL_TEMPLATE = "BMS_Cell_Voltage_"

        temperature_channels = [TEMPERATURE_CHANNEL_TEMPLATE + str(i) for i in range(self.number_of_cells)]
        voltage_channels = [VOLTAGE_CHANNEL_TEMPLATE + str(i) for i in range(self.number_of_cells)]
        channels.extend(temperature_channels)
        channels.extend(voltage_channels)

        self.input_indices = range(len(channels))
        self.output_indices = range(number_of_non_battery_channels + self.number_of_cells, len(channels))
        return channels, number_of_non_battery_channels
    
    def _import_log_for_a_single_run(self, path):
        # Path to where files ccan be found
        folder_path = os.path.join(self.DATA_SRC_PATH, path)

        # All files to read
        filenames = [os.path.join(folder_path, channel) + ".csv" for channel in self.channels]

        # Import files
        raw_data = read_csv_files(filenames)

        # Process all files to be interpolated after the same time steps
        data = create_single_table(raw_data)

        return data

    def _import_run(self, run):
        # Load config for given run
        run_config = self.config.runs[run]

        # Load data
        data = self._import_log_for_a_single_run(run_config.local_path)

        # Divide into input and output
        X = data[:,self.input_indices]
        Y = data[:,self.output_indices]

        # Remove unwanted data at start and end
        X = X[run_config.start_time:run_config.end_time]
        Y = Y[run_config.start_time:run_config.end_time]

        # Normalize
        X = normalize(X, norm='max', axis=0)
        Y = normalize(Y, norm='max', axis=0)

        return X, Y
        