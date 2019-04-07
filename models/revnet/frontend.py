from keras.layers import Input, Flatten, Dense, LSTM, Dropout
from keras.models import Sequential
from base.base_model import BaseModel

from models.revnet.backend import LargeRevNetMemory, SmallRevNetMemory

class RevNet(BaseModel):
    def __init__(self, config):
        super(RevNet, self).__init__(config)

        self.build_model()

    def build_model(self):
        recursive_depth = self.config.model.recursive_depth
        number_of_inputs = len(self.config.data.other_channels)
        architecture = self.config.model.architecture
        # Build the model
        model = Sequential()
        # model.add()

        # if architecture == "LargeMemory":
        #     self.backend = LargeRevNetMemory(model)
        # elif architecture == "SmallMemory":
        #     self.backend = SmallRevNetMemory(model)
        
        # model = self.backend.memory

        # Layer 1
        model.add(LSTM(100, return_sequences=True, input_shape=(recursive_depth, number_of_inputs)))
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(100, return_sequences=True))
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.2))

        # Dense layers
        model.add(Dense(100))
        model.add(Dropout(0.2))
        
        model.add(Dense(100))
        model.add(Dropout(0.2))
        
        model.add(Flatten())

        model.add(Dense(self.config.data.number_of_cells, activation="linear"))

        # Create model
        self.model = model

        self.model.compile(
              loss=self.config.model.loss,
              optimizer=self.config.model.optimizer,
              metrics=['accuracy'])

        self.model.summary()