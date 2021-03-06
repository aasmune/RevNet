from keras.layers import Input, Flatten, Dense
from keras.models import Model
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
        input_layer = Input(shape=(recursive_depth, number_of_inputs))

        if architecture == "LargeMemory":
            self.backend = LargeRevNetMemory(input_layer)
        elif architecture == "SmallMemory":
            self.backend = SmallRevNetMemory(input_layer)
        
        memory = self.backend.memory

        output = Dense(self.config.data.number_of_cells, activation="linear")(memory)

        # Create model
        self.model = Model(input_layer, output)

        self.model.compile(
              loss=self.config.model.loss,
              optimizer=self.config.model.optimizer,
              metrics=['accuracy'])

        self.model.summary()