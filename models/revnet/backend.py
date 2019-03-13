from base.base_model import BaseModel
from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Flatten, Dense

class BaseMemory:
    def __init__(self, recursive_depth, number_of_inputs):
        raise NotImplementedError("Not implemented")
        self._memory = None

    @property
    def memory(self):
        return self._memory


class LargeRevNetMemory(BaseMemory):
    def __init__(self, input_layer):

        # Layer 1
        x = LSTM(1000, return_sequences=True)(input_layer)
        x = Dropout(0.2)(x)

        # Layer 2
        x = LSTM(100, return_sequences=True)(x)
        x = LSTM(100, return_sequences=True)(x)
        x = Dropout(0.2)(x)

        # Layer 3
        x = LSTM(100, return_sequences=True)(x)
        x = LSTM(100, return_sequences=True)(x)
        x = Dropout(0.2)(x)

        self._memory = x