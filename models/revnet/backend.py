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
    def __init__(self, sequential):

        # Layer 1
        sequential.add(LSTM(1000, return_sequences=True))
        sequential.add(Dropout(0.2))

        # Layer 2
        sequential.add(LSTM(100, return_sequences=True))
        sequential.add(LSTM(100, return_sequences=True))
        sequential.add(Dropout(0.2))

        # Layer 3
        sequential.add(LSTM(100, return_sequences=True))
        sequential.add(LSTM(100, return_sequences=True))
        sequential.add(Dropout(0.2))
        
        sequential.add(Flatten())

        self._memory = sequential

class SmallRevNetMemory(BaseMemory):
    def __init__(self, sequential):

        # Layer 1
        sequential.add(LSTM(100, return_sequences=True))
        sequential.add(LSTM(100, return_sequences=True))
        sequential.add(Dropout(0.2))

        sequential.add(Flatten())

        # Dense layers
        sequential.add(Dense(100))
        sequential.add(Dropout(0.2))
        
        sequential.add(Dense(100))
        sequential.add(Dropout(0.2))

        self._memory = sequential