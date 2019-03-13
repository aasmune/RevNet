from base.base_data_loader import BaseDataLoader
from keras.datasets import mnist


class RevNetDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(SimpleMnistDataLoader, self).__init__(config)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = (0,0), (0,0)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def _create_channels(self):
        
        
