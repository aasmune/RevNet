class BaseVisualization(object):
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config

    def plot_loss(self):
        raise NotImplementedError

    def plot_accuracy(self):
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError