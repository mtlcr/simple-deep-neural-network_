

from ._base_optimizer import _BaseOptimizer
import numpy as np


class SGD(_BaseOptimizer):
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        super().__init__(learning_rate, reg)

    def update(self, model):
        """
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        """
        self.apply_regularization(model)

        model.weights['W1'] = np.array(model.weights['W1']) - self.learning_rate * np.array(model.gradients['W1'])
        if 'W2' in model.weights:
            model.weights['W2'] = np.array(model.weights['W2']) - self.learning_rate * np.array(model.gradients['W2'])

