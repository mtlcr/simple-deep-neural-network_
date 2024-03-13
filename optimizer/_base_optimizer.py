


class _BaseOptimizer:
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        self.learning_rate = learning_rate
        self.reg = reg

    def update(self, model):
        pass

    def apply_regularization(self, model):
        """
        Apply L2 penalty to the model. Update the gradient dictionary in the model
        :param model: The model with gradients
        :return: None, but the gradient dictionary of the model should be updated
        """

        key_list = []
        for key in model.gradients.keys():
            key_list.append(key)
        λ = self.reg
        model.gradients['W1'] += λ * model.weights['W1']
        if "W2" in key_list:
            model.gradients['W2'] += λ * model.weights['W2']
