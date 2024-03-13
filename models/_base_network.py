

# Do not use packages that are not in standard distribution of python
import numpy as np
import sys

class _baseNetwork:
    def __init__(self, input_size=28 * 28, num_classes=10):
        self.input_size = input_size
        self.num_classes = num_classes

        self.weights = dict()
        self.gradients = dict()

    def _weight_init(self):
        pass

    def forward(self):
        pass

    def softmax(self, scores):
        """
        Compute softmax scores given the raw output from the model

        :param scores: raw scores from the model (N, num_classes)
        :return:
            prob: softmax probabilities (N, num_classes)
        """
        prob = None

        max_scores = np.max(scores, axis=-1, keepdims=True)
        numerator = np.exp(scores - max_scores)
        denominator = np.sum(numerator, axis=-1, keepdims=True)
        prob = numerator/denominator


        return prob

    def cross_entropy_loss(self, x_pred, y):
        """
        Compute Cross-Entropy Loss based on prediction of the network and labels
        :param x_pred: Probabilities from the model (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The computed Cross-Entropy Loss
        """
        loss = None
        onehot = np.zeros(np.shape(x_pred))
        onehot[np.arange(len(y)), y] = 1
        loss = -np.mean(np.sum(onehot * np.log(x_pred), axis=1))
        return loss

    def compute_accuracy(self, x_pred, y):
        """
        Compute the accuracy of current batch
        :param x_pred: Probabilities from the model (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The accuracy of the batch
        """
        acc = None
        x = [i.argmax() for i in x_pred]
        acc_list = np.zeros(np.shape(x))
        acc_list = np.where(x==y,1,acc_list)
        acc = np.mean(acc_list)

        return acc

    def sigmoid(self, X):
        """
        Compute the sigmoid activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: the value after the sigmoid activation is applied to the input (N, layer size)
        """
        out = None
        out = 1/(1+np.exp(-X))
        return out

    def sigmoid_dev(self, x):
        """
        The analytical derivative of sigmoid function at x
        :param x: Input data
        :return: The derivative of sigmoid function at x
        """
        ds = None
        ds = np.multiply(1/(1+np.exp(-x)), (1 - 1/(1+np.exp(-x))))
        return ds

    def ReLU(self, X):
        """
        Compute the ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: the value after the ReLU activation is applied to the input (N, layer size)
        """
        out = None
        out = np.maximum(0,X)
        return out

    def ReLU_dev(self, X):
        """
        Compute the gradient ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: gradient of ReLU given input X
        """
        out = None
        relu = np.maximum(0,X)
        relu[relu > 0] = 1
        relu[relu <= 0] = 0
        out = relu
        return out
