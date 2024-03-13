

# Do not use packages that are not in standard distribution of python
import numpy as np

from ._base_network import _baseNetwork


class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10):
        """
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => (activation) => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        """
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        """
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        """
        loss = None
        gradient = None
        accuracy = None
        Z = np.dot(X,self.weights['W1'])
        A = self.ReLU(Z)
        p = self.softmax(A)
        loss = self.cross_entropy_loss(p,y)
        accuracy = self.compute_accuracy(p,y)

        if mode != 'train':
            return loss, accuracy

        batch_size = X.shape[0]
        y_one_hot = np.zeros(p.shape)
        y_one_hot[range(batch_size), y] = 1
        dL_dA = (p - y_one_hot) / batch_size
        dA_dZ = self.ReLU_dev(A)
        dL_dZ = dL_dA * dA_dZ
        dL_dW = np.dot(np.transpose(X), dL_dZ)
        self.gradients['W1']  = dL_dW
        return loss, accuracy
