

import time
import numpy as np
import random

import matplotlib.pyplot as plt


def load_csv(path):
    """
    Load the CSV form of MNIST data without any external library
    :param path: the path of the csv file
    :return:
        data: A list of list where each sub-list with 28x28 elements
              corresponding to the pixels in each image
        labels: A list containing labels of images
    """
    data = []
    labels = []
    with open(path, 'r') as fp:
        images = fp.readlines()
        images = [img.rstrip() for img in images]

        for img in images:
            img_as_list = img.split(',')
            y = int(img_as_list[0])  # first entry as label
            x = img_as_list[1:]
            x = [int(px) / 255 for px in x]
            data.append(x)
            labels.append(y)
    return data, labels


def load_mnist_trainval():
    """
    Load MNIST training data with labels
    :return:
        train_data: A list of list containing the training data
        train_label: A list containing the labels of training data
        val_data: A list of list containing the validation data
        val_label: A list containing the labels of validation data
    """
    # Load training data
    print("Loading training data...")
    data, label = load_csv('./data/mnist_train.csv')
    assert len(data) == len(label)
    print("Training data loaded with {count} images".format(count=len(data)))

    # split training/validation data
    train_data = None
    train_label = None
    val_data = None
    val_label = None
    train_data = data[:int((len(data) + 1) * .80)]
    train_label = label[:int((len(label) + 1) * .80)]
    val_data = data[int((len(data) + 1) * .80):]
    val_label = label[int((len(label) + 1) * .80):]

    return train_data, train_label, val_data, val_label


def load_mnist_test():
    """
        Load MNIST testing data with labels
        :return:
            data: A list of list containing the testing data
            label: A list containing the labels of testing data
        """
    # Load training data
    print("Loading testing data...")
    data, label = load_csv('./data/mnist_test.csv')
    assert len(data) == len(label)
    print("Testing data loaded with {count} images".format(count=len(data)))

    return data, label


def generate_batched_data(data, label, batch_size=32, shuffle=False, seed=None):
    """
    Turn raw data into batched forms
    :param data: A list of list containing the data where each inner list contains 28x28
                 elements corresponding to pixel values in images: [[pix1, ..., pix784], ..., [pix1, ..., pix784]]
    :param label: A list containing the labels of data
    :param batch_size: required batch size
    :param shuffle: Whether to shuffle the data: true for training and False for testing
    :return:
        batched_data: (List[np.ndarray]) A list whose elements are batches of images.
        batched_label: (List[np.ndarray]) A list whose elements are batches of labels.
    """
    batched_data = None
    batched_label = None
    if seed:
        random.seed(seed)
        np.random.seed(seed)
    data = np.array(data)
    label = np.array(label)
    if shuffle:
        idx = np.random.permutation(len(data))
        data, label = data[idx], label[idx]
    number_of_batch = int(np.shape(data)[0]/batch_size)
    batched_data = np.array_split(data,number_of_batch)
    batched_label = np.array_split(label,number_of_batch)


    return batched_data, batched_label


def train(epoch, batched_train_data, batched_train_label, model, optimizer, debug=True):
    """
    A training function that trains the model for one epoch
    :param epoch: The index of current epoch
    :param batched_train_data: A list containing batches of images
    :param batched_train_label: A list containing batches of labels
    :param model: The model to be trained
    :param optimizer: The optimizer that updates the network weights
    :return:
        epoch_loss: The average loss of current epoch
        epoch_acc: The overall accuracy of current epoch
    """
    epoch_loss = 0.0
    hits = 0
    count_samples = 0.0
    for idx, (input, target) in enumerate(zip(batched_train_data, batched_train_label)):

        start_time = time.time()
        loss, accuracy = model.forward(input, target)

        optimizer.update(model)
        epoch_loss += loss
        hits += accuracy * input.shape[0]
        count_samples += input.shape[0]

        forward_time = time.time() - start_time
        if idx % 10 == 0 and debug:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Batch Time {batch_time:.3f} \t'
                   'Batch Loss {loss:.4f}\t'
                   'Train Accuracy ' + "{accuracy:.4f}" '\t').format(
                epoch, idx, len(batched_train_data), batch_time=forward_time,
                loss=loss, accuracy=accuracy))
    epoch_loss /= len(batched_train_data)
    epoch_acc = hits / count_samples

    if debug:
        print("* Average Accuracy of Epoch {} is: {:.4f}".format(epoch, epoch_acc))
    return epoch_loss, epoch_acc


def evaluate(batched_test_data, batched_test_label, model, debug=True):
    """
    Evaluate the model on test data
    :param batched_test_data: A list containing batches of test images
    :param batched_test_label: A list containing batches of labels
    :param model: A pre-trained model
    :return:
        epoch_loss: The average loss of current epoch
        epoch_acc: The overall accuracy of current epoch
    """
    epoch_loss = 0.0
    hits = 0
    count_samples = 0.0
    for idx, (input, target) in enumerate(zip(batched_test_data, batched_test_label)):

        loss, accuracy = model.forward(input, target, mode='valid')

        epoch_loss += loss
        hits += accuracy * input.shape[0]
        count_samples += input.shape[0]
        if debug:
            print(('Evaluate: [{0}/{1}]\t'
                   'Batch Accuracy ' + "{accuracy:.4f}" '\t').format(
                idx, len(batched_test_data), accuracy=accuracy))
    epoch_loss /= len(batched_test_data)
    epoch_acc = hits / count_samples

    return epoch_loss, epoch_acc


def plot_curves(train_loss_history, train_acc_history, valid_loss_history, valid_acc_history):
    """
    Plot learning curves with matplotlib. Make sure training loss and validation loss are plot in the same figure and
    training accuracy and validation accuracy are plot in the same figure too.
    :param train_loss_history: training loss history of epochs
    :param train_acc_history: training accuracy history of epochs
    :param valid_loss_history: validation loss history of epochs
    :param valid_acc_history: validation accuracy history of epochs
    :return: None, save two figures in the current directory
    """

    num_of_epoch = len(train_loss_history)
    print("train_loss_history:",np.mean(train_loss_history))
    print("valid_loss_history:",np.mean(valid_loss_history))
    x = np.arange(0.0, num_of_epoch, 1)
    fig, ax = plt.subplots()
    train_loss = ax.plot(x, train_loss_history, label='Train Loss')
    val_loss = ax.plot(x, valid_loss_history, label='Validation Loss')
    ax.legend(loc='center right')
    ax.set(xlabel='Epochs', ylabel='Loss', title='Loss Curve')
    ax.grid()
    fig.savefig("Loss Curve.png")

    print("train_acc_history:",np.mean(train_acc_history))
    print("valid_acc_history:",np.mean(valid_acc_history))
    x = np.arange(0.0, num_of_epoch, 1)
    fig, ax = plt.subplots()
    train_acc = ax.plot(x, train_acc_history, label='Train Accuracy')
    val_acc = ax.plot(x, valid_acc_history, label='Validation Accuracy')
    ax.legend(loc='center right')
    ax.set(xlabel='Epochs', ylabel='Loss', title='Accuracy')
    ax.grid()
    fig.savefig("Accuracy Curve.png")

    # plt.show()
