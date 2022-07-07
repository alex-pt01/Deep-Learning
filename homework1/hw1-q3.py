#!/usr/bin/env python

# Deep Learning Homework 1
import argparse
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import utils

def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        eta = 1
        y_hat = 1 if self.W.dot(x_i).all() >= 0 else -1
        if y_hat != y_i:
            self.W += eta * y_i * x_i

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels 

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """ 
        y_hat = np.argmax(self.W.dot(x_i))
        if y_hat != y_i:
            self.W[y_i, :] += x_i
            self.W[y_hat, :] -= x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        label_scores = self.W.dot(x_i)[:, None]
        # One-hot vector with the true label (num_labels x 1).
        y_one_hot = np.zeros((np.size(self.W, 0), 1))
        y_one_hot[y_i] = 1
        # Softmax function.
        # This gives the label probabilities according to the model (num_labels x 1).
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))
        # SGD update. W is num_labels x num_features.
        self.W += learning_rate * (y_one_hot - label_probabilities) * x_i[None, :]
      

class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size, layers):
        # Initialize an MLP with a single hidden layer.
        self.W1 = np.random.normal(0.1, 0.1, size=(hidden_size, n_features))
        self.b1 = np.zeros(hidden_size)

        self.W2 = np.random.normal(0.1, 0.1, size=(n_classes,hidden_size))
        self.b2 = np.zeros(n_classes)

    def relu(self, x):
        return (x > 0) * x 

    def Drelu(self, x):
        return (x > 0) * 1

    def compute_label_probabilities(self, output):
        # softmax transformation.
        probs = np.exp(output) / np.sum(np.exp(output))
        return probs

    def forward(self, x):
        hiddens = []
        
        #1st iteration
        h1 = x 
        z1 = self.W1.dot(h1) + self.b1
        hiddens.append(self.relu(z1))

        #2nd iteration
        h2 = hiddens[0]
        z2 = self.W2.dot(h2) + self.b2
        
        return z2, hiddens

    def backward(self,x, y, output, hiddens):
        output -= output.max()
        probs = self.compute_label_probabilities(output)

        y_one_hot_vector = np.zeros((10,))
        y_one_hot_vector[y] = 1
        grad_z = probs - y_one_hot_vector

        grad_weights = []
        grad_biases = []

        #1st iteration
        h2 = hiddens[0]
        grad_weights.append(grad_z[:, None].dot(h2[:, None].T))
        grad_biases.append(grad_z)
        grad_h2 = self.W2.T.dot(grad_z)
        grad_z2 = np.multiply(grad_h2,self.Drelu(h2))

        #2nd iteration
        h1 = x
        grad_weights.append(grad_z2[:, None].dot(h1[:, None].T))
        grad_biases.append(grad_z2)
        grad_h1 = self.W1.T.dot(grad_z2)
        grad_z1 = np.multiply(grad_h1,self.Drelu(h1))

        grad_weights.reverse()
        grad_biases.reverse()

        return grad_weights, grad_biases

    def predict_label(self, output):
        # The most probable label is also the label with the largest logit.
        y_hat = np.zeros_like(output)
        y_hat[np.argmax(output)] = 1
        return y_hat

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required

        predicted_labels = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            output, _ = self.forward(X[i])
            y_hat = np.argmax(self.predict_label(output))
            predicted_labels[i] = y_hat
        return predicted_labels  

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def update_weight(self, grad_weights, grad_biases, learning_rate):
        self.W1 -= learning_rate * grad_weights[0]
        self.b1 -= learning_rate * grad_biases[0]
        self.W2 -= learning_rate * grad_weights[1]
        self.b2 -= learning_rate * grad_biases[1]
    
    def train_epoch(self, X, y, learning_rate=0.001):
        for x, y in zip(X, y):
            output, hiddens = self.forward(x)
            grad_weights, grad_biases = self.backward(x, y, output, hiddens)
            self.update_weight(grad_weights, grad_biases, learning_rate)



def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")

    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()