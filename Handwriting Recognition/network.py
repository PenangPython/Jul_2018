"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.

Source from http://neuralnetworksanddeeplearning.com/
Modified by Eric K. (17 July 2018)
"""

# Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt


class Network(object):
    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""

        self.count = 0
        self.training_cost = []
        self.training_accuracy = []
        self.test_result = []
        self.evaluation_cost = []
        self.evaluation_accuracy = []

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.best_set = ()

    def plot_accuracy(self):
        training = self.training_accuracy
        validation = self.evaluation_accuracy
        total_epoch = len(training)
        plt.plot(range(0, total_epoch), training, color='lightblue',
                 linewidth=3, marker='^')
        plt.plot(range(0, total_epoch), validation, color='red',
                 linewidth=3, marker='^')
        plt.axis([0, total_epoch-1, min(min(training), min(validation)),
                  max(max(training), max(validation))])
        plt.show()

    def plot_cost(self):
        training = self.training_cost
        validation = self.evaluation_cost
        total_epoch = len(training)
        plt.plot(range(0, total_epoch), training, color='lightblue',
                 linewidth=3, marker='^')
        plt.plot(range(0, total_epoch), validation, color='red',
                 linewidth=3, marker='^')
        plt.axis([0, total_epoch-1, min(min(training), min(validation)),
                  max(max(training), max(validation))])
        plt.show()

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, evaluation_data=None, eta_reduce_rate=1.0, lmbda=0.0):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        # eta_reduce_rate=1.0, means learning rate did not drop across iteration
        # eta_reduce_rate=0.95, means it drop to 5% every iteration

        print "##### Training Started #####"
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            tempETA = eta * (eta_reduce_rate**j)
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, tempETA)

            if test_data:
                result = self.evaluate(test_data)
                self.test_result.append(result)
                print "=> Epoch {0}: {1} / {2} - ETA:{3:0.4f}".format(
                    j, result, n_test, tempETA)
            else:
                print "=> Epoch {0} Completed".format(j)

            # Calculate Training Cost & Accuracy
            cost = self.total_cost(training_data, lmbda)
            accuracy = self.accuracy(training_data, convert=True)
            self.training_cost.append(cost)
            self.training_accuracy.append(accuracy)
            self.count += 1
            print(
                "Training Cost : {0:0.4f} - Accuracy : {1:0.4f}".format(cost, accuracy))

            if evaluation_data:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                accuracy = self.accuracy(evaluation_data)
                # Save Weights & Biases in best_set, if accuracy higher than previous set
                if (len(self.evaluation_accuracy) > 1):
                    if (accuracy > max(self.evaluation_accuracy)):
                        self.best_set = (self.weights, self.biases)

                self.evaluation_cost.append(cost)
                self.evaluation_accuracy.append(accuracy)
            print(
                "Evaluation Cost : {0:0.4f} - Accuracy : {1:0.4f}".format(cost, accuracy))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        cost = self.cost_derivative(activations[-1], y)
        delta = cost * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in data]
        count = float(sum(int(x == y) for (x, y) in results))
        return count / len(data)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives partial C_x / partial a for the output activations."""
        return (output_activations-y)

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = self.vectorized_result(y)
            cost += cross_entrophy(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def vectorized_result(self, j):
        """Return a 10-dimensional unit vector with a 1.0 in the j'th position
        and zeroes elsewhere.  This is used to convert a digit (0...9)
        into a corresponding desired output from the neural network.
        """
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e

    def guess(self, input):
        """ this function will try to guess the input is which digit
        input must be a single data only, no expected result"""
        result = np.argmax(self.feedforward(input))
        return result

    def check_error(self, input_data):
        """this function will return all wrong guess record"""
        errors = []
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in input_data]
        for x, y in test_results:
            if (x != y):
                errors.append((x, y))
        return errors


# Miscellaneous functions
def cross_entrophy(a, y):
    # a is the output, y is expected result
    cost = np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    return cost


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    s = sigmoid(z)
    return s*(1-s)
