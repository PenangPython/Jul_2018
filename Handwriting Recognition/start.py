import network as network
import drawing as drawing
import pickle as pickle
import mnist_loader

file_name = "my_network"

# Load training data and validation data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Network Hyper-parameters
net_design = [784, 300, 100, 100, 10]
net_design = [784, 10]
learning_rate = 1.5
rate_reduce = 0.95
epoch = 3
mini_batch = 25

# Initiate network
net = network.Network(net_design)

# Train Network
net.SGD(training_data, epoch, mini_batch, learning_rate,
        evaluation_data=validation_data, eta_reduce_rate=rate_reduce, lmbda=1.0)

net.plot_accuracy()

# Save Network to file
with open(file_name, 'wb') as f:
    pickle.dump(net, f)

# Load Network from file
with open(file_name, 'rb') as input_file:
    net = pickle.load(input_file)

# Test Network with Drawing program
drawing.Drawing(net)
