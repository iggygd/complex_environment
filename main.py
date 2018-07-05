import math
import numpy as np
import pprint
import pickle
import random

np.set_printoptions(suppress=True, threshold = np.nan)

from mnist import MNIST
mndata = MNIST('./data')
mndata.gz = True
images, labels = mndata.load_training()
t_images, t_labels = mndata.load_testing()

class LayerException(Exception):
    pass

class Printer:
    def pprint_file(filename, output):
        with open(filename, 'w') as out:
            out.write(pprint.pformat(output))

class Activator:
    class sigmoid:
        def f(x, a = 1, b = 0):
            return 1 / (1 + np.exp(-a*(x+b)))

        def f_prime(x):
            return Activator.sigmoid.f(x)*(1-Activator.sigmoid.f(x))

    class relu:
        def f(x):
            return x * (x > 0)

        def f_prime(x):
            a = 0 if x <= 0 else 1
            return a

    class softplus:
        def f(x, a = 1, b = 0):
            return np.log(1 + np.exp(x))

        def f_prime(x):
            return 1 / (1 + np.exp(-a*(x+b)))

class Layer:
    def __init__(self, size_a, size_i, activator):
        self.A = np.zeros((size_a, 1))
        if size_i is None:
            self.W = None
            self.Z = None
        else:
            self.W = np.zeros((size_a, size_i))
            self.Z = np.zeros((size_a, size_i))
        self.D = np.zeros((size_a, 1))
        self.activator = activator
        self.n = size_i
        self.m = size_a

    #def next_layer(self, output_size):
        #self.W = np.zeros((output_size, self.n))
        #self.Z = np.zeros((output_size, self.n))
        #self.m = output_size

    def debug(self):
        return vars(self)

class NeuralNetwork:
    def __init__(self, input_size, activator):
        self.num_layers = 1
        self.input_size = input_size
        self.output_size = None

        self.layers = []
        self.layers.append(Layer(input_size, None, activator))

        self.finalized = False

    def add_layer(self, layer_size, activator, forced = False):
        if self.finalized is True and forced is True:
            raise LayerException('cannot add layer: output layer is already initialized')

        #self.layers[-1].next_layer(layer_size)
        self.layers.append(Layer(layer_size, self.layers[-1].m, activator))

    def end_layer(self, output_size, activator, forced = False):
        if self.finalized is True and forced is True:
            raise LayerException('cannot end layer: output layer is already initialized')

        #self.layers[-1].next_layer(output_size)
        self.layers.append(Layer(output_size, self.layers[-1].m, activator))

        self.finalized = True

    def call(self, input):
        for l in range(len(self.layers)):
            if l is 0:
                z = None
                x = input
            else:
                z = self.layers[l].W @ x
                x = self.layers[l].activator.f(z)
            self.layers[l].Z = z
            self.layers[l].A = x
        return x

    def error(self, target, output):
        return 1/2*(np.linalg.norm(target - output))**2

    def error_prime(self, outputs, target):
        return (outputs - target)

    def randomize(self):
        if self.finalized is False:
            raise LayerException('error: output layer is not initialized')

        for l in range(1, len(self.layers)):
            self.layers[l].W = np.random.randn(self.layers[l].m, self.layers[l].n)

    def backprop(self, target):
        delta = np.multiply(self.error_prime(self.layers[-1].A, target), self.layers[-1].activator.f_prime(self.layers[-1].Z))
        dnabla = delta @ self.layers[-2].A.T
        self.layers[-1].dnabla = dnabla
        for l in range(2, len(self.layers)):
            delta = np.multiply((self.layers[-l+1].W.T @ delta), self.layers[-l].activator.f_prime(self.layers[-l].Z))
            dnabla = delta @ self.layers[-l-1].A.T
            self.layers[-l].dnabla = dnabla

    def mini_batch(self, batch, eta):
        for layer in self.layers:
            layer.nabla = 0

        for i, o in batch:
            input = Activator.sigmoid.f(np.array([i]).T)
            target = np.zeros((10,1))
            target[o] = 1
            self.call(input)
            self.backprop(target)

            for l in range(1,len(self.layers)):
                self.layers[l].nabla = self.layers[l].nabla + self.layers[l].dnabla

        for l in range(1, len(self.layers)):
            self.layers[l].W = self.layers[l].W - (eta/len(batch))*self.layers[l].nabla

    def training_error(self, set):
        input = Activator.sigmoid.f((np.array([set[0]]).T))
        target = np.zeros((10,1))
        target[set[1]] = 1
        error = self.error(target, self.call(input))
        return error

    def debug_layers(self):
        layers = []
        for layer in self.layers:
            layers.append(layer.debug())

        return layers

    def debug(self):
        return vars(self)


input_size = 28*28
output_size = 10

classifier = NeuralNetwork(input_size, Activator.sigmoid)
classifier.add_layer(input_size, Activator.sigmoid)
classifier.end_layer(output_size, Activator.sigmoid)
classifier.randomize()

#print(classifier.call(images[1]))


training_data = list(zip(images, labels))
testing_data = list(zip(t_images, t_labels))


def train(batch_size, eta):
    for i in range(int(len(training_data)/batch_size)):
        batch = training_data[i*batch_size:i*batch_size+batch_size]
        classifier.mini_batch(batch, eta)
        test = random.choice(testing_data)
        print(i)
        print(classifier.training_error(test))

def test():
    for set in testing_data:
        input = Activator.sigmoid.f((np.array([set[0]]).T))
        target = np.zeros((10,1))
        target[set[1]] = 1
        output = classifier.call(input)
        print("-----------"+str(set[1]))
        print(target)
        print(output)

train(100, 0.5)
test()
