import keras
import keras.layers as kr_ly
import keras.models as kr_md
import keras.initializers as kr_in

import numpy as np

class DenseNN:
    def __init__(self, i_num, o_num):
        self.randomizer = kr_in.RandomNormal(mean=0.0, stddev=1, seed=None)

        self.inputs = kr_ly.Input(shape=(i_num,))
        self.x = kr_ly.LSTM(i_num, activation='tanh', recurrent_activation='hard_sigmoid', kernel_initializer=self.randomizer)(self.inputs)
        #self.x = kr_ly.Dense(i_num, activation='sigmoid', kernel_initializer=self.randomizer)(self.x)
        self.outputs = kr_ly.LSTM(o_num, activation='tanh', recurrent_activation='hard_sigmoid', kernel_initializer=self.randomizer)(self.x)

        self.model = kr_md.Model(inputs=self.inputs, outputs=self.outputs)
        #self.model.compile(optimizer='SGD',
                          #loss='mean_squared_error',
                          #metrics=['accuracy'])

    def debug(self):
        self.model.summary()

NeuralNet = DenseNN(8, 8)
ins = []
for i in range(100):
    ins.append(np.random.randn(1,8))

for input in ins:
    print(NeuralNet.model.predict(input))

for layer in NeuralNet.model.layers:
    print(layer.get_config())

print(NeuralNet.model.inputs)
print(NeuralNet.model.outputs)

NeuralNet.debug()
