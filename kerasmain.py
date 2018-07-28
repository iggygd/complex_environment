import keras
import keras.layers as kr_ly
import keras.models as kr_md
import keras.initializers as kr_in

import funcs, evolver
import random, math, os, sys, copy, time
import pyqtree_l as qtree

import numpy as np
import pygame as pg
from pygame.locals import*

class WorldObj:
    def __init__(self, pos, colour, size):
        self.id = 0
        self.colour = colour
        self.sound = 0
        self.size = size
        self.pos = pos

class SmartObjNN:
    def __init__(self, vis_in, snd_in, fdbk_in, timesteps, mov_degrees):
        self.randomizer = kr_in.RandomNormal(mean=0.0, stddev=1, seed=None)
        self.timesteps = timesteps
        self.vis_num = vis_in*3
        self.snd_num = snd_in
        self.fdbk_num = fdbk_in

        #o_size = mov_degrees + left/right(2) + fdbk_size
        self.out_num = len(mov_degrees) + 2 + fdbk_in

        self.vis_inputs = kr_ly.Input(shape=(timesteps, vis_in*3))
        self.snd_inputs = kr_ly.Input(shape=(timesteps, snd_in))
        self.fdbk_inputs = kr_ly.Input(shape=(timesteps, self.out_num))

        self.vis = kr_ly.LSTM(vis_in*3, activation='hard_sigmoid', recurrent_activation='hard_sigmoid', kernel_initializer=self.randomizer)(self.vis_inputs)
        self.snd = kr_ly.LSTM(snd_in, activation='hard_sigmoid', recurrent_activation='hard_sigmoid', kernel_initializer=self.randomizer)(self.snd_inputs)
        self.fdbk = kr_ly.LSTM(self.out_num, activation='hard_sigmoid', recurrent_activation='hard_sigmoid', kernel_initializer=self.randomizer)(self.fdbk_inputs)

        self.x = kr_ly.concatenate([self.vis, self.snd, self.fdbk])
        self.x = kr_ly.Dense(vis_in*3+snd_in+fdbk_in, activation='hard_sigmoid', kernel_initializer=self.randomizer)(self.x)
        self.outputs = kr_ly.Dense(self.out_num, activation='hard_sigmoid', kernel_initializer=self.randomizer)(self.x)

        self.model = kr_md.Model(inputs=[self.vis_inputs, self.snd_inputs, self.fdbk_inputs], outputs=self.outputs)

    def get_vis_num(self):
        return self.vis_num

    def get_snd_num(self):
        return self.snd_num

    def get_fdbk_num(self):
        return self.fdbk_num

    def get_vis_num(self):
        return self.out_num

    def debug(self):
        self.model.summary()

    def call(self, x):
        return self.model.predict(x)

    def evolve_layers(self):
        for layer in self.model.layers:
            layer.set_weights(evolver.evolve(layer.get_weights()))

class SmartObj(WorldObj):
    def __init__(self, pos, colour, size):
        super().__init__(pos, colour, size)
        self.vis_len = 0
        self.snd_len = 0

    #Define the creature here:
    def _init_characteristics(self, vis_intervals, snd_intervals, mov_degrees, vis_len = 20, snd_len = 40):
        self.vis_intervals = vis_intervals
        self.vis_vectors = []
        for angle in self.vis_intervals:
            self.vis_vectors.append(funcs.to_vector(math.radians(angle)))
        self.vis_fidelity = len(vis_intervals) - 1
        if len(vis_intervals) > 1:
            self.vis_slice = vis_intervals[1] - vis_intervals[0]

        self.snd_intervals = snd_intervals
        self.snd_vectors = []
        for angle in self.snd_intervals:
            self.snd_vectors.append(funcs.to_vector(math.radians(angle)))
        self.snd_fidelity = len(snd_intervals) - 1
        if len(snd_intervals) > 1:
            self.snd_slice = snd_intervals[1] - snd_intervals[0]

        self.mov_degrees = mov_degrees
        self.mov_vectors = []
        for angle in self.mov_degrees:
            self.mov_vectors.append(funcs.to_vector(math.radians(angle)))
        if len(mov_degrees) > 1:
            self.mov_slice = mov_degrees[1] - mov_degrees[0]

        self.vis_len = vis_len
        self.snd_len = snd_len
        self.moment = 1/4*self.mass*self.size**2

    #Define the model in the SmartObjNN class:
    def _init_brain(self, fdbk_in, timesteps):
        self.brain = SmartObjNN(self.vis_fidelity, self.snd_fidelity, fdbk_in, timesteps, self.mov_degrees)

    #Must be called
    def _init_in_out(self):
        self.vis_array = np.zeros((1, self.brain.timesteps, self.brain.vis_num))
        self.snd_array = np.zeros((1, self.brain.timesteps, self.brain.snd_num))
        self.fdbk_array = np.zeros((1, self.brain.timesteps, self.brain.out_num))

        self.out_array = np.zeros(self.brain.out_num)

    def _rand_in_out(self):
        self.vis_array = np.random.rand(1, self.brain.timesteps, self.brain.vis_num)
        self.snd_array = np.random.rand(1, self.brain.timesteps, self.brain.snd_num)
        self.fdbk_array = np.random.rand(1, self.brain.timesteps, self.brain.out_num)

        self.out_array = np.random.rand(self.out_num)

    def debug_in_out(self):
        print(self.vis_array)
        print(self.snd_array)
        print(self.fdbk_array)

        print(self.out_array)

    def handle_input(self, nearby, output):
        self.update_sensory(nearby)
        self.update_feedback()

    def handle_output(self):
        self.out_array = self.brain.call([self.vis_array, self.snd_array, self.fdbk_array])

    def update_sensory(self, nearby):
        for i in range(1, self.brain.timesteps):
            self.vis_array[0][i-1] = self.vis_array[0][i]
            self.snd_array[0][i-1] = self.snd_array[0][i]

        self.vis_array[0][-1] = np.zeros(self.brain.vis_num)
        self.snd_array[0][-1] = np.zeros(self.brain.snd_num)

        for body in nearby:
            if body is self:
                continue
            pnt_vec = body.pos - self.pos
            vec_len = np.linalg.norm(pnt_vec)

            #Vision first
            if vec_len < self.vis_len:
                deg = funcs.real_angle(pnt_vec)
                i = 0
                j = 0
                arr_len = len(self.vis_intervals)
                for interval in self.vis_intervals:
                    if j == arr_len - 1:
                        break
                    if deg > interval and deg < self.vis_intervals[j + 1]:
                        self.vis_array[0][-1][i] = min(self.vis_array[0][-1][i] + body.colour[0], 255)
                        self.vis_array[0][-1][i+1] = min(self.vis_array[0][-1][i+1] + body.colour[1], 255)
                        self.vis_array[0][-1][i+2] = min(self.vis_array[0][-1][i+2] + body.colour[2], 255)
                        i += 3
                        j += 1
                    else:
                        i += 3
                        j += 1

            #Sound next
            if vec_len < self.snd_len:
                deg = funcs.real_angle(pnt_vec)
                j = 0
                arr_len = len(self.snd_intervals)
                for interval in self.snd_intervals:
                    if j == arr_len - 1:
                        break
                    if deg > interval and deg < self.vis_intervals[j + 1]:
                        self.snd_array[0][-1][j] = body.sound
                        j += 1
                    else:
                        j += 1

        #Finalize
        self.vis_array[0][-1] = self.vis_array[0][-1]/255
        #self.snd_array requires no finalizing

    def update_feedback(self):
        for i in range(1, self.brain.timesteps):
            self.fdbk_array[0][i-1] = self.fdbk_array[0][i]

        self.fdbk_array[0][-1] = self.out_array

    def get_output(self):
        return self.out_array

    def get_input(self):
        return self.vis_array, self.snd_array, self.fdbk_array

class DumbObj(WorldObj):
    def __init__(self, position, colour, size):
        super().__init__(position, colour, size)
        self.energy = 0
        self.mass = 0
        pass

class World:
    def __init__(self, size_1, size_2, dt):
        self.size = np.array([size_1, size_2])
        self.dt = dt
