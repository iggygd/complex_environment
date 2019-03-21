import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras
import keras.layers as kr_ly
import keras.models as kr_md
import keras.initializers as kr_in
import CONST

import funcs
import evolver
from worldobj import WorldObj

import random, math
from collections import deque

import numpy as np
import pygame as pg
import optimizer as op
import pymunk.pygame_util as pm_pg_util

class SmartObjMemory:
    def __init__(self):
        self.container = deque(maxlen=1024)
    
    def remember(self, prev_state, prev_action, rewards, state):
        self.container.append((prev_state, prev_action, rewards, state))

    def get_batch(self, size):
        size = min(size, len(self.container))

        return random.sample(self.container, size)

class SmartObjNN:
    def __init__(self, vis_in, snd_in, fdbk_in, timesteps, mov_degrees):
        self.randomizer = kr_in.RandomNormal(mean=0.0, stddev=1, seed=None)
        self.timesteps = timesteps
        self.vis_num = vis_in*3
        self.snd_num = snd_in
        self.fdbk_num = fdbk_in
        self.epsilon = CONST.epsilon

        #o_size = mov_degrees + left/right(2) + fdbk_size
        self.out_num = len(mov_degrees)*2 + 3# + fdbk_in

        self.vis_inputs = kr_ly.Input(shape=(timesteps, vis_in*3))
        self.snd_inputs = kr_ly.Input(shape=(timesteps, snd_in))
        #self.fdbk_inputs = kr_ly.Input(shape=(timesteps, self.out_num))

        
        self.vis = kr_ly.LSTM(vis_in*3, activation='hard_sigmoid', recurrent_activation='hard_sigmoid', kernel_initializer=self.randomizer)(self.vis_inputs)
        self.snd = kr_ly.LSTM(snd_in, activation='hard_sigmoid', recurrent_activation='hard_sigmoid', kernel_initializer=self.randomizer)(self.snd_inputs)
        #self.fdbk = kr_ly.LSTM(self.out_num, activation='hard_sigmoid', recurrent_activation='hard_sigmoid', kernel_initializer=self.randomizer)(self.fdbk_inputs)
        '''
        self.vis_flat = kr_ly.Flatten(self.vis_inputs)
        self.snd_flat = kr_ly.Flatten(self.snd_inputs)
        self.fdbk_flat = kr_ly.Flatten(self.fdbk_inputs)

        self.vis = kr_ly.Dense(vis_in*3, activation='hard_sigmoid', recurrent_activation='hard_sigmoid', kernel_initializer=self.randomizer)(self.vis_flat)
        self.snd = kr_ly.Dense(snd_in, activation='hard_sigmoid', recurrent_activation='hard_sigmoid', kernel_initializer=self.randomizer)(self.snd_flat)
        self.fdbk = kr_ly.Dense(self.out_num, activation='hard_sigmoid', recurrent_activation='hard_sigmoid', kernel_initializer=self.randomizer)(self.fdbk_flat)
        '''

        self.x = kr_ly.concatenate([self.vis, self.snd])#, self.fdbk])
        self.x = kr_ly.Dense(vis_in*3+snd_in, activation='hard_sigmoid', kernel_initializer=self.randomizer)(self.x)

        self.mov_outputs = kr_ly.Dense(len(mov_degrees)*2, activation='hard_sigmoid', kernel_initializer=self.randomizer)(self.x)
        self.turn_outputs = kr_ly.Dense(3, activation='hard_sigmoid', kernel_initializer=self.randomizer)(self.x)
        #self.fdbk_outputs = kr_ly.Dense(fdbk_in, activation='hard_sigmoid', kernel_initializer=self.randomizer)(self.x)

        #self.outputs = kr_ly.Dense(self.out_num, activation='hard_sigmoid', kernel_initializer=self.randomizer)(self.x)

        self.model = kr_md.Model(inputs=[self.vis_inputs, self.snd_inputs], outputs=[self.mov_outputs, self.turn_outputs])
        self.model.compile(loss='mse', optimizer='Adam')
        self.memory = SmartObjMemory()

    def train(self, size):
        batch = self.memory.get_batch(size)

        for state, action, reward, next_state in batch:
            current_thrust = np.argmax(action[0][0])
            current_turn = np.argmax(action[1][0])

            future_thrust = self.model.predict(next_state)[0]
            future_turn = self.model.predict(next_state)[1]

            target_thrust = reward + CONST.gamma*np.amax(future_thrust[0])
            target_turn = reward + CONST.gamma*np.amax(future_turn[0])

            target_f = self.model.predict(state)

            target_f[0][0][current_thrust] = target_thrust
            target_f[1][0][current_turn] = target_turn

            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > CONST.epsilon_min:
            self.epsilon *= CONST.epsilon_decay
            print(self.epsilon, self)

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
    def __init__(self):
        super().__init__()
        self.vis_len = 0
        self.snd_len = 0
        self.smart = True

        self.time = 0

        self.old_fitness = 0
        self.fitness = 0
        self.rewards = 0
        self.hunger = 250
        self.state = None
        self.selected = False

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
        self._init_vectors()

    def _init_vectors(self):
        self.vis_from_0 = self.vis_intervals[0]
        self.snd_from_0 = self.snd_intervals[0]
        self.mov_from_0 = self.mov_degrees[0]

    #Define the model in the SmartObjNN class:
    def _init_brain(self, fdbk_in, timesteps):
        self.brain = SmartObjNN(self.vis_fidelity, self.snd_fidelity, fdbk_in, timesteps, self.mov_degrees)

    #Must be called
    def _init_in_out(self):
        self.vis_array = np.zeros((1, self.brain.timesteps, self.brain.vis_num), dtype='float32')
        self.snd_array = np.zeros((1, self.brain.timesteps, self.brain.snd_num), dtype='float32')
        #self.fdbk_array = np.zeros((1, self.brain.timesteps, self.brain.out_num), dtype='float32')

        self.new_sensor = np.copy(self.vis_array[0][-1])
        self.out_array = self.brain.call([self.vis_array, self.snd_array])# self.fdbk_array])

    def _rand_in_out(self):
        self.vis_array = np.random.rand(1, self.brain.timesteps, self.brain.vis_num).astype('float32')
        self.snd_array = np.random.rand(1, self.brain.timesteps, self.brain.snd_num).astype('float32')
        #self.fdbk_array = np.random.rand(1, self.brain.timesteps, self.brain.out_num).astype('float32')

        self.new_sensor = np.copy(self.vis_array[0][-1])
        self.out_array = self.brain.call([self.vis_array, self.snd_array])# self.fdbk_array])

    def debug_in_out(self):
        print(self.vis_array)
        print(self.snd_array)
        #print(self.fdbk_array)

        print(self.out_array)

    def handle(self, dt, shapes):
        if self.time > 1.0:
            if self.state is not None:
                state = self.state
                next_state = [np.copy(self.vis_array), np.copy(self.snd_array)]
                reward = int(self.rewards)
                print(self.out_array)
                action = np.copy(self.out_array)
                if self.hunger < 0:
                    done = True
                else:
                    done = False
                reward = reward if not done else -10

                self.brain.memory.remember(state, action, reward, next_state)
                if done:
                    self.reset()
                    return None
                if len(self.brain.memory.container) > 32:
                    self.brain.train(32)

                self.rewards = 0
                self.handle_output()
                self.time = 0.0
            else:
                self.state = [np.copy(self.vis_array), np.copy(self.snd_array)]
                self.handle_output()
                self.time = 0.0

        self.handle_body()
        self.handle_input(shapes)
        self.update_optimizer(dt)
        self.action()

    def handle_body(self):
        self.update_vectors()

    def handle_input(self, shapes):
        '''
        Main input handler, check if the input here changes and then remember what happened if it does.
        '''
        self.prev_state = [np.copy(self.vis_array), np.copy(self.snd_array)]#, np.copy(self.fdbk_array)]
        nearby = [x.body.parent for x in shapes if hasattr(x.body, 'parent')]
        other = [x for x in shapes if not hasattr(x.body, 'parent')]

        self.update_timestep()
        state_change = self.update_sensory(nearby, other)

        if False:
            self.prev_action = self.out_array

            self.state = [np.copy(self.vis_array), np.copy(self.snd_array)]#, np.copy(self.fdbk_array)]

            if self.rewards != 0:
                self.brain.memory.remember(self.prev_state, self.prev_action, self.rewards, self.state)
                self.brain.train(32)
                self.rewards = 0
            else:
                self.brain.memory.remember(self.prev_state, self.prev_action, self.rewards, self.state)
                

    def handle_output(self):
        if np.random.rand() <= self.brain.epsilon:
            self.out_array = [[np.random.rand(2,).astype(np.float32)],[np.random.rand(3,).astype(np.float32)]]
        else:
            speed, turning = self.brain.call([self.vis_array, self.snd_array])# self.fdbk_array])
            self.out_array = [[speed[0]], [turning[0]]]

    def update_timestep(self):
        for i in range(1, self.brain.timesteps):
            self.vis_array[0][i-1] = self.vis_array[0][i]
            self.snd_array[0][i-1] = self.snd_array[0][i]

        self.vis_array[0][-1] = np.zeros(self.brain.vis_num)
        self.snd_array[0][-1] = np.zeros(self.brain.snd_num)

    def update_sensory(self, nearby, other):
        self.prev_sensor = np.copy(self.new_sensor)

        self.update_sensory_main(nearby)
        self.update_sensory_aux(other)

        #Finalize
        self.vis_array[0][-1] = self.vis_array[0][-1]/255

        self.new_sensor = np.copy(self.vis_array[0][-1])

        if not np.array_equal(self.new_sensor, self.prev_sensor):
            return (self.new_sensor, self.prev_sensor)
        else:
            return False

    def update_sensory_main(self, nearby):
        for body in nearby:
            if body is self:
                continue
            pnt_vec = body.body.position - self.body.position
            vec_len = np.linalg.norm(pnt_vec)

            #Vision first
            if vec_len < self.vis_len:
                deg = funcs.real_angle(pnt_vec)
                for i in range(0, len(self.vis_intervals) - 1):
                    a, b = self.vis_intervals[i], self.vis_intervals[i+1]
                    if (deg > a and deg < b) or (b < a and deg < b):
                        self.vis_array[0][-1][i*3] = min(self.vis_array[0][-1][i*3] + body.colour[0], 255)
                        self.vis_array[0][-1][i*3+1] = min(self.vis_array[0][-1][i*3+1] + body.colour[1], 255)
                        self.vis_array[0][-1][i*3+2] = min(self.vis_array[0][-1][i*3+2] + body.colour[2], 255)

            #Sound next
            if vec_len < self.snd_len:
                deg = funcs.real_angle(pnt_vec)
                for i in range(0, len(self.snd_intervals) - 1):
                    a, b = self.snd_intervals[i], self.snd_intervals[i+1]
                    if (deg > a and deg < b) or (b < a and deg < b):
                        self.snd_array[0][-1][i] = body.sound

        #Finalize
        #moved to update_senosry
        #self.snd_array requires no finalizing

    def update_sensory_aux(self, shapes):
        p = self.body.position
        for shape in shapes:
            for i in range(0, len(self.vis_vectors) - 1):
                end = [x for x in (p + self.vis_len*self.vis_vectors[i])]
                if shape.segment_query(p, end, radius=1).shape:
                    self.vis_array[0][-1][i*3] = min(self.vis_array[0][-1][i*3] + 255, 255)
                    '''
                    self.vis_array[0][-1][i*3+1] = min(self.vis_array[0][-1][i*3+1] + 128, 255)
                    self.vis_array[0][-1][i*3+2] = min(self.vis_array[0][-1][i*3+2] + 128, 255)
                    '''

    def rebuild_vectors(self, intervals):
        vectors = []
        return vectors

    def update_vectors(self):
        angle = funcs.until_360(math.degrees(self.body.angle))
        self.vis_intervals[0] = funcs.keep_360(angle + self.vis_from_0)
        self.snd_intervals[0] = funcs.keep_360(angle + self.snd_from_0)

        self.mov_degrees[0] = funcs.keep_360(angle + self.mov_from_0)

        for i in range(1, len(self.vis_intervals)):
            self.vis_intervals[i] =  funcs.keep_360(self.vis_intervals[i-1] + self.vis_slice)
        for i in range(1, len(self.snd_intervals)):
            self.snd_intervals[i] =  funcs.keep_360(self.snd_intervals[i-1] + self.snd_slice)

        for i in range(1, len(self.mov_degrees)):
            self.mov_degrees[i] =  funcs.keep_360(self.mov_degrees[i-1] + self.mov_slice)

        self.vis_vectors.clear()
        for angle in self.vis_intervals:
            self.vis_vectors.append(funcs.to_vector(math.radians(angle)))

        self.snd_vectors.clear()
        for angle in self.snd_intervals:
            self.snd_vectors.append(funcs.to_vector(math.radians(angle)))


        self.mov_vectors.clear()
        for angle in self.mov_degrees:
            self.mov_vectors.append(funcs.to_vector(math.radians(angle)))


    def update_feedback(self):
        for i in range(1, self.brain.timesteps):
            self.fdbk_array[0][i-1] = self.fdbk_array[0][i]


        self.fdbk_array[0][-1] = np.concatenate([self.out_array[0][0], self.out_array[1][0], self.out_array[2][0]]).ravel()

    def reset(self):
        size = self.body.space.parent.size 
        x = random.randint(20, size[0] - 20)
        y = random.randint(20, size[1] - 20)
        self.set_position(x, y)
        self.state = None
        self.hunger = 1000
        self.rewards = 0

    def update_optimizer(self, dt):
        self.time += dt
        self.hunger -= 1

    def get_output(self):
        return self.out_array

    def get_input(self):
        return self.vis_array, self.snd_array #, self.fdbk_array

    def get_weights(self):
        return self.brain.model.get_weights()

    def toggle_selection(self):
        if self.selected:
            self.selected = False
        else:
            self.selected = True

    #Physical Methods
    def set_capabilities(self, max_thrust, max_torque, nrg_efficiency):
        self.max_thrust = max_thrust
        self.max_torque = max_torque
        self.nrg_efficiency = nrg_efficiency #internal variable

    def action(self):
        self.apply_force(self.out_array[0][0])
        self.apply_torque(self.out_array[1][0])

    def apply_force(self, outputs):
        tuples = list(zip(*[iter(outputs)] * 2))
        pairs = list(zip(tuples, self.mov_vectors))

        for tuple, vector in pairs:
            imax = np.argmax(tuple)
            
            if imax == 0:
                thrust = 0.7
            else:
                thrust = 0.3

            self.body.force = -thrust*self.max_thrust*vector

    def apply_torque(self, outputs):
        imax = np.argmax(outputs)
        
        if imax == 0:
            torque = -0.5
        elif imax == 1:
            torque = 0
        elif imax == 2:
            torque = 0.5
        
        self.body.torque = torque*self.max_torque

    def consume(self, other):
        if other.consumable:
            self.rewards += 1*op.timing_f(self.time)
            self.time = 1.01
            self.hunger = 250

    #Display Methods
    def display(self, screen):
        p = super().display(screen)
        dim_colour = funcs.dim_color(self.colour, 25)

        for index, vector in enumerate(self.vis_vectors):
            vector = funcs.flipy(vector)
            end = [int(x) for x in (p + self.vis_len*vector)]
            interval = self.vis_array[0][-1][index*3:index*3+3]

            if interval.any():
                colour = [x*255 for x in interval]
                pg.draw.line(screen, colour, p, end)
            else:
                pg.draw.line(screen, dim_colour, p, end)

        for index, vector in enumerate(self.mov_vectors):
            vector = funcs.flipy(vector)
            end = [int(x) for x in (p + self.shape.radius*vector*2)]
            pg.draw.line(screen, dim_colour, p, end)

        end = [int(x) for x in (p + self.body.rotation_vector*self.shape.radius)]
        pg.draw.line(screen, dim_colour, p, end)

        if self.selected:
            pg.draw.circle(screen, (0,255,255), p, int(self.shape.radius) + 2, 1)
