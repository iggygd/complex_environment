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

CONST_WIDTH = 1
CONST_COEFF = 8
CONST_MAXFORCE = 500
CONST_MAXROT = 50
CONST_ROTCOEFF = 8
FPS_LIMIT = 60
CONST_LEARNING_RATE = 0.05

class WorldObj:
    def __init__(self, pos, colour, size):
        self.id = 0
        self.colour = colour
        self.sound = 0
        self.size = size
        self.mass = 1.0

        self.pos = pos
        self.acc = np.array([0.0])
        self.vel = np.array([0,0])

        self.front = np.array([1,0])
        self.front_end = self.pos + self.size*self.front
        self.ang = 0
        self.ang_vel = 0
        self.ang_acc = 0

        self.bbox = (self.pos[0] - size*3, self.pos[1] - size*3, self.pos[0] + size*3, self.pos[1] + size*3)

    def display(self, screen, style):
        if style == "CIRCLE":
            pg.draw.circle(screen, self.colour, (int(self.pos[0]), int(self.pos[1])), self.size, CONST_WIDTH)
            pg.draw.line(screen, self.colour, (int(self.pos[0]), int(self.pos[1])), (int(self.front_end[0]), int(self.front_end[1])))
        else:
            pg.draw.circle(screen, self.colour, (int(self.pos[0]), int(self.pos[1])), self.size, CONST_WIDTH)

    def update_2d(self, dt):
        self.vel = self.vel + self.acc*dt - CONST_COEFF*self.vel*dt
        self.pos = self.pos + self.vel*dt

    def update_rot(self, dt):
        self.ang_vel = self.ang_vel + self.ang_acc*dt - CONST_ROTCOEFF*self.ang_vel*dt
        self.ang     = self.ang     + self.ang_vel*dt

        if self.ang > 360:
            self.ang = self.ang - 360
        elif self.ang < 0:
            self.ang = self.ang + 360

        self.front = funcs.to_vector(self.ang)
        self.front_end = self.pos + self.size*self.front

    def update_bbox(self):
        self.bbox = (self.pos[0] - size*3, self.pos[1] - size*3, self.pos[0] + size*3, self.pos[1] + size*3)

    def apply_force(self, force):
        self.acc = force/self.mass
        pass

    def apply_torque(self, force, moment):
        self.ang_acc = moment*(force/self.mass)
        pass

    def rotate(self, theta):
        self.front = funcs.rotation(self.front, math.radians(theta))
        pass

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

        self.generation = 0
        self.energy = 0
        self.fitness = 0
        self.thrust = 0

        self.force = 0
        self.forced = 0
        self.torque = 0
        self.torqued = 0

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
            #print(i)

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

        #print(self.fdbk_array[0][-1].shape)
        #print(self.out_array.shape)
        self.fdbk_array[0][-1] = self.out_array

    def get_output(self):
        return self.out_array

    def get_input(self):
        return self.vis_array, self.snd_array, self.fdbk_array

    #Physical Methods
    def action(self, max_force, max_torque):
        self.apply_force(self.action_thrust(max_force))
        self.apply_torque(self.action_rotate(max_torque), self.moment)
        #self.action_memory(self.brain.input, self.brain.output)

    def action_thrust(self, max_force):
        outputs = self.out_array[0][0:len(self.mov_degrees):1]
        force = np.array([0,0])
        pairs = list(zip(outputs, self.mov_vectors))

        for output, vector in pairs:
            force = force - output*max_force*vector

        self.force = force
        self.forced = np.linalg.norm(force)
        return force

    def action_rotate(self, max_force):
        outputs = self.out_array[0][len(self.mov_degrees):len(self.mov_degrees)+2:1]
        force = 0

        force = force + outputs[0]*max_force - outputs[1]*max_force

        self.torque = force
        self.torqued = force
        return force

    def calculate_energy(self, dt, force):
        self.energy = self.energy - force * dt
        pass

    #Inherited Functions
    def display(self, screen, style):
        if style == "CIRCLE":
            dim_colour = funcs.dim_color(self.colour, 25)
            for index, vis_vector in enumerate(self.vis_vectors):
                vis_end = self.pos + self.vis_len*vis_vector
                actual_i = index*3
                #print(index)
                #print(vis_vector)
                #print(self.vis_array[0][-1])
                #print(self.vis_array[0][-1][actual_i])
                #print(len(self.vis_array[0][-1]))
                #print(self.vis_vectors)
                #print(len(self.vis_vectors))
                if index != len(self.vis_vectors) - 1:
                    if (self.vis_array[0][-1][actual_i] > 0.2 or self.vis_array[0][-1][actual_i+1] > 0.2 or self.vis_array[0][-1][actual_i+2] > 0.2):
                        colour = int(255*self.vis_array[0][-1][actual_i]), int(255*self.vis_array[0][-1][actual_i+1]), int(255*self.vis_array[0][-1][actual_i+2])
                        pg.draw.line(screen, colour, (int(self.pos[0]), int(self.pos[1])), (int(vis_end[0]), int(vis_end[1])))
                    else:
                        pg.draw.line(screen, dim_colour, (int(self.pos[0]), int(self.pos[1])), (int(vis_end[0]), int(vis_end[1])))

            dim_colour = funcs.dim_color(self.colour, 50)
            for mov_vector in self.mov_vectors:
                mov_end = self.pos + self.size*mov_vector*2
                pg.draw.line(screen, dim_colour, (int(self.pos[0]), int(self.pos[1])), (int(mov_end[0]), int(mov_end[1])))

            pg.draw.circle(screen, self.colour, (int(self.pos[0]), int(self.pos[1])), self.size, CONST_WIDTH)
            pg.draw.line(screen, self.colour, (int(self.pos[0]), int(self.pos[1])), (int(self.front_end[0]), int(self.front_end[1])))
        else:
            pg.draw.circle(screen, self.colour, (int(self.pos[0]), int(self.pos[1])), self.size, CONST_WIDTH)

    def update_2d(self, dt):
        if self.forced > 0:
            self.calculate_energy(dt, self.forced)
            self.forced = 0
        super().update_2d(dt)

    def update_rot(self, dt):
        if self.torqued > 0:
            self.calculate_energy(dt, self.torqued)
            self.torqued = 0

        self.ang_vel = self.ang_vel + self.ang_acc*dt - CONST_ROTCOEFF*self.ang_vel*dt
        ang_vel_dt   = self.ang_vel*dt
        self.ang     = self.ang     + ang_vel_dt

        self.ang = funcs.keep_360(self.ang)

        self.front = funcs.to_vector(math.radians(self.ang))
        self.front_end = self.pos + self.size*self.front

        ''' #deprecated
        for index, interval in enumerate(self.vis_intervals):
            self.vis_intervals[index] = self.vis_intervals[index] + ang_vel_dt
            if self.vis_intervals[index] > 360:
                self.vis_intervals[index] = self.vis_intervals[index] - 360
            elif self.vis_intervals[index] < 0:
                self.vis_intervals[index] = self.vis_intervals[index] + 360
        '''
        vis_init = funcs.keep_360(self.vis_intervals[0] + ang_vel_dt)
        self.vis_intervals = []

        if self.vis_fidelity > 0:
            for i in range(0, self.vis_fidelity + 1):
                self.vis_intervals.append(vis_init)
                vis_init = funcs.keep_360(vis_init + self.vis_slice)
        else:
            self.vis_intervals.append(vis_init)
        #self.vis_intervals = [vis_interval + self.ang for vis_interval in self.vis_intervals]
        self.vis_vectors = []
        for angle in self.vis_intervals:
            self.vis_vectors.append(funcs.to_vector(math.radians(angle)))

        for index, angle in enumerate(self.mov_degrees):
            self.mov_degrees[index] = funcs.keep_360(self.mov_degrees[index] + ang_vel_dt)

        self.mov_vectors = []
        for angle in self.mov_degrees:
            self.mov_vectors.append(funcs.to_vector(math.radians(angle)))

class DumbObj(WorldObj):
    def __init__(self, position, colour, size):
        super().__init__(position, colour, size)
        self.energy = 0
        self.mass = 0
        pass

    #temporary, delete later
    def food_init(self, poisonous = False, lethal = False):
        self.poisonous = poisonous
        self.lethal = lethal

class World:
    def __init__(self, size_1, size_2, dt):
        self.size = np.array([size_1, size_2])
        self.dt = dt

        self.bodies = []

        #look to change to generalized in self.bodies, may be inefficient
        self.foods = []

        self.body_spindex = qtree.Index(bbox = (0,0, self.size[0], self.size[1]))
        self.food_spindex = qtree.Index(bbox = (0,0, self.size[0], self.size[1]))

    def handle_edge(self):
        for body in self.bodies:
            if body.pos[0] > self.size[0]:
                body.pos[0] = 0
            if body.pos[0] < 0:
                body.pos[0] = self.size[0]
            if body.pos[1] > self.size[1]:
                body.pos[1] = 0
            if body.pos[1] < 0:
                body.pos[1] = self.size[1]

    def gen_body(self, amt, vis, snd, mov, size = 7, timesteps = 4, fdbk_in = 4, vis_len = 20, snd_len = 40):
        for i in range(0, amt):
            position = np.array([random.randint(size + 10, self.size[0]), random.randint(size + 10, self.size[1])])
            body = SmartObj(position, funcs.random_color(), size)
            body._init_characteristics(vis, snd, mov, vis_len, snd_len)
            body._init_brain(fdbk_in, timesteps)
            body._init_in_out()
            self.bodies.append(body)

    def gen_food(self, amt, colour = (255,0,0), size = 7):
        for i in range(0, amt):
            position = np.array([random.randint(size + 10, self.size[0]), random.randint(size + 10, self.size[1])])
            food = DumbObj(position, colour, size)
            self.foods.append(food)

    def gen_food_point(self, x, y, colour = (255,0,0), size = 7):
        position = np.array([x,y])
        food = DumbObj(position, colour, size)
        return food


    def update_backend(self, vis, snd, mov, dt):
        for body in self.bodies:
            overlapbbox = (body.pos[0] - body.snd_len, body.pos[1] - body.snd_len, body.pos[0] + body.snd_len, body.pos[1] + body.snd_len)
            nearby_food = self.food_spindex.intersect(overlapbbox)

            body.handle_input(nearby_food, body.get_output())
            body.update_rot(dt)
            body.update_2d(dt)
            body.handle_output()
            body.action(CONST_MAXFORCE, CONST_MAXROT)

            for food in nearby_food:
                if np.linalg.norm(food.pos-body.pos) < (food.size+body.size):
                    if food in self.foods:
                        self.foods.remove(food)
                        self.food_spindex.remove(food, food.bbox)

                        body.fitness = body.fitness + 1
                        new_food = self.gen_food_point(random.randint(0, self.size[0]), random.randint(0, self.size[1]))
                        del food
                        self.foods.append(new_food)
                        self.food_spindex.insert(new_food, new_food.bbox)
                else:
                    pass

    def rebuild_qtree(self):
        self.body_spindex = qtree.Index(bbox = (0,0, self.size[0], self.size[1]))
        for body in self.bodies:
            self.body_spindex.insert(body, body.bbox)

    def rebuild_foodtree(self):
        self.food_spindex = qtree.Index(bbox = (0,0, self.size[0], self.size[1]))
        for food in self.foods:
            self.food_spindex.insert(food, food.bbox)

    def run(self, vis, snd, mov):
        begin = time.time()
        running = True
        self.rebuild_foodtree()
        self.rebuild_qtree()
        self.select_timer(5.0)
        self.fitness_timer(60.0, vis, mov)
        try:
            while running:
                begin = time.time()

                self.handle_edge()
                self.update_backend(vis, mov, self.dt)
                end = time.time()
                delta_func = end - begin
                #print(delta_func)


        except KeyboardInterrupt:
            print('Interrupted')
            try:
                self.exit_save()
                print(delta_func)
            except:
                raise
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

class GraphicWorld(World):
    def __init__(self, size_1, size_2, dt = 0):
        super().__init__(size_1, size_2, dt)
        pg.init()
        self.clock = pg.time.Clock()
        self.debug = pg.time.set_timer(USEREVENT+1,100)
        self.new_top_fitness = pg.time.set_timer(USEREVENT+2,5000)
        self.reset_fitness = pg.time.set_timer(USEREVENT+3, 60000)

        self.manual_select = False

        self.font = pg.font.SysFont('arial', 9)
        self.uif = pg.font.SysFont('arial', 13)

    def pygame_frontend(self):
        for food in self.foods:
            food.display(self.screen, None)

        for body in self.bodies:
            body.display(self.screen, "CIRCLE")

    def run(self, vis, snd, mov):
        self.screen = pg.display.set_mode((self.size[0], self.size[1]))
        running = True
        self.rebuild_qtree()
        self.rebuild_foodtree()

        while running:
            self.clock.tick(FPS_LIMIT)
            self.handle_edge()

            for event in pg.event.get():

                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()

            self.screen.fill((0,0,0))

            dt = self.clock.get_time()/1000
            self.update_backend(vis, snd, mov, dt)
            self.pygame_frontend()

            pg.display.flip()

#TestObj = SmartObj(np.array([0,0]), (255,255,255), 7)

#def_vis = [20,55,90,125,160]
#def_snd = [0,90,180,270]
#def_mov = [270]
#def_fdbk = 8
#def_timesteps = 4
#TestObj._init_characteristics(def_vis, def_snd, def_mov)
#TestObj._init_brain(len(def_vis), len(def_snd), def_fdbk, 4)
#TestObj.brain.debug()
#TestObj.brain.evolve_layers()
