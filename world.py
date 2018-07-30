import numpy as np
import kerasobj as ko
import sys

import pymunk as pm
import pymunk.pygame_util as pm_pg_util
import pygame as pg

from pygame.locals import *

class World:
    def __init__(self, size_1, size_2, gravity, dt):
        self.size = (size_1, size_2)
        self.border_size = 7
        self.dt = dt

        self.bodies = []
        self.space = pm.Space()
        self.space.gravity = gravity

    def border(self, size):
        border = pm.Body(body_type = pm.Body.STATIC)
        border.position = (0, 0)

        l1 = pm.Segment(border, (0, self.size[1]), (0, 0), size)
        l2 = pm.Segment(border, (0, 0), (self.size[0], 0), size)
        l3 = pm.Segment(border, (self.size[0],0), (self.size[0], self.size[1]), size)
        l4 = pm.Segment(border, (self.size[0], self.size[1]), (0,self.size[1]), size)

        self.space.add(l1, l2, l3, l4) # 3
        self.borders = l1,l2,l3,l4

    def set_sbody_body(self, mass, radius):
        self.bdy_mas = mass
        self.bdy_rad = radius

    def set_sbody_characteristics(self, vis_intervals, snd_intervals, mov_degrees, vis_len = 20, snd_len = 40):
        self.vis_int = vis_intervals
        self.snd_int = snd_intervals
        self.mov_deg = mov_degrees
        self.vis_len = vis_len
        self.snd_len = snd_len

    def set_sbody_brain(self, fdbk_in, timesteps):
        self.fdbk_in = fdbk_in
        self.timstep = timesteps

    def add_sbody_at_position(self, x, y):
        body = ko.SmartObj()
        body._init_body(self.bdy_mas, self.bdy_rad)
        body.set_position(x, y)
        body._init_characteristics(self.vis_int, self.snd_int, self.mov_deg, self.vis_len, self.snd_len)
        body._init_brain(self.fdbk_in, self.timstep)
        body._rand_in_out()
        self.space.add(body.body, body.shape)
        self.bodies.append(body)

    def main_loop(self):
        running = True
        while running:
            self.space.step(1/50.0)

    def pre_run(self):
        self.border(self.border_size)


    def update(self):
        pass

class GraphicWorld(World):
    def __init__(self, size_1, size_2, gravity, dt):
        super().__init__(size_1, size_2, gravity, dt)
        pg.init()
        self.screen = pg.display.set_mode((self.size[0], self.size[1]))
        pg.display.set_caption("We did it.")
        self.clock = pg.time.Clock()

        self.debug = True
        self.debug_options = pm_pg_util.DrawOptions(self.screen)

    def draw_bodies(self, screen, body):
        pass

    def run(self):
        self.pre_run()
        running = True

        while running:
            for event in pg.event.get():
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    sys.exit(0)
                elif event.type == MOUSEBUTTONDOWN:
                    pass

            self.screen.fill((0,0,0))
            if self.debug is True:
                self.space.debug_draw(self.debug_options)

            self.space.step(1/50)
            pg.display.flip()
            self.clock.tick(50)
