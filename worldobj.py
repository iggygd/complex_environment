import pymunk as pm
import pygame as pg
import pymunk.pygame_util as pm_pg_util

class WorldBody(pm.Body):
    def __init__(self, parent, mass=0, moment=0, body_type=pm.Body.DYNAMIC):
        super().__init__(mass, moment, body_type)
        self.parent = parent
        self.drag = 0

class WorldObj():
    def __init__(self):
        self.id = 0
        self.colour = (255,255,255)
        self.sound = 0
        self.smart = False

    def _init_body(self, mass, radius):
        self.moment = pm.moment_for_circle(mass, 0, radius)
        self.body = WorldBody(self, mass, self.moment)
        self.shape = pm.Circle(self.body, radius)

    def set_position(self, x, y):
        self.body.position = x, y

    def get_body(self):
        return self.body

    def display(self, screen):
        p = pm_pg_util.to_pygame(self.body.position, screen)
        pg.draw.circle(screen, self.colour, p, int(self.shape.radius), 2)
        return p
