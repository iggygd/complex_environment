import pymunk as pm
import pygame as pg
import pymunk.pygame_util as pm_pg_util

class WorldBody(pm.Body):
    def __init__(self, parent, mass=0, moment=0, body_type=pm.Body.DYNAMIC):
        super().__init__(mass, moment, body_type)
        self.parent = parent

class WorldObj():
    def __init__(self):
        self.id = 0
        self.colour = (255,255,255)
        self.sound = 0
        self.drag = None

        self.smart = False
        self.consumes = None
        self.consumable = False

        self.name = None

    def _init_body(self, mass, radius, drag, colour, name):
        self.moment = pm.moment_for_circle(mass, 0, radius)
        self.body = WorldBody(self, mass, self.moment)
        self.shape = pm.Circle(self.body, radius)
        self.drag = drag
        self.colour = colour
        self.name = name

    def _init_consumption(self, consumes, consumable):
        self.consumes = consumes
        self.consumable = consumable

    def set_position(self, x, y):
        self.body.position = x, y

    def get_body(self):
        return self.body

    def display(self, screen):
        p = pm_pg_util.to_pygame(self.body.position, screen)
        pg.draw.circle(screen, self.colour, p, int(self.shape.radius), 2)
        return p
