import numpy as np
import kerasobj as ko
import funcs, collide
import sys

import pymunk as pm
import pymunk.pygame_util as pm_pg_util
import pygame as pg

from pygame.locals import *

class WorldSpace(pm.Space):
    def __init__(self, world):
        super().__init__()
        self.parent = world

class World:
    def __init__(self, size_1, size_2, gravity, dt):
        self.size = (size_1, size_2)
        self.border_size = 7
        self.dt = dt

        self.params = {}
        self.space = WorldSpace(self)
        self.space.gravity = gravity
        self.collider = self.space.add_default_collision_handler()

        self.collider.begin = collide.A.begin
        self.collider.pre_solve = collide.A.pre_solve
        self.collider.post_solve = collide.A.post_solve
        self.collider.separate = collide.A.separate

    def border(self, size):
        border = pm.Body(body_type = pm.Body.STATIC)
        border.position = (0, 0)

        l1 = pm.Segment(border, (0, self.size[1]), (0, 0), size)
        l2 = pm.Segment(border, (0, 0), (self.size[0], 0), size)
        l3 = pm.Segment(border, (self.size[0],0), (self.size[0], self.size[1]), size)
        l4 = pm.Segment(border, (self.size[0], self.size[1]), (0,self.size[1]), size)

        self.space.add(l1, l2, l3, l4) # 3
        self.borders = l1,l2,l3,l4

    def set_space_params(self, damping, density = 1.1839):
        self.space.damping = damping
        self.density = density

    def load_body_param(self, DICT):
        self.params[DICT['name']] = DICT

    def add_body_at_position(self, x, y, TYPE, weights = None):
        root = self.params[TYPE]

        if root["smart"]:
            body = ko.SmartObj()
        else:
            body = ko.WorldObj()

        pbdy = root['body']
        body._init_body(pbdy['bdy_mas'], pbdy['bdy_rad'], pbdy['drag_co'], pbdy['colour'], TYPE)
        body._init_consumption(root['consumes'], root['consumable'])
        body.set_position(x, y)

        if root["smart"]:
            pchr = root['characteristics']
            pbrn = root['brain']
            pcap = root['capabilities']
            body._init_characteristics(pchr['vis_int'], pchr['snd_int'], pchr['mov_deg'], pchr['vis_len'], pchr['snd_len'])
            body._init_brain(pbrn['fdbk_in'], pbrn['timstep'])
            body._rand_in_out()
            body.set_capabilities(pcap['max_thrust'], pcap['max_torque'], pcap['nrg_efficiency'])
            if weights:
                body.brain.model.set_weights(weights)

        self.space.add(body.body, body.shape)

    def remove_body(self, body):
        pass

    def drag(self, body):
        pass

    def main_loop(self):
        running = True
        while running:
            self.space.step(1/50.0)

    def pre_run(self):
        self.border(self.border_size)

    def update(self):
        for body in self.space.bodies:
            if body.parent.smart:
                p = body.position
                f = max(body.parent.snd_len, body.parent.vis_len)
                bbox = pm.BB(p[0]-f,p[1]-f,p[0]+f,p[1]+f)
                shapes = self.space.bb_query(bbox, pm.ShapeFilter())

                body.parent.handle(self.dt, shapes)
                #body.parent.handle_output()
                #body.parent.handle_body()
                #body.parent.handle_input(shapes)
                #body.parent.update_optimizer(self.dt)
                #body.parent.action()
            #self.drag(body)
        self.space.step(1/50)
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

        self.selected = None

    def draw_object(self, screen, body):
        pass

    def draw_bodies(self, screen, body):
        pass

    def update(self):
        super().update()
        pg.display.flip()
        self.clock.tick(50)

    def save_bodies(self):
        for body in self.space.bodies:
            if body.parent.smart:
                print(f'saved, {str(body.parent)}')
                body.parent.brain.model.save(f'models/{body.parent.fitness}-{id(body.parent)}.h5')

    def display(self):
        for shape in self.space.shapes:
            if isinstance(shape, pm.shapes.Circle):
                if hasattr(shape.body, 'parent'):
                    shape.body.parent.display(self.screen)
                else:
                    p = funcs.to_pygame(shape, self.screen)
                    pg.draw.circle(self.screen, (128,128,128), p, int(shape.radius), 1)

            elif isinstance(shape, pm.shapes.Segment):
                body = shape.body
                p1 = funcs.to_pygame(body.position + shape.a.rotated(body.angle), self.screen)
                p2 = funcs.to_pygame(body.position + shape.b.rotated(body.angle), self.screen)

                pg.draw.lines(self.screen, (255,0,0), False, [p1,p2], int(shape.radius))

    def run(self):
        self.pre_run()
        running = True

        while running:
            for event in pg.event.get():
                if event.type == QUIT:
                    self.save_bodies()
                    sys.exit(0)
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    self.save_bodies()
                    sys.exit(0)
                elif event.type == MOUSEBUTTONDOWN:
                    epos = pm_pg_util.from_pygame(event.pos, self.screen)
                    for body in self.space.bodies:
                        if body.parent.smart:
                            if np.linalg.norm(epos-body.position) < 10:
                                if self.selected:
                                    self.selected.parent.toggle_selection()
                                self.selected = body
                                self.selected.parent.toggle_selection()
                                break
                            else:
                                if self.selected:
                                    self.selected.parent.toggle_selection()
                                self.selected = None

            self.screen.fill((0,0,0))
            if self.selected:
                print(vars(self.selected.parent))
            #if self.debug is True:
            #    self.space.debug_draw(self.debug_options)
            self.display()
            self.update()
