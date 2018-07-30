import worldobj as wo
import kerasobj as ko
import pymunk as pm
import pymunk.pygame_util as pm_pg_util
import pygame as pg
import random, sys

from pygame.locals import *

def main():
    pg.init()
    screen = pg.display.set_mode((600, 600))
    pg.display.set_caption("Joints. Just wait and the L will tip over")
    clock = pg.time.Clock()

    space = pm.Space() #2
    space.gravity = (0.0, -900.0)

    lines = add_static_L(space)
    balls = []
    draw_options = pm_pg_util.DrawOptions(screen)

    ticks_to_next_ball = 10
    while True:
        for event in pg.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                sys.exit(0)

        ticks_to_next_ball -= 1
        if ticks_to_next_ball <= 0:
            ticks_to_next_ball = 25
            ball_shape = add_ball(space)
            balls.append(ball_shape)

        space.step(1/50.0)

        screen.fill((255,255,255))
        space.debug_draw(draw_options)

        pg.display.flip()
        clock.tick(50)

def add_static_L(space):
    body = pm.Body(body_type = pm.Body.STATIC) # 1
    body.position = (0, 0)
    l1 = pm.Segment(body, (0, 600), (0, 0), 7) # 2
    l2 = pm.Segment(body, (0, 0), (600, 0), 7)
    l3 = pm.Segment(body, (600,0), (600,600), 7)
    l4 = pm.Segment(body, (600,600), (0,600), 7)

    space.add(l1, l2, l3, l4) # 3
    return l1,l2,l3,l4

def to_pygame(p):
    """Small hack to convert pymunk to pygame coordinates"""
    return int(p.x), int(-p.y+600)

def add_ball(space):
    mass = 1
    radius = 14
    moment = pm.moment_for_circle(mass, 0, radius) # 1
    body = ko.WorldObj(mass, moment) # 2
    x = random.randint(120, 380)
    body.body.position = x, 550 # 3
    shape = pm.Circle(body.body, radius) # 4
    space.add(body.body, shape) # 5
    return shape

if __name__ == '__main__':
    sys.exit(main())
