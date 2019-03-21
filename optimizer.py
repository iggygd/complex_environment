import math
import random

def timing_f(x):
    try:
        return 1/x
    except ZeroDivisionError:
        return 10

def get_lowest_fitness(space, body):
    objs = [x.parent for x in space.bodies if hasattr(x, 'parent') and body.name == x.parent.name]
    lowest = min(objs, key=lambda x: x.fitness)
    return lowest

def evolve(space, body, threshold):
    if body.fitness > threshold:
        weights = body.get_weights()
        lowest = get_lowest_fitness(space, body)

        for shape in lowest.body.shapes:
            space.remove(shape)
        space.remove(lowest.body)


        x = random.randint(20, space.parent.size[0] - 20)
        y = random.randint(20, space.parent.size[1] - 20)
        space.parent.add_body_at_position(x, y, body.name, weights)
        body.fitness -= 1
