import world
import json
import random
from pathlib import Path

size_x = 600
size_y = 600
ROOT = Path('./type')
TYPES = sorted(ROOT.glob('*.json'))

def read_json(path):
    with path.open() as file:
        return json.load(file)

def add_bodies(num, space, TYPE):
    for i in range(0, num):
        x = random.randint(20, size_x - 20)
        y = random.randint(20, size_y - 20)
        space.add_body_at_position(x, y, TYPE)

theUniverse = world.GraphicWorld(size_x, size_y, (0,0), 1/50.0)
for TYPEPATH in TYPES:
    theUniverse.load_body_param(read_json(TYPEPATH))
theUniverse.set_space_params(.08, density=1.2754)

add_bodies(15, theUniverse, "HERB")
add_bodies(3, theUniverse, "PRED")
add_bodies(20, theUniverse, "FOOD")
theUniverse.run()
