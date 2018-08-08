import world
import json
from pathlib import Path

ROOT = Path('./type')
TYPES = sorted(ROOT.glob('*.json'))

def read_json(path):
    with path.open() as file:
        return json.load(file)

theUniverse = world.GraphicWorld(300, 300, (0,0), 0)
for TYPEPATH in TYPES:
    theUniverse.load_body_param(read_json(TYPEPATH))

for i in range(0,4):
    theUniverse.add_sbody_at_position(55+i*5, 55+i*5, "BODY")
theUniverse.add_sbody_at_position(50, 50, "FOOD")
theUniverse.add_sbody_at_position(25, 25, "BODY")
#theUniverse.add_sbody_at_position(75, 50)
#theUniverse.add_sbody_at_position(75, 125)
#theUniverse.add_sbody_at_position(50, 150)

theUniverse.run()
