import pymunk as pm

class WorldBody(pm.Body):
    def __init__(self, parent, mass=0, moment=0, body_type=pm.Body.DYNAMIC):
        super().__init__(mass, moment, body_type)
        self.parent = parent

class WorldObj():
    def __init__(self):
        self.id = 0
        self.colour = (0,0,0)
        self.sound = 0

    def _init_body(self, mass, radius):
        self.moment = pm.moment_for_circle(mass, 0, radius)
        self.body = WorldBody(self, mass, self.moment)
        self.shape = pm.Circle(self.body, radius)

    def set_position(self, x, y):
        self.body.position = x, y

    def get_body(self):
        return self.body
