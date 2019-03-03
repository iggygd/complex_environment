import pymunk as pm
import optimizer as op
import random

class A:
    def begin(arbiter, space, data):
        '''
        Able to add callback here later
        '''
        a, b = arbiter.shapes
        c = pm.shapes.Circle #have to fix later

        if isinstance(a, c) and isinstance(b,c):
            x = random.randint(20, space.parent.size[0] - 20)
            y = random.randint(20, space.parent.size[1] - 20)

            if a.body.parent.consumable in b.body.parent.consumes:
                b.body.parent.consume(a.body.parent)

                space.remove(a, a.body)
                space.parent.add_body_at_position(x, y, a.body.parent.name)

                #op.evolve(space, b.body.parent, 1)
                return False
            elif b.body.parent.consumable in a.body.parent.consumes:
                a.body.parent.consume(b.body.parent)

                space.parent.add_body_at_position(x, y, b.body.parent.name)
                space.remove(b, b.body)

                #op.evolve(space, a.body.parent, 1)
                return False

        return True

    def pre_solve(arbiter, space, data):
        return True

    def post_solve(arbiter, space, data):
        pass

    def separate(arbiter, space, data):
        pass
