import pymunk as pm

class A:
    def begin(arbiter, space, data):
        return True

    def pre_solve(arbiter, space, data):
        a, b = arbiter.shapes
        c = pm.shapes.Circle

        return True

    def post_solve(arbiter, space, data):
        pass

    def separate(arbiter, space, data):
        pass
