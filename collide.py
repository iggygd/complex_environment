class A:
    def begin(arbiter, space, data):
        print(arbiter.shapes)
        return True

    def pre_solve(arbiter, space, data):
        return True

    def post_solve(arbiter, space, data):
        pass
        
    def separate(arbiter, space, data):
        pass
