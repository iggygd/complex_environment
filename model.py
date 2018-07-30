import kerasmain as km
import numpy as np

obj = km.SmartObj(np.array([0,0]), (255,255,255), 7)
def_vis = [20,55,90,125,160]
def_snd = [0,90,180,270,360]
def_mov = [270]
def_fdbk = 4
def_timesteps = 4
obj._init_characteristics(def_vis, def_snd, def_mov)
obj._init_brain(def_fdbk, 4)
obj._init_in_out()
obj.brain.debug()

for input in obj.get_input():
    print(input)

print(obj.get_output())
