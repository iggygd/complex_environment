import kerasobj as ko
import world

#Relevant world methods
#set_sbody_body(self, mass, radius):
#set_sbody_characteristics(self, vis_intervals, snd_intervals, mov_degrees, vis_len = 20, snd_len = 40):
#set_sbody_brain(self, fdbk_in, timesteps):

DEFAULT_MASS = 1
DEFAULT_RADIUS = 7
DEFAULT_VIS = [20,55,90,125,160]
DEFAULT_SND = [0,90,180,270]
DEFAULT_MOV = [270]
DEFAULT_VIS_LEN = 50
DEFAULT_SND_LEN = 50
DEFAULT_FDBK = 4
DEFAULT_TIMESTEPS = 4

DEFAULT_MAX_THRUST = 50
DEFAULT_MAX_TORQUE = 10
DEFAULT_NRG_EFF = 0

theUniverse = world.GraphicWorld(720, 480, (0,0), 0)
theUniverse.set_sbody_body(DEFAULT_MASS, DEFAULT_RADIUS)
theUniverse.set_sbody_characteristics(DEFAULT_VIS, DEFAULT_SND, DEFAULT_MOV, DEFAULT_VIS_LEN, DEFAULT_SND_LEN)
theUniverse.set_sbody_brain(DEFAULT_FDBK, DEFAULT_TIMESTEPS)
theUniverse.set_sbody_capabilities(DEFAULT_MAX_THRUST, DEFAULT_MAX_TORQUE, DEFAULT_NRG_EFF)

for i in range(0,10):
    theUniverse.add_sbody_at_position(300+i*5, 300+i*5)

theUniverse.run()
