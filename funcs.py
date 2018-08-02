import random, math
import numpy as np

def random_color():
    return random.choice([
    (random.randint(128, 255), random.randint(0, 255), random.randint(0, 255)),
    (random.randint(0, 255), random.randint(128, 255), random.randint(0, 255)),
    (random.randint(0, 255), random.randint(0, 255), random.randint(128, 255))
    ])

def dim_color(colour, pct = 25):
    return (int(colour[0]*(pct/100)), int(colour[1]*(pct/100)), int(colour[2]*(pct/100)))

def gray_color(self, num):
    if num > 1:
        num = 1
    elif num < .15:
        num = .15

    return (int(255*num), int(255*num), int(255*num))

#---------------Angles
def real_angle(pnt_vec):
    theta = math.atan(pnt_vec[1]/pnt_vec[0])
    if pnt_vec[0] < 0 and pnt_vec[1] < 0:
        return 180 + math.degrees(theta)
    elif pnt_vec[0] < 0 and pnt_vec[1] > 0:
        return 180 + math.degrees(theta)
    elif pnt_vec[0] > 0 and pnt_vec[1] < 0:
        return 360 + math.degrees(theta)
    else:
        return math.degrees(theta)

def keep_360(ang):
    if ang > 360:
        return ang - 360
    elif ang < 0:
        return ang + 360
    elif ang == 360:
        return 0
    else:
        return ang

def until_360(ang):
    if ang >= 0 and ang < 360:
        return ang
    else:
        return until_360(keep_360(ang))

#------------Vectors
def normalize_vec(vec):
    return vec / np.linalg.norm(vec)

def perpendicular_vec(self, vec, pnt, radius):
    '''
    P->Q = <(Q*i - P*i) + (Q*j - P*j)> = <(Y*i + Y*j)>
    Z = <(Z*i + Z*j)>
    P->Q * Z = 0

    0 = Z*Y*i + Z*Y*j
    -Zi*Yi/Yj = Zj -> (-Zi, Zj)   | Not equal if condition
    -Zj*Yj/Yi = Zi -> (Zi, -Zj)   | is it starts from a point

    Only need to calculate once
    '''
    Zj = -1*pnt[0]*vec[0]/vec[1]
    return [radius*normalize_vec(np.array([pnt[0], Zj])), -1*radius*normalize_vec(np.array([pnt[0], Zj]))]

def rotation(vec, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s,  c)))

    return np.dot(R, vec)

def to_vector(angle):
    return np.array([np.cos(angle), np.sin(angle)])

#------------Others

def to_pygame(p, screen):
    return int(p.x), int(-p.y+screen.get_height())
