import numpy as np
cimport numpy as np
from vecfuncs import *
from libc.math cimport exp, sqrt
#cython: boundscheck=False, wraparound=False, nonecheck=False

cdef np.ndarray[double, ndim=2] c_ball_collision(np.ndarray[double, ndim=1] b1_pos,
                                                 np.ndarray[double, ndim=1] b2_pos,
                                                 np.ndarray[double, ndim=1] b1_vel,
                                                 np.ndarray[double, ndim=1] b2_vel,
                                                 double b1_r,
                                                 double b2_r,
                                                 double b1_m,
                                                 double b2_m):
    cdef np.ndarray[double, ndim=1] dr = b2_pos - b1_pos
    cdef np.ndarray[double, ndim=1] dv = b2_vel - b1_vel
    cdef double M = b1_m + b2_m
    cdef dist = distance(b1_pos, b2_pos)
    cdef double overlap = b1_r + b2_r - dist
    if overlap > 0.0:
        b2_pos += normalize(dr) * overlap
    cdef np.ndarray[double, ndim=2] new_vels = np.zeros((2,2))
    new_vels[0] =  b1_vel + 2*b2_m/M  * dot(dv, dr)/dot(dr, dr) * dr
    new_vels[1] =  b2_vel - 2*b1_m/M  * dot(dv, dr)/dot(dr, dr) * dr
    return new_vels

def ball_collision(b1, b2):
    new_vels = np.zeros((2,2))
    new_vels = c_ball_collision(b1.pos,
                                b2.pos,
                                b1.vel,
                                b2.vel,
                                b1.rad,
                                b2.rad,
                                b1.mass,
                                b2.mass)
    b1.vel = new_vels[0]
    b2.vel = new_vels[1]
    
    if b2 in b1.neighbors:
        b1.neighbors.remove(b2)
    if b1 in b2.neighbors:
        b2.neighbors.remove(b1)

cdef double c_time_ball_collision(np.ndarray[double, ndim=1] b1_pos,
                                  np.ndarray[double, ndim=1] b2_pos,
                                  np.ndarray[double, ndim=1] b1_vel,
                                  np.ndarray[double, ndim=1] b2_vel,
                                  double b1_r,
                                  double b2_r):
    cdef np.ndarray[double, ndim=1] dr = b2_pos - b1_pos
    cdef np.ndarray[double, ndim=1] dv = b2_vel - b1_vel
    cdef double dr2 = dot(dr, dr)
    cdef double dv2 = dot(dv, dv)
    cdef double dvr = dot(dv, dr)
    cdef double sig = b1_r + b2_r
    cdef double d   = dvr**2 - dv2 * (dr2 - sig**2)
    
    if dvr >= 0.0 or d < 0.0:
        return float('inf')
    else:
        return -1 * (dvr + sqrt(d)) / dv2

def time_ball_collision(b1, b2):
    return c_time_ball_collision(b1.pos,
                                 b2.pos,
                                 b1.vel,
                                 b2.vel,
                                 b1.rad,
                                 b2.rad)

cdef double c_time_wall_collision(np.ndarray[double, ndim=1] ws,
                                  np.ndarray[double, ndim=1] we,
                                  np.ndarray[double, ndim=1] b_pos,
                                  np.ndarray[double, ndim=1] b_vel,
                                  double b_rad):
    cdef np.ndarray[double, ndim=1] future_pos = b_pos + vec_mag(b_vel, 1E10)
    cdef np.ndarray[double, ndim=1] p = intersection_point(b_pos, future_pos, ws, we)
    cdef double d = distance(b_pos, p) - b_rad
    cdef double v = norm(b_vel)
    
    if intersect(b_pos, future_pos, ws, we):
        if v != 0.0:
            return d / v
        else:
            return float('inf')
    else:
        return float('inf')

def time_wall_collision(b, w):
    return c_time_wall_collision(w.start, w.end,
                                 b.pos, b.vel, b.rad)
