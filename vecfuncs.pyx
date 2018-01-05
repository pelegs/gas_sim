import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt
from cpython cimport bool 
#cython: boundscheck=False, wraparound=False, nonecheck=False

cdef double c_dot(np.ndarray[double, ndim=1] v1,
                  np.ndarray[double, ndim=1] v2):
    return v1[0]*v2[0] + v1[1]*v2[1]

def dot(v1, v2):
    return c_dot(v1, v2)

cdef double c_norm(np.ndarray[double, ndim=1] vec):
    return sqrt(c_dot(vec, vec))

def norm(vec):
    return c_norm(vec)

cdef np.ndarray[double, ndim=1] c_normalize(np.ndarray[double, ndim=1] vec):
    cdef double x = vec[0]
    cdef double y = vec[1]
    cdef double L = c_norm(vec)
    cdef np.ndarray[double, ndim=1] norm_vec = np.array([x/L, y/L])
    return norm_vec

def normalize(vec):
    return c_normalize(vec)

cdef np.ndarray[double, ndim=1] c_vec_mag(np.ndarray[double, ndim=1] vec,
                                          double mag):
    cdef np.ndarray[double, ndim=1] vec_out = c_normalize(vec)
    vec_out[0] = mag * vec_out[0]
    vec_out[1] = mag * vec_out[1]
    return vec_out 

def vec_mag(vec, mag):
    return c_vec_mag(vec, mag)

cdef double c_distance(np.ndarray[double, ndim=1] v1,
                       np.ndarray[double, ndim=1] v2):
    return c_norm(v2-v1)

def distance(v1, v2):
    return c_distance(v1, v2)

cdef np.ndarray[double, ndim=1] c_norm_vec(np.ndarray[double, ndim=1] vec):
    cdef double x = -vec[1]
    cdef double y =  vec[0]
    cdef np.ndarray[double, ndim=1] perp = np.array([x, y])
    cdef np.ndarray[double, ndim=1] norm = c_normalize(perp)
    return norm

def norm_vec(vec):
    return c_norm_vec(vec)

cdef c_ccw(np.ndarray[double, ndim=1] a,
           np.ndarray[double, ndim=1] b,
           np.ndarray[double, ndim=1] c):
    cdef double x = (c[1] - a[1]) * (b[0] - a[0])
    cdef double y = (b[1] - a[1]) * (c[0] - a[0])
    return x > y

def ccw(a, b, c):
    return c_ccw(a, b, c)

cdef bool c_intersect(np.ndarray[double, ndim=1] l1s,
                      np.ndarray[double, ndim=1] l1e,
                      np.ndarray[double, ndim=1] l2s,
                      np.ndarray[double, ndim=1] l2e):
    cdef bool c1 = ccw(l1s, l2s, l2e) != ccw(l1e, l2s, l2e)
    cdef bool c2 = ccw(l1s, l1e, l2s) != ccw(l1s, l1e, l2e)
    return c1 and c2

def intersect(l1_s, l1_e, l2_s, l2_e):
    return c_intersect(l1_s, l1_e,
                       l2_s, l2_e)

cdef np.ndarray[double, ndim=1] c_intersection_point(np.ndarray[double, ndim=1] l1s,
                                                     np.ndarray[double, ndim=1] l1e,
                                                     np.ndarray[double, ndim=1] l2s,
                                                     np.ndarray[double, ndim=1] l2e):
    cdef double x1 = l1s[0]
    cdef double y1 = l1s[1]
    cdef double x2 = l1e[0]
    cdef double y2 = l1e[1]
    cdef double x3 = l2s[0]
    cdef double y3 = l2s[1]
    cdef double x4 = l2e[0]
    cdef double y4 = l2e[1]
    cdef double a = ((y3-y4) * (x1-x3) + (x4-x3) * (y1-y3))
    cdef double b = ((x4-x3) * (y1-y2) - (x1-x2) * (y4-y3))
    cdef double t = a/b
    return l1s + t * (l1e - l1s)

def intersection_point(l1_s, l1_e, l2_s, l2_e):
    return c_intersection_point(l1_s, l1_e,
                                l2_s, l2_e)

cdef np.ndarray[double, ndim=1] c_reflect(np.ndarray[double, ndim=1] d,
                                          np.ndarray[double, ndim=1] b):
    cdef np.ndarray[double, ndim=1] n = c_norm_vec(b)
    return d - 2 * c_dot(d, n) * n

def reflect(vec, base):
    return c_reflect(vec, base)

def neighbor_indices(x, y, N):
    return [(i,j)
            for i in range(x-1, x+2)
            for j in range(y-1, y+2)
            if 0 <= i < N and 0 <= j < N]
