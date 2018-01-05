import numpy as np
import vecfuncs

errors = {
          'normalization': 0,
          'distance'     : 0,
          'norm_vec'     : 0
         }

N = 1000

# Checking vector operations
vs = np.random.uniform(-1, 1, size=(N, 2))
for v in vs:
    # Check normalization
    norm_c  = vecfuncs.normalize(v)
    norm_np = 1/np.linalg.norm(v) * v
    if norm_c.all() != norm_np.all():
        errors['normalization'] += 1

    # Check distance
    for u in vs:
        dis_c  = vecfuncs.distance(v, u)
        dis_np = np.linalg.norm(v-u)
        if dis_c != dis_np:
            errors['distance'] += 1

    # Check normal vectors
    N_vec_c = vecfuncs.norm_vec(v)
    L_np = np.linalg.norm(v)
    normalized_vec_np = v / L_np
    x = -normalized_vec_np[1]
    y =  normalized_vec_np[0]
    N_vec_np = np.array([x, y])
    if N_vec_c.any() != N_vec_np.any():
        errors['norm_vec'] += 1

'''
# Checking line segment operations
    lines = np.random.uniform(-1, 1, size=(N, 8))
    for row in lines:
        l1_start = lines[0:2]
        l1_end   = lines[2:4]
        l2_start = lines[4:6]
        l2_end   = lines[6:8]
        intersect_c  = vecfuncs.intersect(l1, l2) 
        intersect_np = 0
'''

for key, error in errors.items():
    print(key, error)
