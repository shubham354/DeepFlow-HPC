import numpy as np
from time import time
import multiprocessing as mp
from functools import partial
from numba import njit
import sys

def diffuse(field, prev_field, dt, viscosity, width, height):
    a = dt * viscosity * width * height
    result = field.copy()
    for k in range(20):
        result[1:-1, 1:-1] = (prev_field[1:-1, 1:-1] +
                            a * (result[2:, 1:-1] + result[:-2, 1:-1] +
                                 result[1:-1, 2:] + result[1:-1, :-2])) / (1 + 4*a)
        result[0, :] = result[-1, :] = result[:, 0] = result[:, -1] = 0
    return result

def project(u, v, p, height, width):
        """Project the velocity field to be mass-conserving"""
        # Calculate divergence
        div = np.zeros((height, width))
        div[1:-1, 1:-1] = (
            (u[1:-1, 2:] - u[1:-1, :-2]) / 2 +
            (v[2:, 1:-1] - v[:-2, 1:-1]) / 2
        )

        # Solve pressure Poisson equation
        p *= 0
        for k in range(20):
            p[1:-1, 1:-1] = (
                (p[2:, 1:-1] + p[:-2, 1:-1] +
                 p[1:-1, 2:] + p[1:-1, :-2] - div[1:-1, 1:-1]) / 4
            )
            p[0, :] = p[-1, :] = p[:, 0] = p[:, -1] = 0

        # Subtract pressure gradient
        u[1:-1, 1:-1] -= (p[1:-1, 2:] - p[1:-1, :-2]) / 2
        v[1:-1, 1:-1] -= (p[2:, 1:-1] - p[:-2, 1:-1]) / 2
        return u, v

def advect(u, v, dye, width, height):
    """Advect velocity and dye fields"""
    prev_u = u.copy()
    prev_v = v.copy()
    prev_dye = dye.copy()

    dt0 = dt * width
    for i in range(1, width-1):
        for j in range(1, height-1):
            x = i - dt0 * prev_u[j, i]
            y = j - dt0 * prev_v[j, i]

            x = max(0.5, min(width-1.5, x))
            y = max(0.5, min(height-1.5, y))

            i0, j0 = int(x), int(y)
            s1, t1 = x - i0, y - j0
            s0, t0 = 1 - s1, 1 - t1

            u[j, i] = (
                s0 * (t0 * prev_u[j0, i0] + t1 * prev_u[j0+1, i0]) +
                s1 * (t0 * prev_u[j0, i0+1] + t1 * prev_u[j0+1, i0+1])
            )
            v[j, i] = (
                s0 * (t0 * prev_v[j0, i0] + t1 * prev_v[j0+1, i0]) +
                s1 * (t0 * prev_v[j0, i0+1] + t1 * prev_v[j0+1, i0+1])
            )
            dye[j, i] = (
                s0 * (t0 * prev_dye[j0, i0] + t1 * prev_dye[j0+1, i0]) +
                s1 * (t0 * prev_dye[j0, i0+1] + t1 * prev_dye[j0+1, i0+1])
            )
    return u, v, dye

dt = 0.1
viscosity = 0.000001
num_steps = 100

width = height = 800
u = np.zeros((height, width), dtype=np.float64)
v = np.zeros((height, width), dtype=np.float64)
p = np.zeros((height, width), dtype=np.float64)
dye = np.zeros((height, width), dtype=np.float64)
    
"""Set up initial conditions with a central vortex"""
center_x, center_y = width//2, height//2
radius = 10
strength = 2.0

for i in range(width):
    for j in range(height):
        dx = i - center_x
        dy = j - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < radius:
            # check correctness i, j
            u[j, i] += -dy/radius * strength
            v[j, i] += dx/radius * strength

dye[center_y-5:center_y+5, center_x-5:center_x+5] = 1

start_time = time()

for _ in range(num_steps):
    u = diffuse(u, u.copy(), dt, viscosity, width, height)
    v = diffuse(v, v.copy(), dt, viscosity, width, height)
    u, v = project(u, v, p, height, width)
    u, v, dye = advect(u, v, dye, width, height)

end_time = time()
