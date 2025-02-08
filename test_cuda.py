import numpy as np
import time
from numba import cuda

# CUDA Kernel for Diffusion
@cuda.jit
def diffuse_kernel(field, prev_field, a, ny, nx):
    i, j = cuda.grid(2)  # 2D thread indexing

    if i >= 1 and i < ny - 1 and j >= 1 and j < nx - 1:
        denom = 1.0 + 4 * a
        bottom = prev_field[i + 1, j]
        top = prev_field[i - 1, j]
        right = prev_field[i, j + 1]
        left = prev_field[i, j - 1]
        temp_sum = bottom + top + right + left

        field[i, j] = (prev_field[i, j] + a * temp_sum) / denom

    # Boundary conditions
    if i == 0 or i == ny - 1 or j == 0 or j == nx - 1:
        field[i, j] = 0.0

# CUDA Kernel for Advection (Simple Upwind Scheme)
@cuda.jit
def advection_kernel(field, velocity_x, velocity_y, dt, ny, nx):
    i, j = cuda.grid(2)  # 2D thread indexing

    if i >= 1 and i < ny - 1 and j >= 1 and j < nx - 1:
        # Simple upwind advection scheme
        dx = int(velocity_x[i, j] * dt)
        dy = int(velocity_y[i, j] * dt)

        prev_i = max(min(i - dy, ny - 1), 0)
        prev_j = max(min(j - dx, nx - 1), 0)

        field[i, j] = field[prev_i, prev_j]

# CUDA Kernel for Projection (Poisson Solver)
@cuda.jit
def projection_kernel(velocity_x, velocity_y, divergence, pressure, ny, nx):
    i, j = cuda.grid(2)  # 2D thread indexing

    if i >= 1 and i < ny - 1 and j >= 1 and j < nx - 1:
        # Compute divergence of the velocity field
        divergence[i, j] = (velocity_x[i + 1, j] - velocity_x[i - 1, j] + 
                            velocity_y[i, j + 1] - velocity_y[i, j - 1]) / 2.0
        
        # Solve Poisson equation for pressure field (using a simple iterative method)
        pressure[i, j] = (divergence[i, j] +
                           (pressure[i + 1, j] + pressure[i - 1, j] +
                            pressure[i, j + 1] + pressure[i, j - 1]) / 4.0)

# Main function that integrates diffusion, advection, and projection
def diffuse_partitioned(field, prev_field, velocity_x, velocity_y, a, dt, iterations=20):
    # Allocate memory on GPU
    d_field = cuda.to_device(field)
    d_prev_field = cuda.to_device(prev_field)
    d_velocity_x = cuda.to_device(velocity_x)
    d_velocity_y = cuda.to_device(velocity_y)
    
    # For projection step
    d_divergence = cuda.device_array_like(field)
    d_pressure = cuda.device_array_like(field)

    # Define grid and block size
    threads_per_block = (16, 16)
    blocks_per_grid = ((ny + threads_per_block[0] - 1) // threads_per_block[0],
                       (nx + threads_per_block[1] - 1) // threads_per_block[1])

    # Main computation loop
    for _ in range(iterations):
        # Diffusion step
        diffuse_kernel[blocks_per_grid, threads_per_block](d_field, d_prev_field, a, ny, nx)
        
        # Advection step
        advection_kernel[blocks_per_grid, threads_per_block](d_field, d_velocity_x, d_velocity_y, dt, ny, nx)
        
        # Projection step (solving Poisson equation for incompressible flow)
        projection_kernel[blocks_per_grid, threads_per_block](d_velocity_x, d_velocity_y, d_divergence, d_pressure, ny, nx)
        
        # Swap the field and prev_field for next iteration
        d_prev_field, d_field = d_field, d_prev_field

    cuda.synchronize()
    return d_field.copy_to_host()  # Copy result back to CPU

if __name__ == "__main__":
    nx, ny = 800,800
    field = np.zeros((ny, nx), dtype=np.float32)
    prev_field = np.random.rand(ny, nx).astype(np.float32)
    velocity_x = np.random.rand(ny, nx).astype(np.float32)  # Velocity field in x-direction
    velocity_y = np.random.rand(ny, nx).astype(np.float32)  # Velocity field in y-direction
    a = 0.1  # Diffusion coefficient
    dt = 0.1  # Time step
    iterations = 20

    start_time = time.time()
    field_updated_serial = diffuse_partitioned(field, prev_field, velocity_x, velocity_y, a, dt, iterations)
    serial_time = time.time() - start_time
    print(f"Processing Time: {serial_time:.4f} seconds")
