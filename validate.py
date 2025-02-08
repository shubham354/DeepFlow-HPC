import numpy as np
import time
from numba import cuda
import psutil
import os

# CUDA Constants
BLOCK_SIZE = 32
SHARED_SIZE = BLOCK_SIZE + 2  # Include halo regions

@cuda.jit
def diffuse_kernel(field, prev_field, a, ny, nx):
    shared = cuda.shared.array((SHARED_SIZE, SHARED_SIZE), dtype=np.float32)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    
    # Global indices
    i = by * BLOCK_SIZE + ty
    j = bx * BLOCK_SIZE + tx
    
    # Load data into shared memory including halo
    if i < ny and j < nx:
        shared[ty+1, tx+1] = prev_field[i, j]
        
        # Load halo regions
        if tx == 0 and j > 0:
            shared[ty+1, 0] = prev_field[i, j-1]
        if tx == BLOCK_SIZE-1 and j < nx-1:
            shared[ty+1, BLOCK_SIZE+1] = prev_field[i, j+1]
        if ty == 0 and i > 0:
            shared[0, tx+1] = prev_field[i-1, j]
        if ty == BLOCK_SIZE-1 and i < ny-1:
            shared[BLOCK_SIZE+1, tx+1] = prev_field[i+1, j]
    
    cuda.syncthreads()
    
    if 1 <= i < ny-1 and 1 <= j < nx-1:
        denom = 1.0 + 4.0 * a
        temp_sum = (shared[ty, tx+1] +    # top
                   shared[ty+2, tx+1] +    # bottom
                   shared[ty+1, tx] +      # left
                   shared[ty+1, tx+2])     # right
        
        field[i, j] = (prev_field[i, j] + a * temp_sum) / denom
    
    # Handle boundaries
    if i == 0 or i == ny-1 or j == 0 or j == nx-1:
        if i < ny and j < nx:
            field[i, j] = 0.0

@cuda.jit
def advect_kernel(field, prev_field, velocity_x, velocity_y, dt, ny, nx):
    i, j = cuda.grid(2)
    
    if 1 <= i < ny-1 and 1 <= j < nx-1:
        # Semi-Lagrangian advection
        pos_x = float(j) - velocity_x[i, j] * dt
        pos_y = float(i) - velocity_y[i, j] * dt
        
        # Clamp positions
        pos_x = max(0.5, min(float(nx-2), pos_x))
        pos_y = max(0.5, min(float(ny-2), pos_y))
        
        # Bilinear interpolation
        x0 = int(pos_x)
        y0 = int(pos_y)
        x1 = x0 + 1
        y1 = y0 + 1
        
        fx = pos_x - float(x0)
        fy = pos_y - float(y0)
        
        c00 = prev_field[y0, x0]
        c10 = prev_field[y1, x0]
        c01 = prev_field[y0, x1]
        c11 = prev_field[y1, x1]
        
        field[i, j] = (1.0-fx)*(1.0-fy)*c00 + \
                      fx*(1.0-fy)*c01 + \
                      (1.0-fx)*fy*c10 + \
                      fx*fy*c11

@cuda.jit
def project_kernel(velocity_x, velocity_y, p, div, ny, nx):
    i, j = cuda.grid(2)
    
    if 1 <= i < ny-1 and 1 <= j < nx-1:
        # Compute divergence
        div[i, j] = -0.5 * (
            velocity_x[i, j+1] - velocity_x[i, j-1] +
            velocity_y[i+1, j] - velocity_y[i-1, j]
        )
        p[i, j] = 0.0
    
    cuda.syncthreads()
    
    if 1 <= i < ny-1 and 1 <= j < nx-1:
        # Jacobi iteration
        p[i, j] = (div[i, j] + 
                  p[i, j+1] + p[i, j-1] +
                  p[i+1, j] + p[i-1, j]) / 4.0

class FluidSimulator:
    def __init__(self, nx, ny, dt=0.1, diffusion=0.0001):
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.diffusion = diffusion
        
        # Initialize fields using pinned memory
        self.field = cuda.pinned_array((ny, nx), dtype=np.float32)
        self.field.fill(0.0)
        
        self.velocity_x = cuda.pinned_array((ny, nx), dtype=np.float32)
        self.velocity_y = cuda.pinned_array((ny, nx), dtype=np.float32)
        
        # Initialize with some interesting patterns
        Y, X = np.mgrid[0:ny, 0:nx]
        self.velocity_x[:] = 0.1 * np.sin(2.0 * np.pi * X / nx)
        self.velocity_y[:] = 0.1 * np.sin(2.0 * np.pi * Y / ny)
        
        # Add some density sources
        center_y, center_x = ny//2, nx//2
        radius = min(nx, ny) // 8
        Y, X = np.ogrid[:ny, :nx]
        mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2
        self.field[mask] = 1.0
        
        # Calculate grid configuration
        self.threads_per_block = (BLOCK_SIZE, BLOCK_SIZE)
        self.blocks_per_grid = (
           max(128, (nx + BLOCK_SIZE - 1) // BLOCK_SIZE),
           max(128, (ny + BLOCK_SIZE - 1) // BLOCK_SIZE)
        )

    def step(self, num_steps=1):
        # Create CUDA stream for asynchronous execution
        stream = cuda.stream()
        
        with stream.auto_synchronize():
            # Allocate device arrays
            d_field = cuda.to_device(self.field, stream=stream)
            d_field_temp = cuda.device_array_like(self.field, stream=stream)
            d_velocity_x = cuda.to_device(self.velocity_x, stream=stream)
            d_velocity_y = cuda.to_device(self.velocity_y, stream=stream)
            d_div = cuda.device_array_like(self.field, stream=stream)
            d_p = cuda.device_array_like(self.field, stream=stream)
            
            try:
                for _ in range(num_steps):
                    # Diffusion
                    diffuse_kernel[self.blocks_per_grid, self.threads_per_block, stream](
                        d_field_temp, d_field, self.diffusion, self.ny, self.nx)
                    
                    # Advection
                    advect_kernel[self.blocks_per_grid, self.threads_per_block, stream](
                        d_field, d_field_temp, d_velocity_x, d_velocity_y,
                        self.dt, self.ny, self.nx)
                    
                    # Projection
                    for _ in range(4):  # Jacobi iterations
                        project_kernel[self.blocks_per_grid, self.threads_per_block, stream](
                            d_velocity_x, d_velocity_y, d_p, d_div, self.ny, self.nx)
                    
                    # Swap buffers
                    d_field, d_field_temp = d_field_temp, d_field
                
                # Copy result back to host
                d_field.copy_to_host(self.field, stream=stream)
                
            finally:
                # Clean up device memory
                del d_field
                del d_field_temp
                del d_velocity_x
                del d_velocity_y
                del d_div
                del d_p

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def benchmark_size(size):
    """Benchmark simulation for a specific grid size"""
    memory_before = get_memory_usage()
    start_time = time.time()
    
    try:
        simulator = FluidSimulator(size, size)
        simulator.step(num_steps=20)
        
        execution_time = time.time() - start_time
        memory_after = get_memory_usage()
        memory_used = memory_after - memory_before
        
        return {
            'size': size,
            'cells': size * size,
            'time': execution_time,
            'cells_per_second': (size * size * 20) / execution_time,
            'memory_used_gb': memory_used,
            'success': True,
            'field_sum': np.sum(simulator.field)
        }
    except Exception as e:
        return {
            'size': size,
            'cells': size * size,
            'time': 0,
            'cells_per_second': 0,
            'memory_used_gb': 0,
            'success': False,
            'error': str(e)
        }

def run_benchmark_suite():
    """Run complete benchmark suite"""
    sizes = [256, 512, 1024, 2048, 4096, 8192]
    
    print("\nCUDA Fluid Simulation Benchmark")
    print("=" * 80)
    print(f"{'Size':>6} | {'Cells':>12} | {'Time (s)':>10} | {'MCells/s':>10} | "
          f"{'Memory (GB)':>11} | {'Field Sum':>12}")
    print("-" * 80)
    
    results = []
    for size in sizes:
        result = benchmark_size(size)
        results.append(result)
        
        if result['success']:
            print(f"{size:6d} | {result['cells']:12d} | {result['time']:10.3f} | "
                  f"{result['cells_per_second']/1e6:10.2f} | "
                  f"{result['memory_used_gb']:11.3f} | {result['field_sum']:12.2f}")
        else:
            print(f"{size:6d} | {result['cells']:12d} | {'FAILED':>10} | "
                  f"{'ERROR':>10} | {'ERROR':>11} | {'ERROR':>12}")
    
    print("=" * 80)
    
    successful_results = [r for r in results if r['success']]
    if successful_results:
        best_perf = max(successful_results, key=lambda x: x['cells_per_second'])
        print(f"\nBest Performance:")
        print(f"Grid Size: {best_perf['size']}x{best_perf['size']}")
        print(f"Processing Speed: {best_perf['cells_per_second']/1e6:.2f} Million cells/second")
        print(f"Memory Usage: {best_perf['memory_used_gb']:.3f} GB")

if __name__ == "__main__":
    run_benchmark_suite()
