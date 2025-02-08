import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import ray

ray.init(ignore_reinit_error=True)

def diffuse_partitioned(field, prev_field, a, iterations=20):
    result = field.copy()
    denom = 1.0 + 4 * a

    for _ in range(iterations):
        for i in range(1, result.shape[0] - 1):
            for j in range(1, result.shape[1] - 1):
                bottom = result[i + 1, j]
                top = result[i - 1, j]
                right = result[i, j + 1]
                left = result[i, j - 1]

                temp_sum = bottom + top + right + left

                result[i, j] = (prev_field[i, j] + a * temp_sum) / denom

        result[0, :] = result[-1, :] = result[:, 0] = result[:, -1] = 0

    return result

@ray.remote
def parallel_diffuse(ray_block, prev_field, a, iterations=20):
    return diffuse_partitioned(ray_block, prev_field, a, iterations)

if __name__ == "__main__":
    nx, ny = 3200, 3200
    field = np.zeros((ny, nx))
    prev_field = np.random.rand(ny, nx)
    a = 0.1
    iterations = 20

    start_time = time.time()
    field_updated_serial = diffuse_partitioned(field, prev_field, a, iterations)
    serial_time = time.time() - start_time

    start_time = time.time()

    block_size_y = ny // 4
    blocks = [field[i * block_size_y:(i + 1) * block_size_y, :] for i in range(4)]

    results = ray.get([parallel_diffuse.remote(block, prev_field, a, iterations) for block in blocks])

    field_updated_parallel = np.vstack(results)
    parallel_time = time.time() - start_time

    print(f"Serial Processing Time: {serial_time:.4f} seconds")
    print(f"Parallel Processing Time: {parallel_time:.4f} seconds")

    print(f"Updated Field After Diffusion (Serial):")
    print(f"Min Value: {np.min(field_updated_serial)}")
    print(f"Max Value: {np.max(field_updated_serial)}")
    print(f"Mean Value: {np.mean(field_updated_serial)}")
    print(f"Standard Deviation: {np.std(field_updated_serial)}")

    print(f"Updated Field After Diffusion (Parallel):")
    print(f"Min Value: {np.min(field_updated_parallel)}")
    print(f"Max Value: {np.max(field_updated_parallel)}")
    print(f"Mean Value: {np.mean(field_updated_parallel)}")
    print(f"Standard Deviation: {np.std(field_updated_parallel)}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(field_updated_parallel, cmap="coolwarm", cbar=True)
    plt.title("Heatmap of the Updated Field (Parallel Diffusion)")
    plt.xlabel("X-axis (Grid Columns)")
    plt.ylabel("Y-axis (Grid Rows)")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(field_updated_serial, cmap="coolwarm", cbar=True)
    plt.title("Heatmap of the Updated Field (Serial Diffusion)")
    plt.xlabel("X-axis (Grid Columns)")
    plt.ylabel("Y-axis (Grid Rows)")
    plt.show()

    ray.shutdown()
