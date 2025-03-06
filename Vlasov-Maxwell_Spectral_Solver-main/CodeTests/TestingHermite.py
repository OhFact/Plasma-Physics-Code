import jax.numpy as jnp
import jax
import time
import matplotlib.pyplot as plt
from orthax import hermite

jax.config.update("jax_enable_x64", True)

"""
To use this file, install orthax, as we are using the hermite functions, 
and comparing it to the recurrence method used in our code
"""


@jax.jit
def hermite_recurrence(n, x):
    x = jnp.asarray(x, dtype=jnp.float64)
    H_n_minus_2 = jnp.ones_like(x)
    H_n_minus_1 = 2 * x

    def body_fn(i, carry):
        H_n_minus_2, H_n_minus_1 = carry
        H_n = 2 * x * H_n_minus_1 - 2 * (i - 1) * H_n_minus_2
        return (H_n_minus_1, H_n)

    _, H_n = jax.lax.fori_loop(2, n + 1, body_fn, (H_n_minus_2, H_n_minus_1))
    return H_n

def hermite_3d_recurrence(n, x, y, z):
    Hx = hermite_recurrence(n, x)
    Hy = hermite_recurrence(n, y)
    Hz = hermite_recurrence(n, z)
    return Hx * Hy * Hz

def grid(grid_size=20):
    linspace = jnp.linspace(-3, 3, grid_size)
    x, y, z = jnp.meshgrid(linspace, linspace, linspace, indexing='ij')
    return x, y, z

def benchmark_hermite_3d(n, grid_size=20):
    x, y, z = grid(grid_size)

    start_time = time.time()
    recurrence_result = hermite_3d_recurrence(n, x, y, z)
    recurrence_time = time.time() - start_time

    c = jnp.zeros((n + 1, n + 1, n + 1))
    c = c.at[n, n, n].set(1)
    start_time = time.time()
    orthax_result = hermite.hermval3d(x, y, z, c)
    orthax_time = time.time() - start_time

    match = jnp.allclose(recurrence_result, orthax_result, atol=1e-6, rtol=1e-4)

    print(f"Order: {n}")
    print(f"Recurrence Time: {recurrence_time:.6f} s")
    print(f"Orthax Time: {orthax_time:.6f} s")
    print(f"Match: {'Yes' if match else 'No'}")

    mid_idx = grid_size // 2
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(recurrence_result[:, :, mid_idx], cmap='BuPu_r', extent=[-10, 10, -10, 10])
    plt.colorbar()
    plt.title("Recurrence")

    plt.subplot(1, 2, 2)
    plt.imshow(orthax_result[:, :, mid_idx], cmap='BuPu_r', extent=[-10, 10, -10, 10])
    plt.colorbar()
    plt.title("Orthax")

    plt.show()


benchmark_hermite_3d(10, grid_size=30)