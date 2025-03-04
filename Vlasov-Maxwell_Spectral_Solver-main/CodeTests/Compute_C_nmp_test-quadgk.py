import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy.special import factorial
from quadax import quadgk

jax.config.update("jax_enable_x64", True)

x_input = jnp.linspace(0, 10, 100)


@jax.jit
def Hermite(n, x):
    x = jnp.asarray(x, dtype=jnp.float64)

    def base_case_0(_): return jnp.ones_like(x)

    def base_case_1(_): return 2 * x

    def recurrence_case(n):
        H_n_minus_2, H_n_minus_1 = jnp.ones_like(x), 2 * x

        def body_fn(i, c):
            H_n_minus_2, H_n_minus_1 = c
            H_n = 2 * x * H_n_minus_1 - 2 * (i - 1) * H_n_minus_2
            return (H_n_minus_1, H_n)

        _, H_n = jax.lax.fori_loop(2, n + 1, body_fn, (H_n_minus_2, H_n_minus_1))
        return H_n

    return jax.lax.switch(n, [base_case_0, base_case_1, recurrence_case], n)

def compute_C_nmp(f, alpha, u, Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, indices):
    p = jnp.floor(indices / (Nn * Nm)).astype(int)
    m = jnp.floor((indices - p * Nn * Nm) / Nn).astype(int)
    n = (indices - p * Nn * Nm - m * Nn).astype(int)

    x = jnp.linspace(0, Lx, Nx)
    y = jnp.linspace(0, Ly, Ny)
    z = jnp.linspace(0, Lz, Nz)

    def integrand(vx, vy, vz, X, Y, Z):
        xi_x = (vx - u[0]) / alpha[0]
        xi_y = (vy - u[1]) / alpha[1]
        xi_z = (vz - u[2]) / alpha[2]
        return (f(X, Y, Z, vx, vy, vz) * Hermite(n, xi_x) * Hermite(m, xi_y) * Hermite(p, xi_z)) / \
            jnp.sqrt(factorial(n) * factorial(m) * factorial(p) * 2 ** (n + m + p))

    C_nmp = jnp.zeros((Ny, Nx, Nz))
    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                X, Y, Z = x[ix], y[iy], z[iz]

                integral, _ = quadgk(lambda vx:
                                     quadgk(lambda vy:
                                            quadgk(lambda vz: integrand(vx, vy, vz, X, Y, Z),
                                                   -5 * alpha[2] + u[2], 5 * alpha[2] + u[2])[0],
                                            -5 * alpha[1] + u[1], 5 * alpha[1] + u[1])[0],
                                     -5 * alpha[0] + u[0], 5 * alpha[0] + u[0])

                C_nmp = C_nmp.at[iy, ix, iz].set(integral)
    return C_nmp


def f(x, y, z, vx, vy, vz):
    return jnp.exp(-(x ** 2 + y ** 2 + z ** 2 + vx ** 2 + vy ** 2 + vz ** 2) / 2)


alpha = [1.0, 1.0, 1.0]
u = [0.0, 0.0, 0.0]
Nx, Ny, Nz = 16, 16, 16
Lx, Ly, Lz = 10.0, 10.0, 10.0
Nn, Nm, Np = 1, 1, 1
indices = 0

start_time = time.time()
result = compute_C_nmp(f, alpha, u, Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, indices)
print(f"quadgk: Result computed, Time = {time.time() - start_time:.6f} sec")

# Visualization
x_vals = jnp.linspace(0, Lx, Nx)
y_vals = jnp.linspace(0, Ly, Ny)
X, Y = jnp.meshgrid(x_vals, y_vals)

plt.figure(figsize=(10, 6))
plt.contourf(X, Y, result[:, :, Nz // 2], cmap='viridis')
plt.colorbar(label="C_nmp Value")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Computed C_nmp Values (Slice at z = Lz/2)")
plt.show()
