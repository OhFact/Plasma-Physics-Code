import time
import jax
import jax.numpy as jnp
from jax.scipy.special import factorial
from jax.scipy.integrate import trapezoid
from quadax import quadts
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# Define the distribution function
def f(x, y, z, vx, vy, vz):
    vt = 1.0  # Set a specific thermal velocity
    return (2 / (((2 * jnp.pi) ** (3 / 2)) * vt ** 3) *
            jnp.exp(-(vx ** 2 + vy ** 2 + vz ** 2) / (2 * vt ** 2)))

# Define the Hermite function
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

# Trapezoidal Integration-Based Function
def compute_C_nmp_trap(f, alpha, u, Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, indices):
    x = jnp.linspace(0, Lx, Nx)
    y = jnp.linspace(0, Ly, Ny)
    z = jnp.linspace(0, Lz, Nz)
    vx = jnp.linspace(-5 * alpha[0] + u[0], 5 * alpha[0] + u[0], 40)
    vy = jnp.linspace(-5 * alpha[1] + u[1], 5 * alpha[1] + u[1], 40)
    vz = jnp.linspace(-5 * alpha[2] + u[2], 5 * alpha[2] + u[2], 40)
    p = jnp.floor(indices / (Nn * Nm)).astype(int)
    m = jnp.floor((indices - p * Nn * Nm) / Nn).astype(int)
    n = (indices - p * Nn * Nm - m * Nn).astype(int)

    X, Y, Z, Vx, Vy, Vz = jnp.meshgrid(x, y, z, vx, vy, vz, indexing='xy')

    xi_x = (Vx - u[0]) / alpha[0]
    xi_y = (Vy - u[1]) / alpha[1]
    xi_z = (Vz - u[2]) / alpha[2]

    return trapezoid(trapezoid(trapezoid(
        (f(X, Y, Z, vx, vy, vz) * Hermite(n, xi_x) * Hermite(m, xi_y) * Hermite(p, xi_z)) / \
        jnp.sqrt(factorial(n) * factorial(m) * factorial(p) * 2 ** (n + m + p)),
        (vx - u[0]) / alpha[0], axis=-3), (vy - u[1]) / alpha[1], axis=-2), (vz - u[2]) / alpha[2], axis=-1)

def compute_C_nmp_quadts(f, alpha, u, Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, indices):
    x = jnp.linspace(0, Lx, Nx)
    y = jnp.linspace(0, Ly, Ny)
    z = jnp.linspace(0, Lz, Nz)
    p = jnp.floor(indices / (Nn * Nm)).astype(int)
    m = jnp.floor((indices - p * Nn * Nm) / Nn).astype(int)
    n = (indices - p * Nn * Nm - m * Nn).astype(int)

    def integral_vz(x, y, z, vx, vy):
        interval = jnp.array([-5 * alpha[2] + u[2], 5 * alpha[2] + u[2]])
        return quadts(lambda vz: f(x, y, z, vx, vy, vz) * Hermite(0, (vz - u[2]) / alpha[2]), interval)[0]

    def integral_vy(x, y, z, vx):
        interval = jnp.array([-5 * alpha[1] + u[1], 5 * alpha[1] + u[1]])
        return quadts(lambda vy: integral_vz(x, y, z, vx, vy) * Hermite(0, (vy - u[1]) / alpha[1]), interval)[0]

    def integral_vx(x, y, z):
        interval = jnp.array([-5 * alpha[0] + u[0], 5 * alpha[0] + u[0]])
        return quadts(lambda vx: integral_vy(x, y, z, vx) * Hermite(0, (vx - u[0]) / alpha[0]), interval)[0]

    C_nmp = jnp.zeros((Ny, Nx, Nz))
    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                C_nmp = C_nmp.at[iy, ix, iz].set(integral_vx(x[ix], y[iy], z[iz]) /
                                                 jnp.sqrt(factorial(0) * factorial(0) * factorial(0) * 2 ** (0 + 0 + 0)))
    return C_nmp

alpha = [1.0, 1.0, 1.0]
u = [0.0, 0.0, 0.0]
Nx, Ny, Nz = 8, 8, 8
Lx, Ly, Lz = 10.0, 10.0, 10.0
Nn, Nm, Np = 1, 1, 1
indices = 0


start_time = time.time()
result_trap = compute_C_nmp_trap(f, alpha, u, Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, indices)
trap_time = time.time() - start_time
print(f"Trapezoidal: Computation time = {trap_time:.6f} sec")

# Visualization
x_vals = jnp.linspace(0, Lx, Nx)
y_vals = jnp.linspace(0, Ly, Ny)
X, Y = jnp.meshgrid(x_vals, y_vals)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.contourf(X, Y, result_trap[:, :, Nz // 2], cmap='viridis')
plt.colorbar(label="C_nmp Value (Trapezoidal)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Computed C_nmp (Trapezoidal)")

plt.show()

start_time = time.time()
result_quadts = compute_C_nmp_quadts(f, alpha, u, Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, indices)
quadts_time = time.time() - start_time
print(f"quadts: Computation time = {quadts_time:.6f} sec")

plt.subplot(1, 2, 2)
plt.contourf(X, Y, result_quadts[:, :, Nz // 2], cmap='magma')
plt.colorbar(label="C_nmp Value (quadts)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Computed C_nmp_ts")

plt.show()
