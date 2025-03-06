import time
import jax
import jax.numpy as jnp
from jax.scipy.special import factorial
from jax.scipy.integrate import trapezoid
from quadax import quadts
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

"""
To use this file install quadax, in order to use the integration functions
Currently the quadax integration is taking far too long, need to look into it
"""


def distribution(vt):
    return lambda x, y, z, vx, vy, vz: (2 / (((2 * jnp.pi) ** (3 / 2)) * vt ** 3) *
                                        jnp.exp(-(vx ** 2 + vy ** 2 + vz ** 2) / (2 * vt ** 2)))

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

def compute_C_nmp_trap(f, alpha, u, Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, indices):
    x = jnp.linspace(0, Lx, Nx)
    y = jnp.linspace(0, Ly, Ny)
    z = jnp.linspace(0, Lz, Nz)
    vx = jnp.linspace(-5 * alpha[0] + u[0], 5 * alpha[0] + u[0], 40)
    vy = jnp.linspace(-5 * alpha[1] + u[1], 5 * alpha[1] + u[1], 40)
    vz = jnp.linspace(-5 * alpha[2] + u[2], 5 * alpha[2] + u[2], 40)
    X, Y, Z, Vx, Vy, Vz = jnp.meshgrid(x, y, z, vx, vy, vz, indexing='ij')
    return trapezoid(trapezoid(trapezoid(f(X, Y, Z, Vx, Vy, Vz), vx, axis=-3), vy, axis=-2), vz, axis=-1)

def compute_C_nmp_quadts(f, alpha, u, Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, indices):
    x = jnp.linspace(0, Lx, Nx)
    y = jnp.linspace(0, Ly, Ny)
    z = jnp.linspace(0, Lz, Nz)
    def integral_vz(x, y, z, vx, vy):
        interval = jnp.array([-5 * alpha[2] + u[2], 5 * alpha[2] + u[2]])
        return quadts(lambda vz: f(x, y, z, vx, vy, vz), interval)[0]
    def integral_vy(x, y, z, vx):
        interval = jnp.array([-5 * alpha[1] + u[1], 5 * alpha[1] + u[1]])
        return quadts(lambda vy: integral_vz(x, y, z, vx, vy), interval)[0]
    def integral_vx(x, y, z):
        interval = jnp.array([-5 * alpha[0] + u[0], 5 * alpha[0] + u[0]])
        return quadts(lambda vx: integral_vy(x, y, z, vx), interval)[0]
    return jnp.array([[[integral_vx(x[ix], y[iy], z[iz]) for iz in range(Nz)] for iy in range(Ny)] for ix in range(Nx)])

vt = 1.0
f = distribution(vt)
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

start_time = time.time()
result_quadts = compute_C_nmp_quadts(f, alpha, u, Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, indices)
quadts_time = time.time() - start_time
print(f"quadts: Computation time = {quadts_time:.6f} sec")

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

plt.subplot(1, 2, 2)
plt.contourf(X, Y, result_quadts[:, :, Nz // 2], cmap='magma')
plt.colorbar(label="C_nmp Value (quadts)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Computed C_nmp (quadts)")

plt.show()