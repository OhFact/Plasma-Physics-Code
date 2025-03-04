import time
import jax
import jax.numpy as jnp
from jax.scipy.special import factorial
from jax.scipy.integrate import trapezoid
from quadax import quadgk, quadcc, quadts
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

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

#def integrate(f, a, b, method):
#    if method == 'quadgk':
#        return quadgk(f, (a, b))
#    elif method == 'quadcc':
#        return quadcc(f, (a, b))
#    elif method == 'quadts':
#        return quadts(f, (a, b))
#    else:
#        return trapezoid(f(x_input), x=x_input)

def compute_C_nmp(f, alpha, u, Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, indices):
    """
    Computes the C value for any given point in 3D position-velocity space.
    Given function f, vars alpha, u, length in all directions L_x, L_y, L_z,
    modes in position space N_x, N_y, N_z, and modes in velocity space N_n, N_m, N_p,
    C_{nmp} is computed.
    """
    p = jnp.floor(indices / (Nn * Nm)).astype(int)
    m = jnp.floor((indices - p * Nn * Nm) / Nn).astype(int)
    n = (indices - p * Nn * Nm - m * Nn).astype(int)

    x = jnp.linspace(0, Lx, Nx)
    y = jnp.linspace(0, Ly, Ny)
    z = jnp.linspace(0, Lz, Nz)
    vx = jnp.linspace(-5 * alpha[0] + u[0], 5 * alpha[0] + u[0], 40)
    vy = jnp.linspace(-5 * alpha[1] + u[1], 5 * alpha[1] + u[1], 40)
    vz = jnp.linspace(-5 * alpha[2] + u[2], 5 * alpha[2] + u[2], 40)

    def add_C_nmp(i, C_nmp):
        ivx = jnp.floor(i / (5 ** 2)).astype(int)
        ivy = jnp.floor((i - ivx * 5 ** 2) / 5).astype(int)
        ivz = (i - ivx * 5 ** 2 - ivy * 5).astype(int)

        vx_slice = jax.lax.dynamic_slice(vx, (ivx * 8,), (8,))
        vy_slice = jax.lax.dynamic_slice(vy, (ivy * 8,), (8,))
        vz_slice = jax.lax.dynamic_slice(vz, (ivz * 8,), (8,))

        X, Y, Z, Vx, Vy, Vz = jnp.meshgrid(x, y, z, vx_slice, vy_slice, vz_slice, indexing='xy')

        xi_x = (Vx - u[0]) / alpha[0]
        xi_y = (Vy - u[1]) / alpha[1]
        xi_z = (Vz - u[2]) / alpha[2]

        return C_nmp + trapezoid(trapezoid(trapezoid(
            (f(X, Y, Z, Vx, Vy, Vz) * Hermite(n, xi_x) * Hermite(m, xi_y) * Hermite(p, xi_z)) /
            jnp.sqrt(factorial(n) * factorial(m) * factorial(p) * 2 ** (n + m + p)),
            (vx_slice - u[0]) / alpha[0], axis=-3), (vy_slice - u[1]) / alpha[1], axis=-2), (vz_slice - u[2]) / alpha[2], axis=-1)

    Nv = 125
    return jax.lax.fori_loop(0, Nv, add_C_nmp, jnp.zeros((Ny, Nx, Nz)))

def f(x, y, z, vx, vy, vz):
    return jnp.exp(-(x**2 + y**2 + z**2 + vx**2 + vy**2 + vz**2) / 2)



alpha = [1.0, 1.0, 1.0]
u = [0.0, 0.0, 0.0]
Nx, Ny, Nz = 16, 16, 16
Lx, Ly, Lz = 10.0, 10.0, 10.0
Nn, Nm, Np = 1, 1, 1
indices = 0

#methods = ['trapezoid', 'quadgk', 'quadcc', 'quadts']
#results = {}
#for method in methods:
#    start_time = time.time()
#    results[method] = compute_C_nmp(f, alpha, u, Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, indices, method)
#    print(f"{method}: Result = {results[method]}, Time = {time.time() - start_time:.6f} sec")
start_time = time.time()
result = compute_C_nmp(f, alpha, u, Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, indices)
print(f"Trapezoid: Result = {result}, Time = {time.time() - start_time:.6f} sec")
# Plot results
plt.figure(figsize=(8, 6))
plt.imshow(result[:, :, Nz//2], extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
plt.colorbar(label='C_{nmp}')
plt.title("Computed C_{nmp} at z=Lz/2")
plt.xlabel("x")
plt.ylabel("y")
plt.show()