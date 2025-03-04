import jax.numpy as jnp


def Orszag_Tang(Lx, Ly, Omega_ce, mi_me):
    """
    I have to add docstrings!
    """

    vte = jnp.sqrt(0.25 / 2) * Omega_ce  # Electron thermal velocity.
    vti = vte * jnp.sqrt(1 / mi_me)  # Ion thermal velocity.
    deltaB = 0.2  # In-plane magnetic field amplitude.
    U0 = deltaB * Omega_ce / jnp.sqrt(mi_me)  # Fluid velocity amplitude.

    # Wavenumbers.
    kx = 2 * jnp.pi / Lx
    ky = 2 * jnp.pi / Ly

    # Electron and ion fluid velocities.
    Ue = lambda x, y, z: U0 * jnp.array(
        [-jnp.sin(ky * y), jnp.sin(kx * x), -deltaB * Omega_ce * (2 * kx * jnp.cos(2 * kx * x) + ky * jnp.cos(ky * y))])
    Ui = lambda x, y, z: U0 * jnp.array([-jnp.sin(ky * y), jnp.sin(kx * x), jnp.zeros_like(x)])

    # Magnetic and electric fields.
    B = lambda x, y, z: jnp.array([-deltaB * jnp.sin(ky * y), deltaB * jnp.sin(2 * kx * x), jnp.ones_like(x)])
    E = lambda x, y, z: jnp.array(
        [jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)])  # Is this consistent with fe, fi?

    # Electron and ion distribution functions.
    fe = (lambda x, y, z, vx, vy, vz: (1 / (((2 * jnp.pi) ** (3 / 2)) * vte ** 3) *
                                       jnp.exp(-((vx - Ue(x, y, z)[0]) ** 2 + (vy - Ue(x, y, z)[1]) ** 2 + (
                                                   vz - Ue(x, y, z)[2]) ** 2) / (2 * vte ** 2))))
    fi = (lambda x, y, z, vx, vy, vz: (1 / (((2 * jnp.pi) ** (3 / 2)) * vti ** 3) *
                                       jnp.exp(-((vx - Ui(x, y, z)[0]) ** 2 + (vy - Ui(x, y, z)[1]) ** 2 + (
                                                   vz - Ui(x, y, z)[2]) ** 2) / (2 * vti ** 2))))

    return B, E, fe, fi
