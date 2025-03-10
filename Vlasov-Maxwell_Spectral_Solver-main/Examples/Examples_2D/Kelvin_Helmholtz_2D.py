import jax.numpy as jnp


def Kelvin_Helmholtz_2D(Lx, Ly, Omega_ce, alpha_e, alpha_i):
    """
    I have to add docstrings!
    """

    vte = alpha_e / jnp.sqrt(2)  # Electron thermal velocity.
    vti = alpha_i / jnp.sqrt(2)  # Ion thermal velocity.
    U0 = 0.001  # Background flow amplitude.
    dU0 = 0.0001  # Velocity perturbation amplitude.

    # Wavenumbers.
    kx = 16 * jnp.pi / Lx

    # Electron and ion fluid velocities.
    Ue = lambda x, y, z: U0 * jnp.array(
        [dU0 * jnp.sin(kx * y / 2), jnp.tanh(kx * (x - Lx / 4)) * (jnp.sign(Lx / 2 - x) + 1) / 2
         + jnp.tanh(kx * (3 * Lx / 4 - x)) * (jnp.sign(x - Lx / 2) + 1) / 2, jnp.zeros_like(x)])
    Ui = lambda x, y, z: U0 * jnp.array(
        [dU0 * jnp.sin(kx * y / 2), jnp.tanh(kx * (x - Lx / 4)) * (jnp.sign(Lx / 2 - x) + 1) / 2
         + jnp.tanh(kx * (3 * Lx / 4 - x)) * (jnp.sign(x - Lx / 2) + 1) / 2, jnp.zeros_like(x)])

    # Ue = lambda x, y, z: U0 * jnp.array([jnp.sin(ky * y), jnp.zeros_like(x), jnp.zeros_like(x)])
    # Ui = lambda x, y, z: U0 * jnp.array([jnp.sin(ky * y), jnp.zeros_like(x), jnp.zeros_like(x)])

    # Magnetic and electric fields.
    B = lambda x, y, z: jnp.array([jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)])
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
