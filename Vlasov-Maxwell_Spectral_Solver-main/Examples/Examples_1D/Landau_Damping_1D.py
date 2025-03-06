import jax.numpy as jnp


def Landau_damping_1D(Lx, Omega_ce, mi_me):
    """
    I have to add docstrings!
    """

    vte = jnp.sqrt(0.25 / 2)  # Electron thermal velocity.
    vti = vte * jnp.sqrt(1 / mi_me)  # Ion thermal velocity.

    kx = 2 * jnp.pi / Lx  # Wavenumber.

    dn = 0.01  # Density fluctuation.

    # Magnetic and electric fields.
    B = lambda x, y, z: jnp.array([Omega_ce * jnp.ones_like(x), jnp.zeros_like(y), jnp.zeros_like(z)])
    E = lambda x, y, z: jnp.array(
        [(dn / kx) * jnp.cos(kx * x), jnp.zeros_like(y), jnp.zeros_like(z)])  # Is this consistent with fe, fi?

    # Electron and ion distribution functions.
    fe = (lambda x, y, z, vx, vy, vz: (1 / (((2 * jnp.pi) ** (3 / 2)) * vte ** 3) *
                                       jnp.exp(-(vx ** 2 + vy ** 2 + vz ** 2) / (2 * vte ** 2))) *
                                      (1 + dn * jnp.sin(kx * x)))
    fi = (lambda x, y, z, vx, vy, vz: (1 / (((2 * jnp.pi) ** (3 / 2)) * vti ** 3) *
                                       jnp.exp(-(vx ** 2 + vy ** 2 + vz ** 2) / (2 * vti ** 2))))

    return B, E, fe, fi


def Landau_damping_HF_1D(Lx, Ly, Lz, Omega_ce, alpha_e, alpha_i, Nn):
    """
    I have to add docstrings!
    """

    vte = alpha_e / jnp.sqrt(2)  # Electron thermal velocity.
    vti = alpha_i / jnp.sqrt(2)  # Ion thermal velocity.

    kx = 2 * jnp.pi / Lx  # Wavenumber.

    dn = 0.0001  # Density fluctuation.

    # Fourier components of magnetic and electric fields.
    Fk_0 = jnp.zeros((6, 1, 3, 1), dtype=jnp.complex128)
    Fk_0 = Fk_0.at[0, 0, 0, 0].set(dn / (2 * kx))
    Fk_0 = Fk_0.at[0, 0, 2, 0].set(dn / (2 * kx))
    #    Fk_0 = Fk_0.at[3, 0, 1, 0].set(Omega_ce)

    # Hermite-Fourier components of electron and ion distribution functions.
    Ce0_mk, Ce0_0, Ce0_k = 0 + 1j * (1 / 2 ** (5 / 2)) * (1 / vte ** 3) * dn, 1 / (
                (2 ** (3 / 2)) * (vte ** 3)) + 0 * 1j, 0 - 1j * (1 / 2 ** (5 / 2)) * (1 / vte ** 3) * dn
    Ci0_0 = 1 / ((2 ** (3 / 2)) * (vti ** 3)) + 0 * 1j
    Ck_0 = jnp.zeros((2 * Nn, 1, 3, 1), dtype=jnp.complex128)
    Ck_0 = Ck_0.at[0, 0, 0, 0].set(Ce0_mk)
    Ck_0 = Ck_0.at[0, 0, 1, 0].set(Ce0_0)
    Ck_0 = Ck_0.at[0, 0, 2, 0].set(Ce0_k)
    Ck_0 = Ck_0.at[Nn, 0, 1, 0].set(Ci0_0)

    return Ck_0, Fk_0
