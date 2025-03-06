import jax.numpy as jnp


def pressure_anisotropy_HF_1D(Lx, Ly, Lz, Omega_ce, alpha_s, Nn):
    """
    I have to add docstrings!
    """

    vte_x = alpha_s[0] / jnp.sqrt(2) # Electron thermal velocity along x.
    vte_perp = alpha_s[1] / jnp.sqrt(2) # Electron thermal velocity along yz.
    vti = alpha_s[3] / jnp.sqrt(2) # Ion thermal velocity.

    kx = 2 * jnp.pi / Lx # Wavenumber.

    dn = 0.01 # Density fluctuation.

    # Fourier components of magnetic and electric fields.
    Fk_0 = jnp.zeros((6, 3, 1, 1), dtype=jnp.complex128)
    Fk_0 = Fk_0.at[0, 0, 0, 0].set(dn / (2 * kx))
    Fk_0 = Fk_0.at[0, 2, 0, 0].set(dn / (2 * kx))
    Fk_0 = Fk_0.at[3, 1, 0, 0].set(Omega_ce)


    # Hermite-Fourier components of electron and ion distribution functions.
    Ce0_mk, Ce0_0, Ce0_k = 0 + 1j * (1 / 2 ** (5/2)) * (1 / vte ** 3) * dn, 1 / ((2 ** (3/2)) * (vte ** 3)) + 0 * 1j, 0 - 1j * (1 / 2 ** (5/2)) * (1 / vte ** 3) * dn
    Ci0_0 = 1 / ((2 ** (3/2)) * (vti ** 3)) + 0 * 1j
    Ck_0 = jnp.zeros((2 * Nn, 3, 1, 1), dtype=jnp.complex128)
    Ck_0 = Ck_0.at[0, 0, 0, 0].set(Ce0_mk)
    Ck_0 = Ck_0.at[0, 1, 0, 0].set(Ce0_0)
    Ck_0 = Ck_0.at[0, 2, 0, 0].set(Ce0_k)
    Ck_0 = Ck_0.at[Nn, 1, 0, 0].set(Ci0_0)

    return Ck_0, Fk_0