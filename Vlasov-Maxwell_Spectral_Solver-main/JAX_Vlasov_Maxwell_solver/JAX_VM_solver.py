# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 10:55:45 2024

@author: cristian
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.signal import convolve
from jax.numpy.fft import fftn, fftshift
from jax.scipy.special import factorial
from jax.scipy.integrate import trapezoid
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt
from functools import partial
from Examples.Examples_1D.Landau_Damping_1D import *
from Examples.Examples_1D.density_perturbation_1D import *
from Examples.Examples_1D.pressure_anisotropy_HF_1D import *
from Examples.Examples_1D.Two_Stream_1D import *
from Examples.Examples_2D.Kelvin_Helmholtz_2D import *
from Examples.Examples_2D.Orszag_Tang import *


@jax.jit
def Hermite(n, x):
    """
    Generates output for any mode and x value.
    n - modes
    x - input value
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    
    def base_case_0(_):
        return jnp.ones_like(x)
    
    def base_case_1(_):
        return 2 * x
    
    def recurrence_case(n):
        H_n_minus_2 = jnp.ones_like(x)
        H_n_minus_1 = 2 * x
        
        def body_fn(i, carry):
            H_n_minus_2, H_n_minus_1 = carry
            H_n = 2 * x * H_n_minus_1 - 2 * (i - 1) * H_n_minus_2
            return (H_n_minus_1, H_n)
        
        _, H_n = jax.lax.fori_loop(2, n + 1, body_fn, (H_n_minus_2, H_n_minus_1))
        return H_n
    
    return jax.lax.switch(n, [base_case_0, base_case_1, recurrence_case], n)

@jax.jit
def generate_Hermite_basis(xi_x, xi_y, xi_z, Nn, Nm, Np, indices):
    """
    Generates a Hermite basis function at a point in space and for specific numbers of modes in each direction.
    xi_x, xi_y, xi_z - spatial coordinates in any direction
    Nn, Nm, Np - number of modes in any direction
    indices - collapsed 3D array, p, m, n reconstructed through operations
    """
    p = jnp.floor(indices / (Nn * Nm)).astype(int)
    m = jnp.floor((indices - p * Nn * Nm) / Nn).astype(int)
    n = (indices - p * Nn * Nm - m * Nn).astype(int)

    hermite_x = Hermite(n, xi_x)
    hermite_y = Hermite(m, xi_y)
    hermite_z = Hermite(p, xi_z)
    
    exp_term = jnp.exp(-(xi_x**2 + xi_y**2 + xi_z**2))
    normalization_factor = 1 / jnp.sqrt((jnp.pi)**3 * 2**(n + m + p) * factorial(n) * factorial(m) * factorial(p))
    Hermite_basis = hermite_x * hermite_y * hermite_z * exp_term * normalization_factor

    return Hermite_basis

def moving_average(data, window_size):
    """
    Computes the moving average in any size window of an array.
    data - iterable variable that contains all data to compute averages over
    window_size - size of window to average over
    """
    data_array = jnp.array(data)
    kernel = jnp.ones(window_size) / window_size
    return jnp.convolve(data_array, kernel, mode='valid')

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

def initialize_system_xp(Omega_ce, mi_me, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns):
    """
    Initializes the system with fields and distributions.
    """
    B, E, fe, fi = Kelvin_Helmholtz_2D(Lx, Ly, Omega_ce, alpha_s[0], alpha_s[1])

    Ce_0 = jax.vmap(compute_C_nmp, in_axes=(None, None, None, None, None, None, None, None, None, None, None, None, 0))(
        fe, alpha_s[:3], u_s[:3], Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, jnp.arange(Nn * Nm * Np))
    Ci_0 = jax.vmap(compute_C_nmp, in_axes=(None, None, None, None, None, None, None, None, None, None, None, None, 0))(
        fi, alpha_s[3:], u_s[3:], Nx, Ny, Nz, Lx, Ly, Lz, Nn, Nm, Np, jnp.arange(Nn * Nm * Np))

    Cek_0 = fftshift(fftn(Ce_0, axes=(-3, -2, -1)), axes=(-3, -2, -1))
    Cik_0 = fftshift(fftn(Ci_0, axes=(-3, -2, -1)), axes=(-3, -2, -1))

    Ck_0 = jnp.concatenate([Cek_0, Cik_0])

    x = jnp.linspace(0, Lx, Nx)
    y = jnp.linspace(0, Ly, Ny)
    z = jnp.linspace(0, Lz, Nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='xy')
    
    Ek_0 = fftshift(fftn(E(X, Y, Z), axes=(-3, -2, -1)), axes=(-3, -2, -1))
    Bk_0 = fftshift(fftn(B(X, Y, Z), axes=(-3, -2, -1)), axes=(-3, -2, -1))

    Fk_0 = jnp.concatenate([Ek_0, Bk_0])
    
    return Ck_0, Fk_0

@jax.jit
def cross_product(k_vec, F_vec):
    """
    Computes cross product of 2 3D vectors.
    k_vec - vector of Fourier transformation variables
    F_vec - force vector
    """
    kx, ky, kz = k_vec
    Fx, Fy, Fz = F_vec
    
    result_x = ky * Fz - kz * Fy
    result_y = kz * Fx - kx * Fz
    result_z = kx * Fy - ky * Fx

    return jnp.array([result_x, result_y, result_z])

def compute_dCk_s_dt(Ck, Fk, kx_grid, ky_grid, kz_grid, Lx, Ly, Lz, nu, alpha_s, u_s, qs, Omega_cs, Nn, Nm, Np, indices):
    """
    Simulates time evolution of the coefficients C.
    Ck - Hermite Fourier coefficients
    Fk - force coefficients
    kx_grid, ky_grid, kz_grid - grid of Fourier transformation variables
    Lx, Ly, Lz - length in all directions
    Nn, Nm, Np - modes in velocity space
    indices - indices for Hermite polynomials
    Omega_cs - gyrokinetic frequency
    alpha_s - thermal velocity of species
    u_s - reference velocity for species
    qs - normalized charge of each species
    """
    s = jnp.floor(indices / (Nn * Nm * Np)).astype(int)
    p = jnp.floor((indices - s * Nn * Nm * Np) / (Nn * Nm)).astype(int)
    m = jnp.floor((indices - s * Nn * Nm * Np - p * Nn * Nm) / Nn).astype(int)
    n = (indices - s * Nn * Nm * Np - p * Nn * Nm - m * Nn).astype(int)
    
    u = jax.lax.dynamic_slice(u_s, (s * 3,), (3,))
    alpha = jax.lax.dynamic_slice(alpha_s, (s * 3,), (3,))
    q, Omega_c = qs[s], Omega_cs[s]
    
    Ck_aux_x = (jnp.sqrt(m * p) * (alpha[2] / alpha[1] - alpha[1] / alpha[2]) * Ck[n + (m-1) * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) * jnp.sign(p) + 
        jnp.sqrt(m * (p + 1)) * (alpha[2] / alpha[1]) * Ck[n + (m-1) * Nn + (p+1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) * jnp.sign(Np - p - 1) - 
        jnp.sqrt((m + 1) * p) * (alpha[1] / alpha[2]) * Ck[n + (m+1) * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p) * jnp.sign(Nm - m - 1) + 
        jnp.sqrt(2 * m) * (u[2] / alpha[1]) * Ck[n + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) - 
        jnp.sqrt(2 * p) * (u[1] / alpha[2]) * Ck[n + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p)) 

    Ck_aux_y = (jnp.sqrt(n * p) * (alpha[0] / alpha[2] - alpha[2] / alpha[0]) * Ck[n-1 + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) * jnp.sign(p) + 
        jnp.sqrt((n + 1) * p) * (alpha[0] / alpha[2]) * Ck[n+1 + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p) * jnp.sign(Nn - n - 1) - 
        jnp.sqrt(n * (p + 1)) * (alpha[2] / alpha[0]) * Ck[n-1 + m * Nn + (p+1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) * jnp.sign(Np - p - 1) + 
        jnp.sqrt(2 * p) * (u[0] / alpha[2]) * Ck[n + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p) - 
        jnp.sqrt(2 * n) * (u[2] / alpha[0]) * Ck[n-1 + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n))
    
    Ck_aux_z = (jnp.sqrt(n * m) * (alpha[1] / alpha[0] - alpha[0] / alpha[1]) * Ck[n-1 + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) * jnp.sign(m) + 
        jnp.sqrt(n * (m + 1)) * (alpha[1] / alpha[0]) * Ck[n-1 + (m+1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) * jnp.sign(Nm - m - 1) - 
        jnp.sqrt((n + 1) * m) * (alpha[0] / alpha[1]) * Ck[n+1 + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) * jnp.sign(Nn - n - 1) + 
        jnp.sqrt(2 * n) * (u[1] / alpha[0]) * Ck[n-1 + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) - 
        jnp.sqrt(2 * m) * (u[0] / alpha[1]) * Ck[n + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m))
    
    Col = 0
        
    dCk_s_dt = (- (kx_grid * 1j / Lx) * alpha[0] * (
        jnp.sqrt((n + 1) / 2) * Ck[n+1 + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(Nn - n - 1) +
        jnp.sqrt(n / 2) * Ck[n-1 + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n) +
        (u[0] / alpha[0]) * Ck[n + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...]
    ) - (ky_grid * 1j / Ly) * alpha[1] * (
        jnp.sqrt((m + 1) / 2) * Ck[n + (m+1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(Nm - m - 1) +
        jnp.sqrt(m / 2) * Ck[n + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m) +
        (u[1] / alpha[1]) * Ck[n + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...]
    ) - (kz_grid * 1j / Lz) * alpha[2] * (
        jnp.sqrt((p + 1) / 2) * Ck[n + m * Nn + (p+1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(Np - p - 1) +
        jnp.sqrt(p / 2) * Ck[n + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p) +
        (u[2] / alpha[2]) * Ck[n + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...]
    ) + q * Omega_c * (
        (jnp.sqrt(2 * n) / alpha[0]) * convolve(Fk[0, ...], Ck[n-1 + m * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(n), mode='same') +
        (jnp.sqrt(2 * m) / alpha[1]) * convolve(Fk[1, ...], Ck[n + (m-1) * Nn + p * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(m), mode='same') +
        (jnp.sqrt(2 * p) / alpha[2]) * convolve(Fk[2, ...], Ck[n + m * Nn + (p-1) * Nn * Nm + s * Nn * Nm * Np, ...] * jnp.sign(p), mode='same')
    ) + q * Omega_c * (
        convolve(Fk[3, ...], Ck_aux_x, mode='same') + 
        convolve(Fk[4, ...], Ck_aux_y, mode='same') + 
        convolve(Fk[5, ...], Ck_aux_z, mode='same')
    ) + Col)
    
    return dCk_s_dt

def ampere_maxwell_current(qs, alpha_s, u_s, Ck, Nn, Nm, Np, Ns):
    """
    Simulates current flowing through the particles by species, and returns the full current.
    qs - array of charge values for different species
    alpha_s - thermal velocity of species
    u_s - mean velocities in x, y, z
    Ck - array of coefficients
    Nn, Nm, Np - modes in velocity space directions
    Ns - Number of species
    """
    def add_current_term(s, partial_sum):
        return partial_sum + qs[s] * alpha_s[s * 3] * alpha_s[s * 3 + 1] * alpha_s[s * 3 + 2] * (
            (1 / jnp.sqrt(2)) * jnp.array([alpha_s[s * 3] * Ck[s * Nn * Nm * Np + 1, ...] * jnp.sign(Nn - 1),
                                           alpha_s[s * 3 + 1] * Ck[s * Nn * Nm * Np + Nn, ...] * jnp.sign(Nm - 1),
                                           alpha_s[s * 3 + 2] * Ck[s * Nn * Nm * Np + Nn * Nm, ...] * jnp.sign(Np - 1)]) + 
                                jnp.array([u_s[s * 3] * Ck[s * Nn * Nm * Np, ...],
                                           u_s[s * 3 + 1] * Ck[s * Nn * Nm * Np, ...],
                                           u_s[s * 3 + 2] * Ck[s * Nn * Nm * Np, ...]]))
    
    return jax.lax.fori_loop(0, Ns, add_current_term, jnp.zeros_like(Ck[:3, ...]))

def ode_system(t, Ck_Fk, args):
    """
    Defines the ODE system for the simulation.
    """
    qs, nu, Omega_cs, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns = args
    
    kx = (jnp.arange(-Nx//2, Nx//2) + 1) * 2 * jnp.pi
    ky = (jnp.arange(-Ny//2, Ny//2) + 1) * 2 * jnp.pi
    kz = (jnp.arange(-Nz//2, Nz//2) + 1) * 2 * jnp.pi
    
    kx_grid, ky_grid, kz_grid = jnp.meshgrid(kx, ky, kz, indexing='xy')
    
    Ck = Ck_Fk[:(-6 * Nx * Ny * Nz)].reshape(Ns * Nn * Nm * Np, Ny, Nx, Nz)
    Fk = Ck_Fk[(-6 * Nx * Ny * Nz):].reshape(6, Ny, Nx, Nz)
    
    dCk_s_dt = jax.vmap(compute_dCk_s_dt, in_axes=(None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0))(
        Ck, Fk, kx_grid, ky_grid, kz_grid, Lx, Ly, Lz, nu, alpha_s, u_s, qs, Omega_cs, Nn, Nm, Np, jnp.arange(Nn * Nm * Np * Ns))
    
    current = ampere_maxwell_current(qs, alpha_s, u_s, Ck, Nn, Nm, Np, Ns)
        
    dBk_dt = - 1j * cross_product(jnp.array([kx_grid/Lx, ky_grid/Ly, kz_grid/Lz]), Fk[:3, ...])
    dEk_dt = 1j * cross_product(jnp.array([kx_grid/Lx, ky_grid/Ly, kz_grid/Lz]), Fk[3:, ...]) - (1 / Omega_cs[0]) * current
            
    dFk_dt = jnp.concatenate([dEk_dt, dBk_dt])
    dy_dt = jnp.concatenate([dCk_s_dt.flatten(), dFk_dt.flatten()])
    
    return dy_dt

#Add dt in vars for VM simulation, replace dt0 = dt

@partial(jax.jit, static_argnums=[9, 10, 11, 12, 13, 14, 15, 17])
def VM_simulation(qs, nu, Omega_cs, alpha_s, mi_me, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns, t_max, t_steps):
    """
    Runs the Vlasov-Maxwell simulation.
    """
    Ck_0, Fk_0 = Landau_damping_HF_1D(Lx, Ly, Lz, Omega_cs[0], alpha_s[0], alpha_s[3], Nn)
    
    initial_conditions = jnp.concatenate([Ck_0.flatten(), Fk_0.flatten()])

    t = jnp.linspace(0, t_max, t_steps)

    args = (qs, nu, Omega_cs, alpha_s, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns)
    
    saveat = SaveAt(ts=t)
    term = ODETerm(ode_system)
    solver = Dopri5()
    result = diffeqsolve(term, solver, t0=0, t1=t_max, dt0=0.05, y0=initial_conditions, args=args, saveat=saveat)
    
    Ck = result.ys[:,:(-6 * Nx * Ny * Nz)].reshape(len(result.ts), Ns * Nn * Nm * Np, Ny, Nx, Nz)
    Fk = result.ys[:,(-6 * Nx * Ny * Nz):].reshape(len(result.ts), 6, Ny, Nx, Nz)
    
    return Ck, Fk, result.ts