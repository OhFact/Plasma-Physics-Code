import json
import pandas as pd
import jax.numpy as jnp
from JAX_Vlasov_Maxwell_solver import *
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from jax.scipy.optimize import minimize
import time


# Define the function to fit
def model_function(params, t):
    A, B, omega, gamma = params
    return A * jnp.cos(omega * t) * jnp.exp(-gamma * t) + B


# Define the loss function (mean squared error)
def loss_function(params, t, data):
    predictions = model_function(params, t)
    return jnp.mean((predictions - data) ** 2)


with open('../ExampleParams/plasma_parameters_Landau_damping_HF_1D.json', 'r') as file:
    parameters = json.load(file)

# Unpack parameters.
Nx, Ny, Nz = parameters['Nx'], parameters['Ny'], parameters['Nz']
Nvx, Nvy, Nvz = parameters['Nvx'], parameters['Nvy'], parameters['Nvz']
Lx, Ly, Lz = parameters['Lx'], parameters['Ly'], parameters['Lz']
Nn, Nm, Np, Ns = parameters['Nn'], parameters['Nm'], parameters['Np'], parameters['Ns']
mi_me = parameters['mi_me']
Omega_cs = parameters['Omega_ce'] * jnp.array([1.0, 1.0 / mi_me])
qs = jnp.array(parameters['qs'])
alpha_s = jnp.concatenate([jnp.array(parameters['alpha_e']), (jnp.array(parameters['alpha_e']) / jnp.sqrt(mi_me))])
u_s = jnp.array(parameters['u_s'])
nu = parameters['nu']
t_steps, t_max = parameters['t_steps'], parameters['t_max']
#dt add here


# Save parameters into txt.
# with open('C:\Cristian\Postdoc\Madison\Code\Simulations\Landau_damping_1D_HF_ini\Landau_damping_1D_S29.txt', 'w') as file:
#     file.write(f"Nx, Ny, Nz: {Nx}, {Ny}, {Nz}\n")
#     file.write(f"Nvx, Nvy, Nvz: {Nvx}, {Nvy}, {Nvz}\n")
#     file.write(f"Lx, Ly, Lz: {Lx}, {Ly}, {Lz}\n")
#     file.write(f"Nn, Nm, Np, Ns: {Nn}, {Nm}, {Np}, {Ns}\n")
#     file.write(f"mi_me: {mi_me}\n")
#     file.write(f"Omega_cs: {Omega_cs.tolist()}\n")
#     file.write(f"qs: {qs.tolist()}\n")
#     file.write(f"alpha_s: {alpha_s.tolist()}\n")
#     file.write(f"u_s: {u_s.tolist()}\n")
#     file.write(f"nu: {nu}\n")
#     file.write(f"t_steps, t_max: {t_steps}, {t_max}\n")


k_norm = jnp.arange(0.1, 2.0, 0.02)
Lx = jnp.sqrt(2) * jnp.pi * alpha_s[0] / k_norm

initial_params = jnp.array(
    [5.0, 0.0, 1.0, 0.02])  # Initialize parameters (A, B, omega, gamma) with some reasonable guesses
frequency = jnp.zeros_like(Lx)
damping_rate = jnp.zeros_like(Lx)
start_time = time.time()
for j in jnp.arange(len(Lx)):
    Ck, Fk, t = VM_simulation(qs, nu, Omega_cs, alpha_s, mi_me, u_s, Lx[j], Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns, t_max,
                              t_steps)
    dCek = Ck.at[:, 0, 1, 0, 0].set(0)

    # # Minimize the loss function to find the best-fit parameters
    # optimized_result = minimize(lambda params: loss_function(params, t, dCek[:, 0, 0, 0, 0].imag), initial_params, method='BFGS', tol=1e-9)

    # # Extract the best-fit parameters
    # best_fit_params = optimized_result.x
    # A, B, omega, gamma = best_fit_params

    # initial_params = jnp.array([A, B, omega, gamma])

    peaks, _ = find_peaks(jnp.abs(dCek[:100, 0, 0, 0, 0].imag))
    p = jnp.polyfit(t[peaks], jnp.log(jnp.abs(dCek[:100, 0, 0, 0, 0].imag[peaks])), 1)
    # frequency = frequency.at[j].set(omega)
    damping_rate = damping_rate.at[j].set(p[0])
    # k_norm = k_norm.at[j].set(jnp.sqrt(2) * jnp.pi * alpha_s[0] / Lx[j])
end_time = time.time()

print(f"Runtime: {end_time - start_time} seconds")

####################################################################################################################################################

data_damping_rate = pd.read_csv(
    'C:\\Users\\Sreep\\Desktop\\Vlasov-Maxwell_Spectral_Solver-main\\Vlasov-Maxwell_Spectral_Solver-main\\CodeTests\\WolframCodeCSV\\damping_rate_1000000.csv',
    header=None)
damping_rate_theo = jnp.array(data_damping_rate.values)

#data_omega = pd.read_csv('C:\\Users\\Sreep\\Desktop\\Vlasov-Maxwell_Spectral_Solver-main\\Vlasov-Maxwell_Spectral_Solver-main\\CodeTests\\WolframCodeCSV\\omega.csv',header=None)
#omega_theo = jnp.array(data_omega.values)

plt.figure(figsize=(8, 6))
plt.plot(k_norm, damping_rate, label='$\gamma$', linestyle='None', marker='o', color='red', linewidth=3.0)
plt.plot(damping_rate_theo[:, 0], damping_rate_theo[:, 1], label='$\gamma_{theo}$', linestyle='-', color='blue',
         linewidth=3.0)
plt.xlim(0.0, 1.25)
plt.ylim(-1.4, 0.1)
plt.ylabel(r'$\gamma/\omega_{pe}$', fontsize=16)
plt.xlabel(r'$kv_{e,th}/\omega_{pe}$', fontsize=16)
plt.title(rf'$\gamma$ from linear fit, $t_{{max}}\omega_{{pe}} = {t_max}, \nu = {nu}, m_i/m_e = {mi_me}$', fontsize=16)
plt.legend().set_draggable(True)
plt.show()

#plt.figure(figsize=(8, 6))
#plt.plot(k_norm, frequency, label='$\omega$', linestyle='None', marker='o', color='red', linewidth=3.0)
#plt.plot(omega_theo[:, 0], omega_theo[:, 1], label='$\gamma_{theo}$', linestyle='-', color='blue', linewidth=3.0)
#plt.ylabel(r'$\omega/\omega_{pe}$', fontsize=16)
#plt.xlabel(r'$kv_{e,th}/\omega_{pe}$', fontsize=16)
#plt.title(rf'Nonlinear, $t_{{max}}\omega_{{pe}} = {t_max}, \nu = {nu}$', fontsize=16)
#plt.legend().set_draggable(True)
#plt.show()