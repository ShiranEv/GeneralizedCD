# %%
# Quantum Stabilization and Noise Simulation with Qudits
# Author: Shiran Even Haim
# Description: Simulates CD stabilization using qubits (d=2) and qudits (d=4), and analyzes robustness to noise.

# %% Imports
from math import sqrt, pi, cos, sin
from cmath import exp
import matplotlib.pyplot as plt
import numpy as np
from numpy import zeros, linspace
from qutip import *

# %% Parameters
N = 100  # Hilbert space size of oscillator
l = sqrt(2 * pi)
beta = l / 2
eps = 0.1 / (2 * sqrt(2))

# Define stabilizers
stabilizer_z = displace(N, 1j * l)
stabilizer_x = displace(N, l)
stabilizer_y = displace(N, l * (1 + 1j))
# %% Helper Functions
def Z_operator_ancilla(d):
    return sum([exp(1j * 2 * pi * i / d) * basis(d, i) * basis(d, i).dag() for i in range(d)])

def X_operator_ancilla(d):
    return sum([basis(d, (i + 1) % d) * basis(d, i).dag() for i in range(d)])

def gen_CD(d, alpha, N):
    return (alpha * tensor(Z_operator_ancilla(d), create(N)) -
            np.conj(alpha) * tensor(Z_operator_ancilla(d).dag(), destroy(N))).expm()

def gen_state_plus_ancilla(d):
    return ket2dm(sum([basis(d, i) for i in range(d)]).unit())

def change_measurement_basis(d, N):
    if d == 2:
        change_measurement_basis_op = (sigmaz() + sigmay()) / sqrt(2)
    elif d == 4:
        change_measurement_basis_op = Qobj(0.5 * np.array([
            [-1, -1j, 1, -1j], 
            [1j, 1, 1j, -1], 
            [1, -1j, -1, -1j], 
            [1j, -1, 1j, 1]
        ]))
    return tensor(change_measurement_basis_op, qeye(N))

def stabilization_step(d, i, state_osc, N, ancilla_noise=None, oscillator_noise=None, p=0, k_a=0, k_phi=0):
    if d == 2:
        if i % 4 == 0: 
            # sharpen q
            alpha_1 = -1j * beta
            alpha_2 = eps * 2
        elif i % 4 == 1: 
            # trim q
            alpha_1 = eps
            alpha_2 = -1j * beta
        elif i % 4 == 2: 
            # sharpen p
            alpha_1 = beta
            alpha_2 = 1j * eps * 2
        elif i % 4 == 3: 
            # trim p
            alpha_1 = 1j * eps
            alpha_2 = beta
    elif d == 4:
        phase = exp(1j * pi / 4) * sqrt(2)
        if i % 2 == 0: 
            # sharpen
            alpha_1 = phase * beta
            alpha_2 = phase * (-1j) * eps * 2
        else: 
            # trim
            alpha_1 = phase * (-1j) * eps
            alpha_2 = phase * beta
    
    # Apply the first CD operation
    state_tot = tensor(gen_state_plus_ancilla(d), state_osc)

    # Apply ancilla noise
    if ancilla_noise == 'Z':
        z_noise = tensor(Z_operator_ancilla(d), qeye(N))
        state_tot = p * z_noise * state_tot * z_noise.dag() + (1 - p) * state_tot
    elif ancilla_noise == 'X':
        x_noise = tensor(X_operator_ancilla(d), qeye(N))
        state_tot = p * x_noise * state_tot * x_noise.dag() + (1 - p) * state_tot
    elif ancilla_noise == 'D':
        state_tot = p * tensor(qeye(d)/d, ptrace(state_tot, 1).unit()) + (1 - p) * state_tot

    state_tot = gen_CD(d, alpha_1, N) * state_tot * gen_CD(d, alpha_1, N).dag()
    
    # Apply ancilla noise
    if ancilla_noise == 'Z':
        z_noise = tensor(Z_operator_ancilla(d), qeye(N))
        state_tot = p * z_noise * state_tot * z_noise.dag() + (1 - p) * state_tot
    elif ancilla_noise == 'X':
        x_noise = tensor(X_operator_ancilla(d), qeye(N))
        state_tot = p * x_noise * state_tot * x_noise.dag() + (1 - p) * state_tot
    elif ancilla_noise == 'D':
        state_tot = p * tensor(qeye(d)/d, ptrace(state_tot, 1).unit()) + (1 - p) * state_tot

    # Change measurement basis
    state_tot = change_measurement_basis(d, N) * state_tot * change_measurement_basis(d, N).dag()
    state_tot = gen_CD(d, alpha_2, N) * state_tot * gen_CD(d, alpha_2, N).dag()
    # Partial trace to get the oscillator state
    state_osc = ptrace(state_tot, 1).unit()

    # Apply noise to the oscillator state
    if oscillator_noise != None:
        state_osc = mesolve(qeye(N), state_osc, [0, 1], [k_a * destroy(N), k_phi * num(N)], []).states[-1]

    return state_osc


# %% Main CD Stabilization Loop
time_CD_ops = 79  # Number of CD operations

# Initialize expectation arrays
expect_2_z = zeros(time_CD_ops, dtype=complex)
expect_2_x = zeros(time_CD_ops, dtype=complex)
expect_2_y = zeros(time_CD_ops, dtype=complex)
expect_4_z = zeros(time_CD_ops, dtype=complex)
expect_4_x = zeros(time_CD_ops, dtype=complex)
expect_4_y = zeros(time_CD_ops, dtype=complex)


for d in [2, 4]:  # d = 2 (qubit), d = 4 (qudit)
    state_osc = thermal_dm(N, 0)
    for i in range(time_CD_ops):
        state_osc = stabilization_step(d, i, state_osc, N)

        # Save expectation values
        if d == 2:
            expect_2_z[i] = expect(state_osc, stabilizer_z)
            expect_2_x[i] = expect(state_osc, stabilizer_x)
            expect_2_y[i] = expect(state_osc, stabilizer_y)
        elif d == 4:
            expect_4_z[i] = expect(state_osc, stabilizer_z)
            expect_4_x[i] = expect(state_osc, stabilizer_x)
            expect_4_y[i] = expect(state_osc, stabilizer_y)

    # Save the last oscillator state
    if d == 2:
        stabilized_state_2 = state_osc
    elif d == 4:
        stabilized_state_4 = state_osc

# %% Fig 1 - Plot Expectation Values over Time
fig = plt.figure(dpi=300)
# plt.plot(abs(expect_2_z), '--', color='tab:orange', label="Qubit z")
# plt.plot(abs(expect_4_z), color='tab:orange', label="Qudit z")
plt.plot(abs(expect_2_x), '--', color='tab:blue', label="Qubit X")
plt.plot(abs(expect_4_x), color='tab:blue', label="Qudit X")
plt.plot(abs(expect_2_y), '--', color='tab:green', label="Qubit Y")
plt.plot(abs(expect_4_y), color='tab:green', label="Qudit Y")
plt.ylim(0, 1.1)
plt.xlim(0, time_CD_ops - 1)
plt.legend()
plt.show()

# %% Oscillator noise simulation
k_list = linspace(0, 0.1, 11)
num_k = len(k_list)
noise_time = 42

expect_2_noise_x = zeros(num_k, dtype=complex)
expect_4_noise_x = zeros(num_k, dtype=complex)
expect_2_noise_y = zeros(num_k, dtype=complex)
expect_4_noise_y = zeros(num_k, dtype=complex)
# no error correction
expect_noise_x_no = zeros(num_k, dtype=complex)
expect_noise_y_no = zeros(num_k, dtype=complex)

k_a = 0
k_phi = 0

# for i, k_a in enumerate(k_list):  # photon loss
for j, k_phi in enumerate(k_list):  # dephasing
    state_osc_2 = stabilized_state_2
    state_osc_4 = stabilized_state_4
    state_osc_no_correction = stabilized_state_2  # Use the last state from d=2 as the initial state for no correction
    for i in range(noise_time):
        state_osc_2 = stabilization_step(2, i, state_osc_2, N, oscillator_noise=True, k_a=k_a, k_phi=k_phi)
        state_osc_4 = stabilization_step(4, i, state_osc_4, N, oscillator_noise=True, k_a=k_a, k_phi=k_phi)
        state_osc_no_correction = mesolve(qeye(N), state_osc_no_correction, [0, 1], [k_a * destroy(N), k_phi * num(N)], []).states[-1]
    # Save expectation values
    expect_2_noise_x[j] = expect(state_osc_2, stabilizer_x)
    expect_2_noise_y[j] = expect(state_osc_2, stabilizer_y)
    expect_4_noise_x[j] = expect(state_osc_4, stabilizer_x)
    expect_4_noise_y[j] = expect(state_osc_4, stabilizer_y)
    expect_noise_x_no[j] = expect(state_osc_no_correction, stabilizer_x)
    expect_noise_y_no[j] = expect(state_osc_no_correction, stabilizer_y)

# %% Fig 3 - Plot Expectation Values with Oscillator Noise
fig = plt.figure(dpi=300)

plt.plot(k_list, abs(expect_2_noise_x), '--', color='tab:blue', label="        ")
plt.plot(k_list, abs(expect_4_noise_x), color='tab:blue', label="        ")
plt.plot(k_list, abs(expect_noise_x_no), linestyle='dotted', color='tab:blue', label="        ")
plt.plot(k_list, abs(expect_2_noise_y), '--', color='tab:green', label="        ")
plt.plot(k_list, abs(expect_4_noise_y), color='tab:green', label="        ")
plt.plot(k_list, abs(expect_noise_y_no), linestyle='dotted', color='tab:green', label="        ")
plt.legend()
plt.ylim(0, 1.1)
plt.xlim(0, 0.101)
plt.show()

# %% Ancilla Noise Simulation - X, Z and Depolarizing Noise
p_list = linspace(0, 0.5, 11)
num_p = len(p_list)
noise_time = 20

expect_2_noise_Z_x = zeros(num_p, dtype=complex)
expect_4_noise_Z_x = zeros(num_p, dtype=complex)
expect_2_noise_Z_y = zeros(num_p, dtype=complex)
expect_4_noise_Z_y = zeros(num_p, dtype=complex)

expect_2_noise_X_x = zeros(num_p, dtype=complex)
expect_4_noise_X_x = zeros(num_p, dtype=complex)
expect_2_noise_X_y = zeros(num_p, dtype=complex)
expect_4_noise_X_y = zeros(num_p, dtype=complex)

expect_2_noise_D_x = zeros(num_p, dtype=complex)
expect_4_noise_D_x = zeros(num_p, dtype=complex)
expect_2_noise_D_y = zeros(num_p, dtype=complex)
expect_4_noise_D_y = zeros(num_p, dtype=complex)

for d in [2, 4]:
    for j, p in enumerate(p_list): 
        print(f"Processing dimension {d}, p={p:.2f}")
        state_osc_Z = stabilized_state_2 if d == 2 else stabilized_state_4
        state_osc_X = stabilized_state_2 if d == 2 else stabilized_state_4
        state_osc_D = stabilized_state_2 if d == 2 else stabilized_state_4
        # Initialize the state for the current dimension
        for i in range(noise_time):
            state_osc_Z = stabilization_step(d, i, state_osc_Z, N, ancilla_noise="Z", p=p)
            state_osc_X = stabilization_step(d, i, state_osc_X, N, ancilla_noise="X", p=p)
            state_osc_D = stabilization_step(d, i, state_osc_D, N, ancilla_noise="D", p=p)

        # Save expectation values
        if d == 2:
            expect_2_noise_Z_x[j] = expect(state_osc_Z, stabilizer_x)
            expect_2_noise_Z_y[j] = expect(state_osc_Z, stabilizer_y)
            expect_2_noise_X_x[j] = expect(state_osc_X, stabilizer_x)
            expect_2_noise_X_y[j] = expect(state_osc_X, stabilizer_y)
            expect_2_noise_D_x[j] = expect(state_osc_D, stabilizer_x)
            expect_2_noise_D_y[j] = expect(state_osc_D, stabilizer_y)
        else:
            expect_4_noise_Z_x[j] = expect(state_osc_Z, stabilizer_x)
            expect_4_noise_Z_y[j] = expect(state_osc_Z, stabilizer_y)
            expect_4_noise_X_x[j] = expect(state_osc_X, stabilizer_x)
            expect_4_noise_X_y[j] = expect(state_osc_X, stabilizer_y)
            expect_4_noise_D_x[j] = expect(state_osc_D, stabilizer_x)
            expect_4_noise_D_y[j] = expect(state_osc_D, stabilizer_y)

# %% Plot Expectation Values with Ancilla Noise
fig = plt.figure(dpi=300)
plt.plot(p_list, abs(expect_2_noise_Z_x), '--', color='tab:blue', label="        ")
plt.plot(p_list, abs(expect_4_noise_Z_x), color='tab:blue', label="        ")
plt.plot(p_list, abs(expect_2_noise_Z_y), '--', color='tab:green', label="        ")
plt.plot(p_list, abs(expect_4_noise_Z_y), color='tab:green', label="        ")
plt.ylim(0, 1.1)
plt.xlim(0, 0.5)
plt.xticks([0, 0.1, 0.2, 0.3, 0.4])
plt.show()
# %% Plot Expectation Values with Ancilla Noise
fig = plt.figure(dpi=300)
plt.plot(p_list, abs(expect_2_noise_X_x), '--', color='tab:blue', label="        ")
plt.plot(p_list, abs(expect_4_noise_X_x), color='tab:blue', label="        ")
plt.plot(p_list, abs(expect_2_noise_X_y), '--', color='tab:green', label="        ")
plt.plot(p_list, abs(expect_4_noise_X_y), color='tab:green', label="        ")
plt.ylim(0, 1.1)
plt.xlim(0, 0.5)
plt.xticks([0, 0.1, 0.2, 0.3, 0.4])
plt.yticks([])
plt.show()
# %% Plot Expectation Values with Ancilla Noise
fig = plt.figure(dpi=300)
plt.plot(p_list, abs(expect_2_noise_D_x), '--', color='tab:blue', label="        ")
plt.plot(p_list, abs(expect_4_noise_D_x), color='tab:blue', label="        ")
plt.plot(p_list, abs(expect_2_noise_D_y), '--', color='tab:green', label="        ")
plt.plot(p_list, abs(expect_4_noise_D_y), color='tab:green', label="        ")
plt.legend()
plt.ylim(0, 1.1)
plt.xlim(0, 0.5)
plt.show()

# %% Section 4
g_all = 4 / (sqrt(2))
d = 4
# Ancilla States
state_qudit = sum([basis(d, i) for i in range(d)]).unit()
N_osc = 20
qudit_ancilla_state = ptrace(tensor(state_qudit * state_qudit.dag(), qeye(N_osc)) * gen_CD(d, g_all, N_osc) * tensor(state_qudit, basis(N_osc, 0)) , 1).unit()
# plot_wigner(qudit_ancilla_state)
# plt.show()
# plot_fock_distribution(qudit_ancilla_state)
# plt.show()
res_state = sum([displace(N_osc, g_all * exp(1j * 2 * pi * i / d)) * basis(N_osc, 0) for i in range(d)]).unit()
# plot_fock_distribution(res_state)
# plt.show()
print(f"Fidelity for qudit ancilla: {fidelity(qudit_ancilla_state, res_state)}")



# Cat ancilla
N_cat = 40
alpha = 4
state_cat = sum([displace(N_cat, alpha * exp(1j * 2 * pi * i / d)) * basis(N_cat, 0) for i in range(d)]).unit()
operator_cat = destroy(N_cat)
g_cat = g_all / alpha

alpha_bad = 3
state_cat_bad = sum([displace(N_cat, alpha_bad * exp(1j * 2 * pi * i / d)) * basis(N_cat, 0) for i in range(d)]).unit()
g_cat_bad = g_all / alpha_bad

# Spin ancilla
s = 25
state_spin = sum([spin_coherent(s, pi / 2, 2 * pi * i / d)  for i in range(d)]).unit()
operator_spin = jmat(s, '-')
g_spin = g_all / s
# plot_wigner(state_spin)
# plt.show()

# Spin ancilla bad
s = 19/2
state_spin_bad = sum([spin_coherent(s, pi / 2, 2 * pi * i / d)  for i in range(d)]).unit()
operator_spin_bad = jmat(s, '-')
g_spin_bad = g_all / s
# plot_wigner(state_spin_bad)
# plt.show()

# PB ancilla
N_pb = 40
sigma = 4
m = 20
state_pb = sum([exp(-(n - m) ** 2 / (4 * sigma)) * basis(N_pb, n) for n in range(N_pb)]).unit()
operator_pb = sum([exp(1j * 2 * pi * s / d) * basis(N_pb, s) * basis(N_pb, s).dag() for s in range(N_pb)])
g_pb = g_all

# PB ancilla bad
sigma = 1
m = 30
state_pb_bad = sum([exp(-(n - m) ** 2 / (4 * sigma)) * basis(N_pb, n) for n in range(N_pb)]).unit()

state_ancilla = [state_pb, state_cat, state_spin, state_pb_bad, state_cat_bad, state_spin_bad]
operator_ancilla = [operator_pb, operator_cat, operator_spin, operator_pb, operator_cat, operator_spin_bad]
g = [g_pb, g_cat, g_spin, g_pb, g_cat_bad, g_spin_bad]
titles = ["rotor good", "cat good", "spin good", "rotor bad", "cat bad", "spin bad"]
for i in range(len(state_ancilla)):
    operator = (g[i] * tensor(operator_ancilla[i], create(N_osc)) - np.conj(g[i]) * tensor(operator_ancilla[i].dag(), destroy(N_osc))).expm() 
    state_tot = tensor(state_ancilla[i], basis(N_osc, 0))
    rho = ptrace(tensor(state_ancilla[i] * state_ancilla[i].dag(), qeye(N_osc)) * operator * state_tot, 1).unit()
    print(f"Fidelity for {titles[i]}: {fidelity(res_state, rho)}")
    # plot_wigner(rho)
    # plt.show()
# %%
