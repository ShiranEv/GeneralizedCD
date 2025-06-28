# %%
from math import sqrt, pi, cos
from cmath import exp
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
import time
from numpy import zeros, linspace
from qutip import *

# %% functions
def gen_CD(d, alpha, N):
    Z_operator_ancilla = sum([exp(1j * 2 * np.pi * i/d) * basis(d, i) * basis(d, i).dag() for i in range(d)])
    return (alpha * tensor(Z_operator_ancilla, create(N)) - np.conj(alpha) * tensor(Z_operator_ancilla.dag(), destroy(N))).expm()

def gen_state_plus_ancilla(d):
    return ket2dm((sum([basis(d, i) for i in range(d)])).unit())

def change_measurement_basis(d):
    if d == 2:
        # return (-1j * sigmax() * np.pi/4).expm()
        return (sigmaz()+sigmay())/sqrt(2)
    elif d == 4:
        return Qobj(0.5*np.array([
                    [-1, -1j, 1, -1j], 
                    [1j, 1, 1j, -1], 
                    [1, -1j, -1, -1j], 
                    [1j, -1, 1j, 1]]))
    
# %%
# Parameters
N = 100
l = sqrt(2*pi)
beta = l/2
eps = 0.1/(2*sqrt(2))

# %% stabillization simulation

time_CD_ops = 80


stabilizer_z = displace(N, 1j * l)
stabilizer_x = displace(N, l)
stabilizer_y = displace(N, l * (1 + 1j))

expect_2_z = zeros(time_CD_ops, dtype=complex)
expect_2_x = zeros(time_CD_ops, dtype=complex)
expect_2_y = zeros(time_CD_ops, dtype=complex)
expect_4_z = zeros(time_CD_ops, dtype=complex)
expect_4_x = zeros(time_CD_ops, dtype=complex)
expect_4_y = zeros(time_CD_ops, dtype=complex)

for d in [2, 4]:
    state_osc = thermal_dm(N, 0)
    for i in range(time_CD_ops):
        if d == 2:
            if i % 4 == 0:
                # print(i, 'trim q')
                alpha_1 = eps
                alpha_2 = -1j * beta
            if i % 4 == 1:
                # print(i, 'sharpen p')
                alpha_1 = beta
                alpha_2 = -1j * eps * 2
            if i % 4 == 2:
                # print(i, 'trim p')
                alpha_1 = 1j * eps
                alpha_2 = beta
            if i % 4 == 3:
                # print(i, 'sharpen q')
                alpha_1 = -1j * beta
                alpha_2 = -eps * 2
        if d == 4:
            if i % 2 == 0:
                # trim
                alpha_1 = exp(1j*pi/4) * sqrt(2) * (-1j) * eps
                alpha_2 = exp(1j*pi/4) * sqrt(2) * beta
            else: 
                # sharpen
                alpha_1 = exp(1j*pi/4) * sqrt(2) * beta
                alpha_2 = exp(1j*pi/4) * sqrt(2) * (-1j) * eps*2

        state_tot = tensor(gen_state_plus_ancilla(d), state_osc)
        state_tot = gen_CD(d, alpha_1, N) * state_tot * gen_CD(d, alpha_1, N).dag()
        change_basis_tot = tensor(change_measurement_basis(d), qeye(N))
        state_tot = change_basis_tot * state_tot * change_basis_tot.dag()
        state_tot = gen_CD(d, alpha_2, N) * state_tot * gen_CD(d, alpha_2, N).dag()
        state_osc = ptrace(state_tot, 1).unit()
    
        if d == 2:
            expect_2_z[i] = expect(state_osc, stabilizer_z)
            expect_2_x[i] = expect(state_osc, stabilizer_x)
            expect_2_y[i] = expect(state_osc, stabilizer_y)
        if d == 4:
            expect_4_z[i] = expect(state_osc, stabilizer_z)
            expect_4_x[i] = expect(state_osc, stabilizer_x)
            expect_4_y[i] = expect(state_osc, stabilizer_y)

initial_state = state_osc

#%% plot expectation value in the creation
fig = plt.figure(dpi=300)
# plt.plot(abs(expect_2_z), '--', color='tab:orange', label="        ")
# plt.plot(abs(expect_4_z), color='tab:orange', label="        ")
plt.plot(abs(expect_2_x), '--', color='tab:blue', label="        ")
plt.plot(abs(expect_4_x), color='tab:blue', label="        ")
plt.plot(abs(expect_2_y), '--', color='tab:green', label="        ")
plt.plot(abs(expect_4_y), color='tab:green', label="        ")
plt.ylim(0, 1.1)
plt.xlim(0, time_CD_ops - 1)
plt.legend()
plt.show()
# %% qudit noise in stabilization simulation

# take the value of CD num from the stabilization graph, for the same value of S. for example 
# 20 in qudit and 30 in qubit, and plot S vs p

# noise = in measurement or in operation or in initialization
num_p_val = 5
p_val = linspace(0, 1/2, num_p_val)

expect_2_z_noise = zeros(num_p_val, dtype=complex)
expect_2_x_noise = zeros(num_p_val, dtype=complex)
expect_2_y_noise = zeros(num_p_val, dtype=complex)
expect_4_z_noise = zeros(num_p_val, dtype=complex)
expect_4_x_noise = zeros(num_p_val, dtype=complex)
expect_4_y_noise = zeros(num_p_val, dtype=complex)

time_CD_ops_2 = np.where(abs(expect_2_x) >= 0.86)[0][0] + 1
time_CD_ops_4 = np.where(abs(expect_4_x) >= 0.86)[0][0] + 1

time_CD_ops_2 = 20
time_CD_ops_4 = 20

for j, p in enumerate(p_val):
    # qubit 2
    d = 2
    state_osc = initial_state
    for i in range(time_CD_ops_2):
        if i % 4 == 0:
            # print(i, 'trim q')
            alpha_1 = eps
            alpha_2 = -1j * beta
        if i % 4 == 1:
            # print(i, 'sharpen p')
            alpha_1 = beta
            alpha_2 = -1j * eps * 2
        if i % 4 == 2:
            # print(i, 'trim p')
            alpha_1 = 1j * eps
            alpha_2 = beta
        if i % 4 == 3:
            # print(i, 'sharpen q')
            alpha_1 = -1j * beta
            alpha_2 = -eps * 2
        
        state_tot = tensor(gen_state_plus_ancilla(d), state_osc)
        state_tot = gen_CD(d, alpha_1, N) * state_tot * gen_CD(d, alpha_1, N).dag()
        # noise
        state_tot = p * tensor(qeye(d), ptrace(state_tot, 1).unit()) + (1-p) * state_tot

        change_basis_tot = tensor(change_measurement_basis(d), qeye(N))
        state_tot = change_basis_tot * state_tot * change_basis_tot.dag()
        state_tot = gen_CD(d, alpha_2, N) * state_tot * gen_CD(d, alpha_2, N).dag()
        state_osc = ptrace(state_tot, 1).unit()
    
        expect_2_z_noise[j] = expect(state_osc, stabilizer_z)
        expect_2_x_noise[j] = expect(state_osc, stabilizer_x)
        expect_2_y_noise[j] = expect(state_osc, stabilizer_y)

    # qudit 4
    d = 4
    state_osc = initial_state
    for i in range(time_CD_ops_4):
        if i % 2 == 0:
            # trim
            alpha_1 = exp(1j*pi/4) * sqrt(2) * (-1j) * eps
            alpha_2 = exp(1j*pi/4) * sqrt(2) * beta
        else: 
            # sharpen
            alpha_1 = exp(1j*pi/4) * sqrt(2) * beta
            alpha_2 = exp(1j*pi/4) * sqrt(2) * (-1j) * eps*2

        state_tot = tensor(gen_state_plus_ancilla(d), state_osc)
        state_tot = gen_CD(d, alpha_1, N) * state_tot * gen_CD(d, alpha_1, N).dag()
        # noise
        state_tot = p * tensor(qeye(d), ptrace(state_tot, 1).unit()) + (1-p) * state_tot

        change_basis_tot = tensor(change_measurement_basis(d), qeye(N))
        state_tot = change_basis_tot * state_tot * change_basis_tot.dag()
        state_tot = gen_CD(d, alpha_2, N) * state_tot * gen_CD(d, alpha_2, N).dag()
        state_osc = ptrace(state_tot, 1).unit()

        expect_4_z_noise[j] = expect(state_osc, stabilizer_z)
        expect_4_x_noise[j] = expect(state_osc, stabilizer_x)
        expect_4_y_noise[j] = expect(state_osc, stabilizer_y)

#%% plot expectation value vs noise p
fig = plt.figure(dpi=300)
# plt.plot(abs(expect_2_z), '--', color='tab:orange', label="        ")
# plt.plot(abs(expect_4_z), color='tab:orange', label="        ")
plt.plot(p_val, abs(expect_2_x_noise),'--', color='tab:blue', label='2 x')
plt.plot(p_val, abs(expect_4_x_noise), color='tab:blue', label='4 x')
plt.plot(p_val, abs(expect_2_y_noise),'--', color='tab:green', label='2 y')
plt.plot(p_val, abs(expect_4_y_noise), color='tab:green', label='4 y')
plt.ylim(0, 1.1)
plt.legend()
plt.show()

# %%
time_CD_ops = 80
num_p_val = 3
p_val = linspace(0, 0.2, num_p_val)
expect_2_x_noise_x = zeros((time_CD_ops, num_p_val), dtype=complex)
expect_4_x_noise_x = zeros((time_CD_ops, num_p_val), dtype=complex)
expect_2_x_noise_z = zeros((time_CD_ops, num_p_val), dtype=complex)
expect_4_x_noise_z = zeros((time_CD_ops, num_p_val), dtype=complex)

expect_2_z_noise_x = zeros((time_CD_ops, num_p_val), dtype=complex)
expect_4_z_noise_x = zeros((time_CD_ops, num_p_val), dtype=complex)
expect_2_z_noise_z = zeros((time_CD_ops, num_p_val), dtype=complex)
expect_4_z_noise_z = zeros((time_CD_ops, num_p_val), dtype=complex)

for d in [2, 4]:
    for j, p in enumerate(p_val):
        state_osc_z = thermal_dm(N, 0)
        state_osc_x = thermal_dm(N, 0)
        for i in range(time_CD_ops):
            if d == 2:
                if i % 4 == 0:
                    # print(i, 'trim q')
                    alpha_1 = eps
                    alpha_2 = -1j * beta
                if i % 4 == 1:
                    # print(i, 'sharpen p')
                    alpha_1 = beta
                    alpha_2 = -1j * eps * 2
                if i % 4 == 2:
                    # print(i, 'trim p')
                    alpha_1 = 1j * eps
                    alpha_2 = beta
                if i % 4 == 3:
                    # print(i, 'sharpen q')
                    alpha_1 = -1j * beta
                    alpha_2 = -eps * 2
            if d == 4:
                if i % 2 == 0:
                    # trim
                    alpha_1 = exp(1j*pi/4) * sqrt(2) * (-1j) * eps
                    alpha_2 = exp(1j*pi/4) * sqrt(2) * beta
                else: 
                    # sharpen
                    alpha_1 = exp(1j*pi/4) * sqrt(2) * beta
                    alpha_2 = exp(1j*pi/4) * sqrt(2) * (-1j) * eps*2

            state_tot_z = tensor(gen_state_plus_ancilla(d), state_osc_z)
            state_tot_z = gen_CD(d, alpha_1, N) * state_tot_z * gen_CD(d, alpha_1, N).dag()
            state_tot_x = tensor(gen_state_plus_ancilla(d), state_osc_x)
            state_tot_x = gen_CD(d, alpha_1, N) * state_tot_x * gen_CD(d, alpha_1, N).dag()
            # noise
            z_noise = tensor(sum([exp(1j * 2 * np.pi * i/d) * basis(d, i) * basis(d, i).dag() for i in range(d)]), qeye(N))
            x_noise = tensor(sum([basis(d, (i+1)%d) * basis(d, i).dag() for i in range(d)]), qeye(N))
            state_tot_z = p * z_noise * state_tot_z * z_noise.dag() + (1-p) * state_tot_z
            state_tot_x = p * x_noise * state_tot_x * x_noise.dag() + (1-p) * state_tot_x
            change_basis_tot = tensor(change_measurement_basis(d), qeye(N))
            state_osc_z = ptrace(gen_CD(d, alpha_2, N) * change_basis_tot * state_tot_z * change_basis_tot.dag() * gen_CD(d, alpha_2, N).dag(), 1).unit()
            state_osc_x = ptrace(gen_CD(d, alpha_2, N) * change_basis_tot * state_tot_x * change_basis_tot.dag() * gen_CD(d, alpha_2, N).dag(), 1).unit()
            if d == 2:
                expect_2_x_noise_z[i,j] = expect(state_osc_z, stabilizer_x)
                expect_2_x_noise_x[i,j] = expect(state_osc_x, stabilizer_x)

                expect_2_z_noise_z[i,j] = expect(state_osc_z, stabilizer_z)
                expect_2_z_noise_x[i,j] = expect(state_osc_x, stabilizer_z)
            if d == 4:
                expect_4_x_noise_z[i,j] = expect(state_osc_z, stabilizer_x)
                expect_4_x_noise_x[i,j] = expect(state_osc_x, stabilizer_x)

                expect_4_z_noise_z[i,j] = expect(state_osc_z, stabilizer_z)
                expect_4_z_noise_x[i,j] = expect(state_osc_x, stabilizer_z)


#%% plot expectation value in the creation
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
fig = plt.figure(dpi=300)
for j, p in enumerate(p_val):
    plt.plot(abs(expect_2_x_noise_z[:,j]), '--', color=colors[j], label="z, p=" + str(round(p, 2)))
    plt.plot(abs(expect_4_x_noise_z[:,j]), color=colors[j], label="z, p=" + str(round(p, 2)))
    plt.plot(abs(expect_2_x_noise_x[:,j]), '--', color=colors[j+3], label="x, p=" + str(round(p, 2)))
    plt.plot(abs(expect_4_x_noise_x[:,j]), color=colors[j+3], label="x, p=" + str(round(p, 2)))
plt.ylim(0, 1.1)
plt.xlim(0, time_CD_ops - 1)
plt.legend()
plt.show()

fig = plt.figure(dpi=300)
for j, p in enumerate(p_val):
    plt.plot(abs(expect_2_z_noise_z[:,j]), '--', color=colors[j], label="z, p=" + str(round(p, 2)))
    plt.plot(abs(expect_4_z_noise_z[:,j]), color=colors[j], label="z, p=" + str(round(p, 2)))
    plt.plot(abs(expect_2_z_noise_x[:,j]), '--', color=colors[j+3], label="x, p=" + str(round(p, 2)))
    plt.plot(abs(expect_4_z_noise_x[:,j]), color=colors[j+3], label="x, p=" + str(round(p, 2)))
plt.ylim(0, 1.1)
plt.xlim(0, time_CD_ops - 1)
plt.legend()
plt.show()

fig = plt.figure(dpi=300)
plt.plot(p_val, abs(expect_2_x_noise_z[-1,:]), '--', color=colors[0], label="Sx, z, 2")
plt.plot(p_val, abs(expect_4_x_noise_z[-1,:]), color=colors[0], label="Sx, z, 4")
plt.plot(p_val, abs(expect_2_x_noise_x[-1,:]), '--', color=colors[1], label="Sx, x, 2")
plt.plot(p_val, abs(expect_4_x_noise_x[-1,:]), color=colors[1], label="Sx, x, 4")

plt.plot(p_val, abs(expect_2_z_noise_z[-1,:]), '--', color=colors[0], label="Sz, z, 2")
plt.plot(p_val, abs(expect_4_z_noise_z[-1,:]), color=colors[0], label="Sz, z, 4")
plt.plot(p_val, abs(expect_2_z_noise_x[-1,:]), '--', color=colors[1], label="Sz, x, 2")
plt.plot(p_val, abs(expect_4_z_noise_x[-1,:]), color=colors[1], label="Sz, x, 4")
plt.ylim(0, 1.1)
plt.legend()
plt.show()

# %% fig 4
def coherent_t_p(thet, ph, N):
    N_k = N - 1
    return sum([sqrt(np.math.comb(N_k, k)) * (cos(thet / 2) ** (N_k - k)) * (
            (np.exp(1j * ph) * sin(thet / 2)) ** k) * basis(N, k) for k in range(N)]).unit()


g_all = 4 / (sqrt(2))

N_cat = 40
alpha = 4
state_cat = (coherent(N_cat, alpha) + coherent(N_cat, 1j * alpha) + coherent(N_cat, -alpha) + coherent(N_cat,
                                                                                                       -1j * alpha)).unit()
operator_cat = destroy(N_cat)
g_cat = g_all / alpha
# plt.bar(range(N_cat),abs(state_cat[:,0][:,0])**2)


N_cat_bad = 25
alpha_bad = 3
state_cat_bad = (coherent(N_cat_bad, alpha_bad) + coherent(N_cat_bad, 1j * alpha_bad) + coherent(N_cat_bad,
                                                                                                 -alpha_bad) + coherent(
    N_cat_bad, -1j * alpha_bad)).unit()
operator_cat_bad = destroy(N_cat_bad)
g_cat_bad = g_all / alpha
# plt.bar(range(N_cat_bad),abs(state_cat_bad[:,0][:,0])**2)


d = 4
state_qudit = (basis(d, 0) + basis(d, 1) + basis(d, 2) + basis(d, 3)).unit()
operator_qudit = basis(d, 0) * basis(d, 0).dag() + 1j * basis(d, 1) * basis(d, 1).dag() - basis(d, 2) * basis(d,
                                                                                                              2).dag() - 1j * basis(
    d, 3) * basis(d, 3).dag()
g_qudit = g_all

N_spin = 200
theta = pi / 2
state_spin = (coherent_t_p(theta, 0, N_spin) + coherent_t_p(theta, pi, N_spin) + coherent_t_p(theta, pi / 2,
                                                                                              N_spin) + coherent_t_p(
    theta, 3 * pi / 2, N_spin)).unit()
s = (N_spin - 1) / 2
operator_spin = jmat(s, '+')
g_spin = g_all / s

N = 40
sigma = 4
m = 20
state_pb = sum([exp(-(n - m) ** 2 / (4 * sigma)) * basis(N, n) for n in range(N)]).unit()
operator_pb = sum([exp(1j * 2 * pi * s / d) * basis(N, s) * basis(N, s).dag() for s in range(N)])
g_pb = g_all
# plt.bar(range(N),abs(state_pb[:,0][:,0])**2)

sum([exp(-(n - m) ** 2 / (4 * sigma)) * basis(N, n) for n in range(N)]).norm()

N = 40
sigma = 1
m = 30
state_pb_bad = sum([exp(-(n - m) ** 2 / (4 * sigma)) * basis(N, n) for n in range(N)]).unit()
operator_pb = sum([exp(1j * 2 * pi * s / d) * basis(N, s) * basis(N, s).dag() for s in range(N)])
g_pb = g_all
# plt.bar(range(N),abs(state_pb_bad[:,0][:,0])**2)
# variance(num(N), state_pb_bad)


state_ancilla = [state_qudit, state_pb, state_cat, state_spin, state_pb_bad, state_cat_bad]
operator_ancilla = [operator_qudit, operator_pb, operator_cat, operator_spin, operator_pb, operator_cat_bad]
g = [g_qudit, g_pb, g_cat, g_spin, g_pb, g_cat_bad]
titles = ["qudit", "rotor good", "cat good", "spin good", "rotor bad", "cat bad", "spin bad"]

fid = np.ones(len(state_ancilla))
N_osc = 20
for i in range(len(state_ancilla)):
    operator = (g[i] * (
            tensor(operator_ancilla[i], create(N_osc)) + tensor(operator_ancilla[i].dag(), destroy(N_osc)))).expm()
    state_tot = tensor(state_ancilla[i], basis(N_osc, 0))
    rho = ptrace(tensor(state_ancilla[i] * state_ancilla[i].dag(), qeye(N_osc)) * operator * state_tot, 1).unit()
    # plt.bar(range(N_osc),abs(rho[:,0][:,0])**2)
    cat_res = (coherent(N_osc, g_all) + coherent(N_osc, 1j * g_all) + coherent(N_osc, -g_all) + coherent(N_osc,
                                                                                                         -1j * g_all)).unit()
    fid[i] = fidelity(ket2dm(cat_res), rho)
    vec = linspace(-2 * g_all, 2 * g_all, 200)
    W = wigner(rho, vec, vec, method='clenshaw')
    wlim = abs(W).max()

    fig, ax = plt.subplots(1, 1, dpi=300, figsize=(7.2, 6))
    cf = ax.contourf(vec, vec, W, 100, norm=Normalize(-wlim, wlim), cmap=cmap)
    ax.grid()
    fig.colorbar(cf, ax=ax)
    ax.set_title(titles[i], fontsize=12)
    plt.show()
