"""
Coupled rotating Boussinesq Atmosphere-Ocean model -- pseudo-spectral solver
=============================================================================

Equations solved (2D, doubly periodic domain [0,L]x[0,L]):

Atmosphere (velocity u_a, temperature theta_a):

    d u_a/dt + (u_a . grad) u_a =
        - (1/Roa) u_a^perp - (1/Roa) grad(theta_a)
        + gamma * (u_o - mean_removed_solenoidal(u_a))
        + (1/Rea) Laplacian(u_a)  -  grad(p_a)        [p_a enforces div(u_a)=0]

    d theta_a/dt + (u_a . grad) theta_a =
        gamma * (theta_o - theta_a) + (1/Pea) Laplacian(theta_a)

Ocean (velocity u_o, temperature theta_o):

    d u_o/dt + (u_o . grad) u_o =
        - (1/Roo) u_o^perp
        + sigma * (mean_removed_solenoidal(u_a) - u_o)
        + (1/Reo) Laplacian(u_o)  -  grad(p_o + q_a)  [p_o enforces div(u_o)=0]

    d theta_o/dt + (u_o . grad) theta_o =
        sigma * (theta_a - theta_o) + (1/Peo) Laplacian(theta_o)

Helmholtz decomposition of the domain-mean-removed atmospheric velocity:

    bar(u_a) := u_a - <u_a>_D  =  bar(u_a)^sol + grad(q_a)

Here "^perp" rotates a vector by +90 degrees: (u,v)^perp = (-v, u).
gamma, sigma < 0 are coupling constants.

NOTE ON RECONSTRUCTION
-----------------------
The equations as pasted by the user were corrupted by a bad PDF->text copy
(line breaks and symbols scrambled). The system above is the physically
consistent reconstruction (rotating shallow-water / QG-type Boussinesq
atmosphere-ocean coupling with Rossby/Reynolds/Peclet numbers and a
solenoidal-velocity + temperature nudging coupling, as used in idealised
coupled atmosphere-ocean toy models). Edit `rhs_ua`, `rhs_theta_a`,
`rhs_uo`, `rhs_theta_o` below if your source has different terms --
the numerical machinery (spectral derivatives, dealiasing, Leray
projection, RK4 stepping) does not need to change.

Because grad(p_a), grad(p_o+q_a) are pure gradients, they are removed
automatically by the incompressibility (Leray) projection applied to the
full right-hand side of each momentum equation; q_a therefore has no
dynamical effect on u_o and is only reconstructed for diagnostics.

Method
------
- Fourier pseudo-spectral in space (periodic BCs), 2/3-rule dealiasing.
- Classical explicit RK4 in time.
- Incompressibility enforced each RK stage via spectral Leray projection.
"""

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Parameters (edit as needed)
# ----------------------------------------------------------------------
N          = 128          # grid points per direction
L          = 2 * np.pi    # domain size (periodic box [0,L]^2)

Roa, Roo   = 0.5, 0.5      # Rossby numbers (atmosphere, ocean)
Rea, Reo   = 2.0e3, 2.0e3  # Reynolds numbers
Pea, Peo   = 2.0e3, 2.0e3  # Peclet numbers
gamma      = -0.5          # atmosphere<-ocean temperature coupling (gamma<0)
sigma      = -0.5          # ocean<-atmosphere velocity coupling (sigma<0)

dt         = 2.0e-3
n_steps    = 4000
save_every = 200

rng = np.random.default_rng(0)

# ----------------------------------------------------------------------
# Spectral grid
# ----------------------------------------------------------------------
x = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, x, indexing="ij")

k1d = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
KX, KY = np.meshgrid(k1d, k1d, indexing="ij")
K2 = KX**2 + KY**2
K2_safe = K2.copy()
K2_safe[0, 0] = 1.0  # avoid division by zero at the mean mode

# 2/3-rule dealiasing mask
kmax = k1d.max()
DEALIAS = (np.abs(KX) < (2.0 / 3.0) * kmax) & (np.abs(KY) < (2.0 / 3.0) * kmax)


def fft2(f):
    return np.fft.fft2(f)


def ifft2(fhat):
    return np.real(np.fft.ifft2(fhat))


def grad_phys(fhat):
    """Physical-space gradient of a scalar field given in Fourier space."""
    fx = ifft2(1j * KX * fhat)
    fy = ifft2(1j * KY * fhat)
    return fx, fy


def laplacian_hat(fhat):
    return -K2 * fhat


def leray_project(u1hat, u2hat):
    """Project a vector field (given in Fourier space) onto its
    divergence-free part (removes any gradient component)."""
    div = KX * u1hat + KY * u2hat
    u1p = u1hat - KX * div / K2_safe
    u2p = u2hat - KY * div / K2_safe
    u1p[0, 0] = u1hat[0, 0]
    u2p[0, 0] = u2hat[0, 0]
    return u1p, u2p


def dealias(fhat):
    out = fhat.copy()
    out[~DEALIAS] = 0.0
    return out


def advect(u1, u2, u1hat, u2hat, fhat):
    """Pseudo-spectral evaluation of (u . grad) f, f given by fhat,
    velocity given both in physical space (u1,u2) and Fourier (u1hat,u2hat).
    Returns the result in Fourier space, dealiased."""
    fx, fy = grad_phys(fhat)
    prod = u1 * fx + u2 * fy
    return dealias(fft2(prod))


def vector_advect(u1, u2, u1hat, u2hat):
    """(u . grad) u for a vector field; returns (adv1hat, adv2hat)."""
    adv1 = advect(u1, u2, u1hat, u2hat, u1hat)
    adv2 = advect(u1, u2, u1hat, u2hat, u2hat)
    return adv1, adv2


def mean_removed_solenoidal(u1hat, u2hat):
    """bar(u)^sol : remove domain mean, then Leray-project (periodic domain
    => solenoidal part of Helmholtz decomposition of the mean-removed field)."""
    b1, b2 = u1hat.copy(), u2hat.copy()
    b1[0, 0] = 0.0
    b2[0, 0] = 0.0
    return leray_project(b1, b2)


def irrotational_potential_hat(u1hat, u2hat):
    """q_a such that bar(u_a) = bar(u_a)^sol + grad(q_a)  (diagnostic only)."""
    b1, b2 = u1hat.copy(), u2hat.copy()
    b1[0, 0] = 0.0
    b2[0, 0] = 0.0
    div = KX * b1 + KY * b2  # = Laplacian(q_a) in Fourier space is -K2*qhat
    qhat = np.zeros_like(div)
    mask = K2 > 0
    qhat[mask] = (1j * div[mask]) / K2[mask]  # grad(q)=b_irrot => i k qhat = ...
    return qhat


# ----------------------------------------------------------------------
# Right-hand sides
# ----------------------------------------------------------------------
def rhs(state):
    ua1h, ua2h, thah, uo1h, uo2h, thoh = state

    ua1, ua2 = ifft2(ua1h), ifft2(ua2h)
    uo1, uo2 = ifft2(uo1h), ifft2(uo2h)

    # advection
    adv_ua1, adv_ua2 = vector_advect(ua1, ua2, ua1h, ua2h)
    adv_uo1, adv_uo2 = vector_advect(uo1, uo2, uo1h, uo2h)
    adv_tha = advect(ua1, ua2, ua1h, ua2h, thah)
    adv_tho = advect(uo1, uo2, uo1h, uo2h, thoh)

    # coupling term: domain-mean-removed solenoidal part of u_a
    ua_sol1h, ua_sol2h = mean_removed_solenoidal(ua1h, ua2h)

    # Coriolis-like perp terms: u^perp = (-u2, u1)
    ua1_perp_h, ua2_perp_h = -ua2h, ua1h
    uo1_perp_h, uo2_perp_h = -uo2h, uo1h

    # grad(theta_a)
    gth_a1h = 1j * KX * thah
    gth_a2h = 1j * KY * thah

    # --- atmosphere momentum ---
    rhs_ua1 = (-adv_ua1 - (1.0 / Roa) * ua1_perp_h - (1.0 / Roa) * gth_a1h
               + gamma * (uo1h - ua_sol1h) - (1.0 / Rea) * K2 * ua1h)
    rhs_ua2 = (-adv_ua2 - (1.0 / Roa) * ua2_perp_h - (1.0 / Roa) * gth_a2h
               + gamma * (uo2h - ua_sol2h) - (1.0 / Rea) * K2 * ua2h)
    rhs_ua1, rhs_ua2 = leray_project(rhs_ua1, rhs_ua2)

    # --- atmosphere temperature ---
    rhs_tha = -adv_tha + gamma * (thoh - thah) - (1.0 / Pea) * K2 * thah

    # --- ocean momentum ---
    rhs_uo1 = (-adv_uo1 - (1.0 / Roo) * uo1_perp_h
               + sigma * (ua_sol1h - uo1h) - (1.0 / Reo) * K2 * uo1h)
    rhs_uo2 = (-adv_uo2 - (1.0 / Roo) * uo2_perp_h
               + sigma * (ua_sol2h - uo2h) - (1.0 / Reo) * K2 * uo2h)
    rhs_uo1, rhs_uo2 = leray_project(rhs_uo1, rhs_uo2)

    # --- ocean temperature ---
    rhs_tho = -adv_tho + sigma * (thah - thoh) - (1.0 / Peo) * K2 * thoh

    return np.array([rhs_ua1, rhs_ua2, rhs_tha, rhs_uo1, rhs_uo2, rhs_tho])


def rk4_step(state, dt):
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ----------------------------------------------------------------------
# Initial conditions: a few Gaussian vortices / blobs (edit freely, or
# load your own ua0, theta_a0, uo0, theta_o0 arrays of shape (N,N))
# ----------------------------------------------------------------------
def gaussian_vortex(x0, y0, amp, sigma_v):
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    psi = amp * np.exp(-r2 / (2 * sigma_v ** 2))
    # velocity from streamfunction: u = -dpsi/dy, v = dpsi/dx
    psih = fft2(psi)
    u1 = -ifft2(1j * KY * psih)
    u2 = ifft2(1j * KX * psih)
    return u1, u2


ua1_0, ua2_0 = gaussian_vortex(L * 0.35, L * 0.5, amp=3.0, sigma_v=L * 0.08)
uo1_0, uo2_0 = gaussian_vortex(L * 0.65, L * 0.5, amp=-2.0, sigma_v=L * 0.1)

theta_a0 = 1.0 * np.exp(-((X - L * 0.3) ** 2 + (Y - L * 0.6) ** 2) / (2 * (L * 0.1) ** 2))
theta_o0 = -1.0 * np.exp(-((X - L * 0.7) ** 2 + (Y - L * 0.4) ** 2) / (2 * (L * 0.12) ** 2))

state = np.array([
    dealias(fft2(ua1_0)), dealias(fft2(ua2_0)), dealias(fft2(theta_a0)),
    dealias(fft2(uo1_0)), dealias(fft2(uo2_0)), dealias(fft2(theta_o0)),
])
state[0], state[1] = leray_project(state[0], state[1])
state[3], state[4] = leray_project(state[3], state[4])


# ----------------------------------------------------------------------
# Time integration
# ----------------------------------------------------------------------
def energy(state):
    ua1, ua2 = ifft2(state[0]), ifft2(state[1])
    uo1, uo2 = ifft2(state[3]), ifft2(state[4])
    Ea = 0.5 * np.mean(ua1 ** 2 + ua2 ** 2)
    Eo = 0.5 * np.mean(uo1 ** 2 + uo2 ** 2)
    return Ea, Eo


history = {"t": [], "Ea": [], "Eo": []}
snapshots = []

t = 0.0
for step in range(n_steps + 1):
    if step % save_every == 0:
        Ea, Eo = energy(state)
        history["t"].append(t)
        history["Ea"].append(Ea)
        history["Eo"].append(Eo)
        snapshots.append((t, ifft2(state[2]).copy(), ifft2(state[5]).copy()))
        print(f"step {step:5d}  t={t:7.3f}  Ea={Ea:.4e}  Eo={Eo:.4e}")
    if step == n_steps:
        break
    state = rk4_step(state, dt)
    t += dt

# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(history["t"], history["Ea"], label="atmosphere KE")
axes[0].plot(history["t"], history["Eo"], label="ocean KE")
axes[0].set_xlabel("t"); axes[0].set_ylabel("mean kinetic energy")
axes[0].legend(); axes[0].set_title("Kinetic energy evolution")

t_final, tha_final, tho_final = snapshots[-1]
im = axes[1].contourf(X, Y, tha_final - tho_final, levels=30, cmap="RdBu_r")
axes[1].set_title(f"theta_a - theta_o at t={t_final:.2f}")
plt.colorbar(im, ax=axes[1])
plt.tight_layout()
plt.savefig("coupled_atmos_ocean_summary.png", dpi=150)

fig2, axes2 = plt.subplots(2, 2, figsize=(10, 9))
ua1f, ua2f = ifft2(state[0]), ifft2(state[1])
uo1f, uo2f = ifft2(state[3]), ifft2(state[4])
thaf, thof = ifft2(state[2]), ifft2(state[5])

im0 = axes2[0, 0].contourf(X, Y, thaf, levels=30, cmap="coolwarm")
axes2[0, 0].set_title("theta_a (final)"); plt.colorbar(im0, ax=axes2[0, 0])

im1 = axes2[0, 1].contourf(X, Y, thof, levels=30, cmap="coolwarm")
axes2[0, 1].set_title("theta_o (final)"); plt.colorbar(im1, ax=axes2[0, 1])

speed_a = np.sqrt(ua1f ** 2 + ua2f ** 2)
im2 = axes2[1, 0].contourf(X, Y, speed_a, levels=30, cmap="viridis")
axes2[1, 0].quiver(X[::8, ::8], Y[::8, ::8], ua1f[::8, ::8], ua2f[::8, ::8])
axes2[1, 0].set_title("|u_a| (final)"); plt.colorbar(im2, ax=axes2[1, 0])

speed_o = np.sqrt(uo1f ** 2 + uo2f ** 2)
im3 = axes2[1, 1].contourf(X, Y, speed_o, levels=30, cmap="viridis")
axes2[1, 1].quiver(X[::8, ::8], Y[::8, ::8], uo1f[::8, ::8], uo2f[::8, ::8])
axes2[1, 1].set_title("|u_o| (final)"); plt.colorbar(im3, ax=axes2[1, 1])

plt.tight_layout()
plt.savefig("coupled_atmos_ocean_fields.png", dpi=150)
print("Saved coupled_atmos_ocean_summary.png and coupled_atmos_ocean_fields.png")
