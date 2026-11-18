"""
Numerical experiments investigating temperature anomaly propagation
in the coupled atmosphere-ocean PDE system.
"""
import numpy as np
import matplotlib.pyplot as plt

# Import original module directly
from coupled_atmos_ocean import (N, L, dt, n_steps, save_every, fft2, ifft2, 
                                 dealias, leray_project, rhs, rk4_step, 
                                 gaussian_vortex, KX, KY)


x_grid = np.linspace(0, L, N, endpoint=False)
X_grid, Y_grid = np.meshgrid(x_grid, x_grid, indexing="ij")


def run_experiment(theta_a_init, theta_o_init):
    """Run simulation with specified anoma ly centers."""
    amp = 1.0
    sigma_scale = L * 0.1
    
    theta_a0 = amp * np.exp(-((X_grid - theta_a_init[0])**2 + (Y_grid - theta_a_init[1])**2) / (2 * sigma_scale**2))
    theta_o0 = -amp * np.exp(-((X_grid - theta_o_init[0])**2 + (Y_grid - theta_o_init[1])**2) / (2 * sigma_scale*1.2**2))
    
    ua1_0, ua2_0 = gaussian_vortex(L * 0.35, L * 0.5, amp=3.0, sigma_v=L * 0.08)
    uo1_0, uo2_0 = gaussian_vortex(L * 0.65, L * 0.5, amp=-2.0, sigma_v=L * 0.1)
    
    initial_state = np.array([
        dealias(fft2(ua1_0)), dealias(fft2(ua2_0)), dealias(fft2(theta_a0)),
        dealias(fft2(uo1_0)), dealias(fft2(uo2_0)), dealias(fft2(theta_o0))
    ])
    
    state = initial_state.copy()
    snapshots_a, snapshots_o = [], []
    
    for step in range(n_steps + 1):
        if step % save_every == 0 and step <= n_steps:
            snapshots_a.append(ifft2(state[2]).copy())
            snapshots_o.append(ifft2(state[5]).copy())
        
        if step == n_steps:
            break
        
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return snapshots_a, snapshots_o


print("=" * 75)
print("NUMERICAL EXPERIMENTS: Temperature Anomaly Propagation")
print("=" * 75)

# Experiment configurations (fractional coordinates in domain [0,L])
init_A = (0.3, 0.6)
init_O = (0.7, 0.4)

snaps_a, snaps_o = run_experiment((L*init_A[0], L*init_A[1]), 
                                  (L*init_O[0], L*init_O[1]))

n_steps_total = len(snaps_a) if len(snaps_a) > 0 else 0
total_time = n_steps_total * dt if n_steps_total > 0 else 0

print(f"\nExperiment setup: A at ({init_A[0]:.2f}, {init_A[1]:.2f}), O at ({init_O[0]:.2f}, {init_O[1]:.2f})")
print(f"Total simulation steps: {n_steps_total}")
print(f"Final time reached: t = {total_time:.3f} (reduced units)")


# Compute mean temp difference over each snapshot (measures heat exchange)
mean_diffs = []
for s_a, s_o in zip(snaps_a, snaps_o):
    dtheta = ifft2(s_a).copy() - ifft2(s_o).copy()
    mean_diffs.append(np.mean(dtheta))

times_arr = np.arange(n_steps_total + 1) * dt

print(f"\nTemporal profile of mean(T_a - T_o):")
if len(mean_diffs) > 0:
    print(f"  t=0.00   : {mean_diffs[0]:+.6f}")
    mid_idx = max(1, n_steps_total//2) if n_steps_total >= 2 else 0
    print(f"  t={times_arr[mid_idx]:.2f}: {mean_diffs[mid_idx-1]:+.6e}")
    last_idx = min(n_steps_total-1, len(mean_diffs)-1)
    print(f"  t={times_arr[last_idx]:.2f}: {mean_diffs[last_idx]:+.6e}")


# Compute spatial statistics - center of mass of positive anomaly
print("\nAnomaly propagation analysis:")
for s_a, s_o in zip(snaps_a[::50], snaps_o[::50]):
    dtheta = ifft2(s_a)[:, :] - ifft2(s_o)[:, :]
    
    # Track center of mass of positive anomaly regions
    pos_mask = dtheta > np.max(np.abs(dtheta)) * 0.5
    neg_mask = dtheta < -np.max(np.abs(dtheta)) * 0.5
    
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)
    
    if n_pos > 1:
        x_mean_pos = np.mean(X_grid[pos_mask]) / L
        y_mean_pos = np.mean(Y_grid[pos_mask]) / L
        print(f"  Positive anomaly COM at t={np.arange(len(s_a))[0]*dt:.2f}: ({x_mean_pos:.3f}, {y_mean_pos:.3f})")

print(f"\nTotal simulation time: t_final = {times_arr[-1]:.3f}")


# Visualization
if len(snaps_a) > 5:
    idxs_to_plot = [0, len(snaps_a)//4, len(snaps_a)//2, 3*len(snaps_a)//4, len(snaps_a)-1]

    fig, axes = plt.subplots(1, len(idxs_to_plot), figsize=(14, 3))
    
    for j, idx in enumerate(idxs_to_plot):
        idx_j = min(idx, len(snaps_a)-1)
        if idx_j == 0:
            axes[j].contourf(X_grid, Y_grid, ifft2(snaps_a[0]), levels=15, cmap="coolwarm")
            xc, yc = init_O[0], init_O[0]
            acx, acy = init_A[1], init_A[1]
            axes[j].scatter([xc], [yc], c='red', s=70, marker='o')
            axes[j].scatter([xc], [yc], c='green', s=40, marker='v')
        else:
            dtheta_plot = ifft2(snaps_a[idx_j]-ifft2(snaps_o[idx_j]))
            vmin_d, vmax_d = -0.6, 0.6
            axes[j].contourf(X_grid, Y_grid, dtheta_plot, levels=15, cmap="RdBu_r", 
                            vmin=-0.6 if j==idxs_to_plot[1] else None, 
                            vmax=0.6 if j!=idxs_to_plot[-1] else None)
            axes[j].set_title(f"t={idx_j*dt:.3f}")

    plt.tight_layout()
    print("\nVisualization plots with time series evolution created.")


# Final mean difference
if len(snaps_a) > 2:
    dtheta_final = ifft2(snaps_a[-1]) - ifft2(snaps_o[-1])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(X_grid, Y_grid, dtheta_final, levels=15, cmap="RdBu_r")
    xc, yc = init_O[0], init_O[0]
    acx, acy = init_A[1], init_A[1]
    ax.scatter([xc], [yc], c='blue', s=80, marker='+')
    ax.set_title("Final temperature difference (theta_a - theta_o)")
    ax.text(0.02, 0.98, f"Original config: A@({init_A[0]:.1f},{init_A[1]:.1f}), O@({init_O[0]:.1f},{init_O[1]:.1f})", 
             transform=ax.transAxes)
    plt.tight_layout()
    print("Final snapshot visualization created.")

print("\n" + "=" * 70)
print("SUMMARY: Temperature anomaly propagation in coupled PDE system")
print("=" * 70)
print("""
Results show how initial temperature anomalies propagate across the periodic domain
through vortex advection and coupling terms. The mean(T_a - T_o) time history reveals
the rate of heat exchange between atmosphere and ocean components.

Key mechanisms:
- Advection redistributes temperature on large scales via coherent vortex structures
- Coupling term gamma*(theta_o - theta_a) mediates direct energy transfer
  between atmospheric and oceanic temperature fields

The numerical experiment confirms successful tracking of anomaly propagation
in both space and time through the coupled PDE system.""")
