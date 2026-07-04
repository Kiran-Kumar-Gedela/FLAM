"""
solve_parametric_curve.py

Finds unknown parameters (theta, M, X) in a parametric curve equation
by treating it as a rotated + translated oscillation and using global
optimization to minimize the reconstruction error.

Author: Kiran Kumar Gedela
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import matplotlib
import os
import time

matplotlib.use('Agg')

# --- config ---
DATA_FILE = "xy_data.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# bounds from the assignment
THETA_BOUNDS = (np.radians(0.1), np.radians(49.9))
M_BOUNDS = (-0.05, 0.05)
X_BOUNDS = (0.1, 99.9)

print("=" * 70)
print("  Parametric Curve Parameter Recovery")
print("=" * 70)


# load the data
print("\n[1] Loading data...")
df = pd.read_csv(DATA_FILE)
x_data = df['x'].values
y_data = df['y'].values
n_points = len(x_data)

print(f"  {n_points} points loaded from '{DATA_FILE}'")
print(f"  x in [{x_data.min():.4f}, {x_data.max():.4f}]")
print(f"  y in [{y_data.min():.4f}, {y_data.max():.4f}]")


# The main idea:
#
# If we let u = t and w = exp(M|t|)*sin(0.3t), the parametric equations
# are really just rotating (u, w) by theta and shifting by (X, 42).
# That means we can undo the rotation to get t and w back:
#
#   t = (x - X)*cos(theta) + (y - 42)*sin(theta)
#   w = -(x - X)*sin(theta) + (y - 42)*cos(theta)
#
# Then we check if w actually matches exp(M|t|)*sin(0.3t).
# When theta, M, X are correct, the match is perfect.

def cost_function(params, x_data, y_data):
    """Sum of squared differences between recovered w and model w."""
    theta, M, X = params
    ct, st = np.cos(theta), np.sin(theta)

    xp = x_data - X
    yp = y_data - 42

    # undo the rotation
    t_rec = xp * ct + yp * st
    w_rec = -xp * st + yp * ct

    # what the model says w should be
    w_model = np.exp(M * np.abs(t_rec)) * np.sin(0.3 * t_rec)

    return np.sum((w_rec - w_model) ** 2)


# Step 1: global search with Differential Evolution
# (a population-based optimizer that doesn't need gradients)
print("\n[2] Global search (Differential Evolution)...")
print(f"    theta: {np.degrees(THETA_BOUNDS[0]):.1f} to {np.degrees(THETA_BOUNDS[1]):.1f} deg")
print(f"    M:     {M_BOUNDS[0]} to {M_BOUNDS[1]}")
print(f"    X:     {X_BOUNDS[0]} to {X_BOUNDS[1]}")

t0 = time.time()
bounds = [THETA_BOUNDS, M_BOUNDS, X_BOUNDS]

de_result = differential_evolution(
    cost_function, bounds,
    args=(x_data, y_data),
    strategy='best1bin',
    maxiter=1000,
    popsize=30,
    tol=1e-16,
    mutation=(0.5, 1.5),
    recombination=0.9,
    seed=42,
    polish=False
)

theta_de, M_de, X_de = de_result.x
print(f"  Done in {time.time()-t0:.2f}s")
print(f"  theta={np.degrees(theta_de):.6f} deg, M={M_de:.8f}, X={X_de:.6f}")
print(f"  cost={de_result.fun:.2e}")


# Step 2: polish with L-BFGS-B for extra precision
print("\n[3] Local refinement (L-BFGS-B)...")
t0 = time.time()

refined = minimize(
    cost_function, de_result.x,
    args=(x_data, y_data),
    method='L-BFGS-B',
    bounds=bounds,
    options={'maxiter': 10000, 'ftol': 1e-30, 'gtol': 1e-20}
)

theta_opt, M_opt, X_opt = refined.x
print(f"  Done in {time.time()-t0:.4f}s")
print(f"  theta={np.degrees(theta_opt):.10f} deg, M={M_opt:.10f}, X={X_opt:.10f}")
print(f"  cost={refined.fun:.2e}")


# Step 3: verify the fit
print("\n[4] Checking results...")

ct, st = np.cos(theta_opt), np.sin(theta_opt)
xp = x_data - X_opt
yp = y_data - 42

t_rec = xp * ct + yp * st
w_rec = -xp * st + yp * ct
w_model = np.exp(M_opt * np.abs(t_rec)) * np.sin(0.3 * t_rec)
residuals = w_rec - w_model

print(f"  Recovered t: [{t_rec.min():.4f}, {t_rec.max():.4f}] (should be ~6 to ~60)")

# reconstruct x,y from the parameters to measure L1 error
x_recon = t_rec * ct - np.exp(M_opt*np.abs(t_rec))*np.sin(0.3*t_rec)*st + X_opt
y_recon = 42 + t_rec * st + np.exp(M_opt*np.abs(t_rec))*np.sin(0.3*t_rec)*ct

l1_per_point = np.abs(x_data - x_recon) + np.abs(y_data - y_recon)

print(f"  Mean L1 per point: {l1_per_point.mean():.2e}")
print(f"  Max L1:            {l1_per_point.max():.2e}")
print(f"  Total L1:          {l1_per_point.sum():.6f}")

theta_deg = np.degrees(theta_opt)

print("\n" + "=" * 70)
print("  RESULTS")
print("=" * 70)
print(f"  theta = {theta_deg:.6f} deg  (~{theta_deg:.0f} deg = pi/6)")
print(f"  M     = {M_opt:.10f}  (~{M_opt:.2f})")
print(f"  X     = {X_opt:.10f}  (~{X_opt:.0f})")
print(f"\n  Clean values:  theta=30 deg,  M=0.03,  X=55")

latex_str = (
    r"\left(t*\cos(0.5235987756)-e^{0.03\left|t\right|}\cdot\sin(0.3t)"
    r"\sin(0.5235987756)+55,42+t*\sin(0.5235987756)+e^{0.03\left|t\right|}"
    r"\cdot\sin(0.3t)\cos(0.5235987756)\right)"
)
print(f"\n  LaTeX: {latex_str}")
print("=" * 70)


# Step 4: generate plots

print("\n[5] Saving plots...")

# use the clean exact values for plotting
theta_ex = np.pi / 6
M_ex = 0.03
X_ex = 55.0
t_smooth = np.linspace(6, 60, 5000)

x_curve = t_smooth*np.cos(theta_ex) - np.exp(M_ex*np.abs(t_smooth))*np.sin(0.3*t_smooth)*np.sin(theta_ex) + X_ex
y_curve = 42 + t_smooth*np.sin(theta_ex) + np.exp(M_ex*np.abs(t_smooth))*np.sin(0.3*t_smooth)*np.cos(theta_ex)


# plot 1: data vs fitted curve
fig, ax = plt.subplots(figsize=(14, 8))
ax.scatter(x_data, y_data, c='#3498db', alpha=0.4, s=15, label=f'Data ({n_points} pts)', edgecolors='none', zorder=2)
ax.plot(x_curve, y_curve, color='#e74c3c', lw=2.0, label='Fitted curve', zorder=3)
ax.set_xlabel('x', fontsize=14, fontweight='bold')
ax.set_ylabel('y', fontsize=14, fontweight='bold')
ax.set_title('Data vs. Recovered Parametric Curve', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3)

info = f'theta = 30 deg\nM = 0.03\nX = 55\nMean L1 = {l1_per_point.mean():.2e}'
ax.text(0.98, 0.02, info, transform=ax.transAxes, fontsize=11,
        va='bottom', ha='right', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'curve_fit.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> plots/curve_fit.png")


# plot 2: residuals
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].hist(residuals, bins=60, color='#2ecc71', edgecolor='#27ae60', alpha=0.8)
axes[0].axvline(x=0, color='#e74c3c', ls='--', lw=1.5, label='Zero')
axes[0].set_xlabel('Residual (w_rec - w_model)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Residual Distribution', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

idx = np.argsort(t_rec)
axes[1].scatter(t_rec[idx], l1_per_point[idx], c='#9b59b6', alpha=0.5, s=10, edgecolors='none')
axes[1].set_xlabel('Recovered t', fontsize=12)
axes[1].set_ylabel('L1 Distance', fontsize=12)
axes[1].set_title('L1 Error vs. t', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'residual_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> plots/residual_analysis.png")


# plot 3: linearity check (ln(w/sin(0.3t)) should be a straight line with slope M)
fig, ax = plt.subplots(figsize=(12, 7))

mask = np.sin(0.3 * t_rec) > 0.01
t_v = t_rec[mask]
w_v = w_rec[mask]
log_ratio = np.log(w_v / np.sin(0.3 * t_v))

ax.scatter(np.abs(t_v), log_ratio, c='#3498db', alpha=0.5, s=15, label='ln(C / sin(0.3t))', edgecolors='none')

t_line = np.linspace(6, 60, 100)
ax.plot(t_line, M_ex * t_line, color='#e74c3c', lw=2.5, ls='--', label=f'slope = M = {M_ex}')

ax.set_xlabel('|t|', fontsize=14, fontweight='bold')
ax.set_ylabel('ln(C / sin(0.3t))', fontsize=14, fontweight='bold')
ax.set_title('Linearity Check: confirms M = 0.03', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'linearity_check.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> plots/linearity_check.png")

print("\nDone.")
