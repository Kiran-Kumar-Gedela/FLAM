"""
verify_solution.py
==================
Independent verification script for the FLAM R&D parametric curve assignment.

Takes the found parameters θ, M, X and verifies them against the data by:
1. Recovering t for each data point via inverse rotation
2. Reconstructing (x, y) from the parameters and recovered t
3. Computing L1 distances between original and reconstructed points
4. Reporting pass/fail based on acceptable error thresholds

Author: Kiran Kumar Gedela
"""

import numpy as np
import pandas as pd

# ============================================================================
# Solution Parameters (found by solve_parametric_curve.py)
# ============================================================================
THETA_DEG = 30.0                      # degrees
THETA_RAD = np.pi / 6                 # exact: π/6 radians
M = 0.03
X = 55.0
Y_OFFSET = 42                         # given constant in the equation

# Acceptable error threshold
L1_THRESHOLD = 1e-3  # per-point L1 distance threshold

# ============================================================================
# Load Data
# ============================================================================
print("=" * 60)
print("  Verification of Solution: theta=30 deg, M=0.03, X=55")
print("=" * 60)

df = pd.read_csv("xy_data.csv")
x_data = df['x'].values
y_data = df['y'].values
n = len(x_data)
print(f"\nLoaded {n} data points from xy_data.csv")

# ============================================================================
# Step 1: Recover parameter t via inverse rotation
# ============================================================================
ct = np.cos(THETA_RAD)
st = np.sin(THETA_RAD)

# Translate
xp = x_data - X
yp = y_data - Y_OFFSET

# Inverse rotation
t_recovered = xp * ct + yp * st       # = t (the parameter along the curve)
w_recovered = -xp * st + yp * ct      # = e^(M|t|) * sin(0.3t)

print(f"\nRecovered t range: [{t_recovered.min():.6f}, {t_recovered.max():.6f}]")
print(f"Expected t range:  (6, 60)")

# Check t is in valid range
t_in_range = np.all((t_recovered > 5.5) & (t_recovered < 60.5))
print(f"All t values in valid range: {'[PASS]' if t_in_range else '[FAIL]'}")

# ============================================================================
# Step 2: Reconstruct (x, y) from found parameters
# ============================================================================
x_pred = (t_recovered * ct
          - np.exp(M * np.abs(t_recovered)) * np.sin(0.3 * t_recovered) * st
          + X)
y_pred = (Y_OFFSET + t_recovered * st
          + np.exp(M * np.abs(t_recovered)) * np.sin(0.3 * t_recovered) * ct)

# ============================================================================
# Step 3: Compute L1 Distances
# ============================================================================
l1_x = np.abs(x_data - x_pred)
l1_y = np.abs(y_data - y_pred)
l1_per_point = l1_x + l1_y

mean_l1 = l1_per_point.mean()
max_l1 = l1_per_point.max()
sum_l1 = l1_per_point.sum()
median_l1 = np.median(l1_per_point)

print(f"\n--- L1 Distance Results ---")
print(f"  Mean L1 distance:   {mean_l1:.6e}")
print(f"  Median L1 distance: {median_l1:.6e}")
print(f"  Max L1 distance:    {max_l1:.6e}")
print(f"  Sum L1 distance:    {sum_l1:.6f}")

# ============================================================================
# Step 4: Verify w matches the model prediction
# ============================================================================
w_predicted = np.exp(M * np.abs(t_recovered)) * np.sin(0.3 * t_recovered)
w_residuals = w_recovered - w_predicted

print(f"\n--- Model Residuals (w_recovered vs w_predicted) ---")
print(f"  Mean absolute residual: {np.abs(w_residuals).mean():.6e}")
print(f"  Max absolute residual:  {np.abs(w_residuals).max():.6e}")
print(f"  Sum of squares:         {np.sum(w_residuals**2):.6e}")

# ============================================================================
# Step 5: Pass/Fail Determination
# ============================================================================
print(f"\n{'=' * 60}")
if mean_l1 < L1_THRESHOLD:
    print(f"  [PASS] VERIFICATION PASSED")
    print(f"  Mean L1 = {mean_l1:.2e} < threshold {L1_THRESHOLD:.0e}")
else:
    print(f"  [FAIL] VERIFICATION FAILED")
    print(f"  Mean L1 = {mean_l1:.2e} >= threshold {L1_THRESHOLD:.0e}")

print(f"\n  Solution: theta = {THETA_DEG} deg (pi/6 rad), M = {M}, X = {X}")
print(f"{'=' * 60}")
