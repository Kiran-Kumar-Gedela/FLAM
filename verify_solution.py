"""
verify_solution.py

Quick independent check: plugs in theta=30 deg, M=0.03, X=55 and
measures how well the parametric curve matches the given data.

Author: Kiran Kumar Gedela
"""

import numpy as np
import pandas as pd

# our solution
THETA_RAD = np.pi / 6   # 30 degrees
M = 0.03
X = 55.0

print("=" * 60)
print("  Verifying: theta=30 deg, M=0.03, X=55")
print("=" * 60)

# load data
df = pd.read_csv("xy_data.csv")
x_data = df['x'].values
y_data = df['y'].values
print(f"\n{len(x_data)} points loaded")

# undo the rotation to recover t
ct = np.cos(THETA_RAD)
st = np.sin(THETA_RAD)
xp = x_data - X
yp = y_data - 42

t_rec = xp * ct + yp * st
w_rec = -xp * st + yp * ct

print(f"Recovered t range: [{t_rec.min():.6f}, {t_rec.max():.6f}]")
print(f"Expected: (6, 60)")

ok = np.all((t_rec > 5.5) & (t_rec < 60.5))
print(f"t in range: {'yes' if ok else 'NO'}")

# reconstruct (x,y) and compare
x_pred = t_rec*ct - np.exp(M*np.abs(t_rec))*np.sin(0.3*t_rec)*st + X
y_pred = 42 + t_rec*st + np.exp(M*np.abs(t_rec))*np.sin(0.3*t_rec)*ct

l1 = np.abs(x_data - x_pred) + np.abs(y_data - y_pred)

print(f"\nL1 distances:")
print(f"  mean:   {l1.mean():.6e}")
print(f"  median: {np.median(l1):.6e}")
print(f"  max:    {l1.max():.6e}")
print(f"  total:  {l1.sum():.6f}")

# also check the w residuals directly
w_pred = np.exp(M * np.abs(t_rec)) * np.sin(0.3 * t_rec)
w_err = w_rec - w_pred

print(f"\nModel residuals (w):")
print(f"  mean abs: {np.abs(w_err).mean():.6e}")
print(f"  max abs:  {np.abs(w_err).max():.6e}")
print(f"  SSR:      {np.sum(w_err**2):.6e}")

print(f"\n{'=' * 60}")
if l1.mean() < 1e-3:
    print(f"  PASSED  (mean L1 = {l1.mean():.2e})")
else:
    print(f"  FAILED  (mean L1 = {l1.mean():.2e})")
print(f"  theta=30 deg (pi/6), M=0.03, X=55")
print(f"{'=' * 60}")
