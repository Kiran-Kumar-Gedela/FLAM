# Parametric Curve — Unknown Parameter Recovery

**Author:** Kiran Kumar Gedela

## Problem

We are given a parametric curve defined by:

$$x = t \cdot \cos(\theta) - e^{M|t|} \cdot \sin(0.3t) \cdot \sin(\theta) + X$$

$$y = 42 + t \cdot \sin(\theta) + e^{M|t|} \cdot \sin(0.3t) \cdot \cos(\theta)$$

and 1,500 data points $(x_i, y_i)$ that lie on this curve (provided in `xy_data.csv`).

**Goal:** Find the three unknown parameters — $\theta$, $M$, and $X$ — so that the curve perfectly passes through all the given data points.

**Allowed Ranges:**

| Parameter | Allowed range |
|-----------|---------------|
| $\theta$ (angle) | $0^\circ < \theta < 50^\circ$ |
| $M$ (growth rate) | $-0.05 < M < 0.05$ |
| $X$ (horizontal shift) | $0 < X < 100$ |
| $t$ (curve parameter) | $6 < t < 60$ |

---

## Answer

$$\theta = 30^\circ \qquad M = 0.03 \qquad X = 55$$

In radians: $\theta = \pi/6 \approx 0.5235987756$

Plugging these in, the curve becomes the coordinate pair $(x(t), y(t))$:

$$x(t) = t \cos\left(\frac{\pi}{6}\right) - e^{0.03|t|} \sin(0.3t) \sin\left(\frac{\pi}{6}\right) + 55$$

$$y(t) = 42 + t \sin\left(\frac{\pi}{6}\right) + e^{0.03|t|} \sin(0.3t) \cos\left(\frac{\pi}{6}\right)$$

for $6 < t < 60$.

### Submission (LaTeX / Desmos format)

As required in the assignment instructions, here is the exact answer formatted as a single LaTeX expression for submission. The outer parentheses represent the 2D coordinate pair $(x(t), y(t))$, with a comma explicitly separating the x-coordinate formula from the y-coordinate formula so it graphs correctly in Desmos.

```latex
\left(t*\cos(0.5235987756)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.5235987756)+55, 42+t*\sin(0.5235987756)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.5235987756)\right)
```

---

## How I Solved It

### The key observation

The equations actually describe a **2D rotation**. If we define:

$$u = t \qquad \text{and} \qquad w = e^{M|t|} \cdot \sin(0.3t)$$

Then the equations become:

$$x - X = u\cos\theta - w\sin\theta$$
$$y - 42 = u\sin\theta + w\cos\theta$$

This is exactly the rotation matrix applied to the vector $(u, w)$, then shifted by $(X, 42)$. We can invert this rotation by multiplying by the transpose to recover $u$ and $w$:

$$t = (x - X)\cos\theta + (y - 42)\sin\theta$$
$$w = -(x - X)\sin\theta + (y - 42)\cos\theta$$

This allows us to recover the original parameter $t$ directly from any data point. It only depends on $\theta$ and $X$, completely independent of $M$.

### The fitting strategy and optimizations

Once I could recover $t$ and $w$ for a guess of $(\theta, X)$, I needed to check whether $w$ matches the theoretical model $e^{M|t|}\sin(0.3t)$ for some $M$. 

To find the perfect parameters, I used two optimization techniques:
1. **Differential Evolution**: A global optimization algorithm (Storn & Price, 1997) that aggressively searches the entire parameter space to find the rough global minimum.
2. **L-BFGS-B**: A highly efficient local refinement algorithm (Byrd et al., 1995) to polish the results to extreme machine precision.

### A sanity check for M

If the parameters are correct, then $w / \sin(0.3t) = e^{M|t|}$, so:

$$\ln\left(\frac{w}{\sin(0.3t)}\right) = M \cdot |t|$$

This should be a straight line passing through the origin with slope $M$. The linearity plot in the results confirms this perfectly with slope = 0.03.

---

## Results

### Fit quality

| Metric | Value |
|--------|-------|
| Mean L1 distance per point | $2.06 \times 10^{-5}$ |
| Mean L2 distance per point | $1.46 \times 10^{-5}$ |
| Max L1 distance | $5.53 \times 10^{-5}$ |
| Sum of all L1 distances | $0.031$ |

Both the L1 and L2 errors are incredibly small. Specifically, the L1 error is as small as theoretically possible given the floating-point truncation present in the provided `xy_data.csv` (which only goes to about 6 decimal places). The fit is essentially mathematically exact.

### Plots

**Data vs. fitted curve** — the curve passes exactly through every given point:

![curve fit](plots/curve_fit.png)

**Residual distribution** — residuals are tightly centered around zero (order of $10^{-5}$):

![residuals](plots/residual_analysis.png)

**Linearity check** — confirming $M = 0.03$:

![linearity](plots/linearity_check.png)

---

## Running the Code

Dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`

```bash
python solve_parametric_curve.py    # finds the parameters, saves plots
python verify_solution.py           # independent check with known answer
```
