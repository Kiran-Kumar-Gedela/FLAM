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

## Complete Thought Process: How I Solved It

To arrive at the mathematically exact parameters, I followed a structured, three-step analytical process.

### Step 1: The Key Mathematical Observation (Decoupling)

At first glance, the equations look highly non-linear and difficult to solve directly. However, by carefully studying their structure, I realized they describe a **2D rotation**. If we define the terms:

$$u = t \qquad \text{and} \qquad w = e^{M|t|} \cdot \sin(0.3t)$$

Then the original parametric equations simplify to:

$$x - X = u\cos\theta - w\sin\theta$$
$$y - 42 = u\sin\theta + w\cos\theta$$

This is exactly the standard 2D rotation matrix applied to the vector $(u, w)$, followed by a translation by $(X, 42)$. Because rotation matrices are orthogonal, we can easily invert this rotation by multiplying by the transpose matrix to recover $u$ and $w$:

$$t = (x - X)\cos\theta + (y - 42)\sin\theta$$
$$w = -(x - X)\sin\theta + (y - 42)\cos\theta$$

**Why this is crucial:** This inversion allows us to recover the original parameter $t$ directly from any data point $(x_i, y_i)$. Most importantly, recovering $t$ only depends on $\theta$ and $X$, and is completely independent of $M$. 

### Step 2: The Fitting Strategy and Optimizations

With the equations decoupled, the problem becomes much simpler. For any guess of $(\theta, X)$, I can compute $t$ and $w$ for all 1,500 points. Then, I just need to check if the recovered $w$ matches the theoretical model $e^{M|t|}\sin(0.3t)$ for some $M$. 

To find the perfect parameters $(\theta, M, X)$, I designed a cost function that calculates the sum of squared errors between the recovered $w$ and the theoretical $w$. I minimized this cost function using two robust optimization techniques:

1. **Differential Evolution:** I started with this global optimization algorithm (Storn & Price, 1997). Since it is population-based and doesn't rely on gradients, it aggressively searched the entire allowed parameter space to find the rough global minimum without getting stuck in local traps.
2. **L-BFGS-B:** Once the global region was found, I used this highly efficient local refinement algorithm (Byrd et al., 1995) to polish the results down to extreme machine precision.

### Step 3: Sanity Check and Verification

As a final check to ensure the math was completely solid, I linearized the exponential component. If the parameters are correct, then $w / \sin(0.3t) = e^{M|t|}$, which means:

$$\ln\left(\frac{w}{\sin(0.3t)}\right) = M \cdot |t|$$

Plotting this should yield a perfect straight line passing through the origin with a slope of $M$. The linearity plot in the results section confirms this flawlessly, proving that $M = 0.03$.

---

## Results

### Fit Quality Metrics

| Metric | Value |
|--------|-------|
| Mean L1 distance per point | $2.06 \times 10^{-5}$ |
| Mean L2 distance per point | $1.46 \times 10^{-5}$ |
| Max L1 distance | $5.53 \times 10^{-5}$ |
| Sum of all L1 distances | $0.031$ |

Both the L1 and L2 errors are incredibly small. Specifically, the L1 error is as small as theoretically possible given the floating-point truncation present in the provided `xy_data.csv` (which only goes to about 6 decimal places). The fit is essentially mathematically exact.

### Plots

**1. Data vs. fitted curve** — the analytical curve passes exactly through every given point from the dataset:

![curve fit](plots/curve_fit.png)

**2. Residual distribution** — the residuals between the model and the data are tightly centered around zero (order of $10^{-5}$):

![residuals](plots/residual_analysis.png)

**3. Linearity check** — the logarithmic transformation yields a perfect straight line, confirming $M = 0.03$:

![linearity](plots/linearity_check.png)

---

## Conclusion

By treating the parametric equations as an inverted 2D rotation, we decoupled the variables and reduced a highly complex non-linear problem into a tractable optimization task. Utilizing Differential Evolution followed by L-BFGS-B allowed us to find the exact parameters ($\theta = 30^\circ$, $M = 0.03$, $X = 55$) with zero mathematical error (accounting for the dataset's floating-point precision constraints). The visual verification plots and the near-zero L1/L2 distances confirm the complete accuracy of this solution.

---

## Running the Code

Dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`

```bash
python solve_parametric_curve.py    # finds the parameters, saves plots
python verify_solution.py           # independent check with known answer
```

---

## References

1. Storn, R. & Price, K. (1997). Differential Evolution — A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces. *Journal of Global Optimization*, 11(4), 341–359.
2. Byrd, R.H., Lu, P., Nocedal, J. & Zhu, C. (1995). A Limited Memory Algorithm for Bound Constrained Optimization. *SIAM Journal on Scientific Computing*, 16(5), 1190–1208.
3. Preparata, F.P. & Shamos, M.I. (1985). *Computational Geometry: An Introduction*. Springer-Verlag.
4. Weisstein, E.W. Rotation Matrix. *MathWorld*.
