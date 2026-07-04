# FLAM — AI R&D Assignment: Parametric Curve Parameter Recovery

## 📋 Problem Statement

Given a set of **1,500 data points** $(x, y)$ lying on a parametric curve, the task is to recover three unknown parameters $\theta$, $M$, and $X$ from the following parametric equations:

$$x = t \cdot \cos(\theta) - e^{M|t|} \cdot \sin(0.3t) \cdot \sin(\theta) + X$$

$$y = 42 + t \cdot \sin(\theta) + e^{M|t|} \cdot \sin(0.3t) \cdot \cos(\theta)$$

**Constraints on the unknowns:**
| Parameter | Range |
|-----------|-------|
| $\theta$ | $0° < \theta < 50°$ |
| $M$ | $-0.05 < M < 0.05$ |
| $X$ | $0 < X < 100$ |
| $t$ (curve parameter) | $6 < t < 60$ |

**Data provided:** `xy_data.csv` containing 1,500 $(x, y)$ points sampled from the curve.

---

## ✅ Solution (Submission)

### Recovered Parameters

| Parameter | Value | Exact Form |
|-----------|-------|------------|
| $\theta$ | **30°** | $\frac{\pi}{6} \approx 0.5235987756$ radians |
| $M$ | **0.03** | — |
| $X$ | **55** | — |

### LaTeX Submission String

```
\left(t*\cos(0.5235987756)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.5235987756)+55,42+t*\sin(0.5235987756)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.5235987756)\right)
```

### Parametric Equations with Recovered Values

$$x = t \cdot \cos\left(\frac{\pi}{6}\right) - e^{0.03|t|} \cdot \sin(0.3t) \cdot \sin\left(\frac{\pi}{6}\right) + 55$$

$$y = 42 + t \cdot \sin\left(\frac{\pi}{6}\right) + e^{0.03|t|} \cdot \sin(0.3t) \cdot \cos\left(\frac{\pi}{6}\right)$$

### Desmos Visualization

The curve can be visualized on [Desmos](https://www.desmos.com/calculator/rfj91yrxob) by entering the LaTeX string above.

---

## 🔬 Mathematical Approach

### Step 1: Understanding the Structure — Rotation Decomposition

The key insight is recognizing that the parametric equations represent a **2D rotation followed by a translation**. This is a well-known transformation in computational geometry (Preparata & Shamos, 1985).

Define two component functions:
- $u(t) = t$ — the "spine" (linear component along the curve)  
- $w(t) = e^{M|t|} \cdot \sin(0.3t)$ — the "oscillation" (perpendicular displacement)

Then the parametric equations become:

$$x - X = u \cdot \cos(\theta) - w \cdot \sin(\theta)$$

$$y - 42 = u \cdot \sin(\theta) + w \cdot \cos(\theta)$$

This is exactly the formula for rotating the point $(u, w)$ by angle $\theta$, then translating by $(X, 42)$. In matrix form:

$$\begin{pmatrix} x - X \\ y - 42 \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} u \\ w \end{pmatrix}$$

This is the standard **2D rotation matrix** $R(\theta)$ (Weisstein, "Rotation Matrix", MathWorld).

### Step 2: Inverting the Rotation

Since rotation matrices are orthogonal, $R^{-1} = R^T$ (Anton & Rorres, 2013, *Elementary Linear Algebra*). Applying the inverse:

$$\begin{pmatrix} u \\ w \end{pmatrix} = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x - X \\ y - 42 \end{pmatrix}$$

Which gives us the **recovery formulas**:

$$t_{\text{recovered}} = (x - X)\cos\theta + (y - 42)\sin\theta$$

$$w_{\text{recovered}} = -(x - X)\sin\theta + (y - 42)\cos\theta$$

**Critical property:** The first equation recovers $t$ from any data point, using only $\theta$ and $X$ — it does not involve $M$ at all. This separates the parameter estimation problem.

### Step 3: The Linearity Test

For the correct $\theta$ and $X$, the recovered $w$ must satisfy:

$$w_{\text{recovered}} = e^{M|t|} \cdot \sin(0.3t)$$

Taking the natural logarithm of both sides (for points where $\sin(0.3t) > 0$):

$$\ln\left(\frac{w_{\text{recovered}}}{\sin(0.3t_{\text{recovered}})}\right) = M \cdot |t_{\text{recovered}}|$$

This means that if we plot $\ln(C/\sin(0.3t))$ against $|t|$, we should get a **perfect straight line through the origin** with slope $= M$. This linear relationship serves as both a verification criterion and a way to estimate $M$ via linear regression.

### Step 4: Global Optimization via Differential Evolution

Rather than manually iterating over grid points, we use **Differential Evolution** (Storn & Price, 1997) — a population-based stochastic optimization algorithm that:

1. Maintains a population of candidate solutions
2. Creates new candidates by combining existing ones (mutation + crossover)
3. Selects survivors based on fitness (lower cost = better)
4. Converges to the global optimum without requiring gradient information

The **cost function** minimizes the sum of squared residuals:

$$\mathcal{L}(\theta, M, X) = \sum_{i=1}^{1500} \left( w_{\text{recovered},i} - e^{M|t_{\text{recovered},i}|} \cdot \sin(0.3 \cdot t_{\text{recovered},i}) \right)^2$$

### Step 5: Local Refinement via L-BFGS-B

After global optimization finds the approximate basin, **L-BFGS-B** (Byrd et al., 1995) — a limited-memory quasi-Newton method for bound-constrained problems — refines the solution to machine precision.

---

## 📊 Results & Verification

### Fit Quality Metrics

| Metric | Value |
|--------|-------|
| Mean L1 distance (per point) | $2.06 \times 10^{-5}$ |
| Max L1 distance | $5.53 \times 10^{-5}$ |
| Sum L1 distance (all points) | $0.031$ |
| Sum of squared residuals | $4.38 \times 10^{-7}$ |
| Recovered $t$ range | $[6.05, 59.99]$ |

The tiny residuals (~$10^{-5}$) are due to floating-point rounding in the CSV data (values stored to 6 decimal places). The fit is **exact to within numerical precision**.

### Visual Verification

#### 1. Data Points vs. Fitted Curve
The red curve passes perfectly through all 1,500 blue data points:

![Curve Fit](plots/curve_fit.png)

#### 2. Residual Distribution
Residuals are tightly centered around zero with magnitude ~$10^{-5}$, confirming the fit quality:

![Residual Analysis](plots/residual_analysis.png)

#### 3. Linearity Verification
$\ln(C/\sin(0.3t))$ vs $|t|$ forms a perfect straight line with slope $M = 0.03$:

![Linearity Check](plots/linearity_check.png)

---

## 🛠️ How to Reproduce

### Prerequisites

```bash
pip install numpy pandas scipy matplotlib
```

### Running the Solution

```bash
# Step 1: Run the main solver
python solve_parametric_curve.py

# Step 2: Run independent verification
python verify_solution.py
```

### Expected Output

```
FINAL RESULTS
======================================================================
  theta = 30.000000 deg ~ 30 deg = pi/6 ~ 0.5235987756 rad
  M     = 0.0300000000 ~ 0.03
  X     = 55.0000000000 ~ 55

VERIFICATION PASSED
  Mean L1 = 2.06e-05 < threshold 1e-03
```

---

## 📁 Repository Structure

```
FLAM/
├── README.md                    # This file — solution explanation
├── xy_data.csv                  # Given data: 1,500 (x, y) points
├── solve_parametric_curve.py    # Main solver script
├── verify_solution.py           # Independent verification script
└── plots/
    ├── curve_fit.png            # Data vs. fitted curve overlay
    ├── residual_analysis.png    # Residual distribution + L1 per t
    └── linearity_check.png     # log-linearity verification
```

---

## 📖 References

1. **Storn, R. & Price, K.** (1997). "Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces." *Journal of Global Optimization*, 11(4), 341–359. [DOI: 10.1023/A:1008202821328](https://doi.org/10.1023/A:1008202821328)

2. **Byrd, R.H., Lu, P., Nocedal, J. & Zhu, C.** (1995). "A Limited Memory Algorithm for Bound Constrained Optimization." *SIAM Journal on Scientific Computing*, 16(5), 1190–1208. [DOI: 10.1137/0916069](https://doi.org/10.1137/0916069)

3. **Preparata, F.P. & Shamos, M.I.** (1985). *Computational Geometry: An Introduction*. Springer-Verlag, New York.

4. **Anton, H. & Rorres, C.** (2013). *Elementary Linear Algebra: Applications Version* (11th ed.). Wiley. (Chapter 4: Rotation Matrices and Orthogonal Transformations)

5. **Weisstein, E.W.** "Parametric Equations." *MathWorld — A Wolfram Web Resource*. [https://mathworld.wolfram.com/ParametricEquations.html](https://mathworld.wolfram.com/ParametricEquations.html)

6. **Weisstein, E.W.** "Rotation Matrix." *MathWorld — A Wolfram Web Resource*. [https://mathworld.wolfram.com/RotationMatrix.html](https://mathworld.wolfram.com/RotationMatrix.html)

7. **SciPy Documentation.** `scipy.optimize.differential_evolution`. [https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)

---

## 📝 Summary

The parametric curve fitting problem was solved by recognizing the underlying **rotation + translation** structure of the equations. By applying the inverse rotation, the parameter $t$ can be recovered for each data point using only $\theta$ and $X$. This reduces the problem to a standard nonlinear optimization, which was solved using **Differential Evolution** (global search) followed by **L-BFGS-B** (local refinement). The solution **θ = 30°, M = 0.03, X = 55** produces a perfect fit with mean L1 distance of $2.06 \times 10^{-5}$.
