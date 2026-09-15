# Delayed-SDE Control Benchmark

## Reproducible Numerical Verification and Stochastic Optimization of a Delayed Stochastic Control System

This repository presents a reproducible numerical benchmark for a stochastic dynamical system controlled by a discrete proportional-derivative (PD) controller with a finite control delay.

The study focuses on **numerical consistency, analytical verification, reproducibility, and independent out-of-sample evaluation** rather than on claiming a general optimal-control theorem.

The computational pipeline combines:

- an explicit delayed stochastic model;
- Euler–Maruyama discretization;
- a finite-dimensional augmented-state representation;
- an exact finite-horizon linear-Gaussian moment oracle;
- Monte Carlo verification;
- automatic-differentiation verification;
- Monte Carlo scaling analysis;
- timestep refinement;
- discrete stability and stationary-moment analysis;
- stochastic projected-gradient optimization;
- frozen-candidate independent out-of-sample evaluation.

The implementation is written in **JAX with 64-bit floating-point arithmetic**.

---

## 1. Research Objective

The central computational question is:

> Can a delayed stochastic control system be implemented in a way that is internally consistent with an analytically tractable finite-dimensional representation, while allowing reproducible gradient-based optimization and independent numerical validation?

The verification hierarchy is

$$
\text{model}
\rightarrow
\text{discretization}
\rightarrow
\text{implementation}
\rightarrow
\text{analytical oracle}
\rightarrow
\text{numerical verification}
\rightarrow
\text{optimization}
\rightarrow
\text{independent OOS evaluation}.
$$

The objective is methodological as well as computational: each numerical claim is tied to an explicit verification stage.

---

## 2. Mathematical Model

We consider the delayed stochastic differential equation

$$
dx_t
=
\left(
-\gamma x_t
+
u_{t-\tau}
\right)dt
+
\sigma\,dW_t,
$$

where:

- $x_t$ is the scalar system state;
- $u_t$ is the control input;
- $\gamma>0$ is the damping coefficient;
- $\sigma\geq0$ is the noise amplitude;
- $W_t$ is a standard Wiener process;
- $\tau\geq0$ is the control delay.

The discrete tracking error is

$$
e_k=x_k-x^\star,
$$

where $x^\star$ is the prescribed target.

The discrete proportional-derivative controller is

$$
u_k
=
-K_p e_k
-
K_d
\frac{e_k-e_{k-1}}{\Delta t}.
$$

The delay is represented on the numerical grid as

$$
\tau=D\Delta t,
\qquad
D\in\mathbb{N}.
$$

---

## 3. Numerical Discretization

The continuous-time model is discretized using Euler–Maruyama:

$$
x_{k+1}
=
x_k
+
\left(
-\gamma x_k
+
u_{k-D}
\right)\Delta t
+
\sigma\sqrt{\Delta t}\,\xi_k,
$$

with

$$
\xi_k\sim\mathcal{N}(0,1).
$$

The analytical oracle in this repository is exact for the implemented **discrete-time linear-Gaussian model**, not an exact solution of the original continuous-time delayed SDE.

---

## 4. Cost Functional

The instantaneous quadratic cost is

$$
\ell_k
=
q e_k^2
+
r u_k^2,
$$

where

$$
q>0,
\qquad
r\geq0.
$$

The reported objective is the mean cost over a fixed tail interval:

$$
J(\theta)
=
\frac{1}{N_{\mathrm{tail}}}
\sum_{k=k_{\mathrm{tail}}}^{H-1}
\ell_k,
$$

where

$$
\theta=(K_p,K_d).
$$

For the reference configuration,

$$
\Delta t=0.01,
\qquad
H=600,
$$

with tail duration

$$
T_{\mathrm{tail}}=0.5,
$$

giving

$$
N_{\mathrm{tail}}=50.
$$

---

## 5. Equilibrium and Target-Tracking Limitation

The deterministic equilibrium satisfies

$$
0
=
-\gamma x_{\mathrm{eq}}
+
u_{\mathrm{eq}}.
$$

At equilibrium,

$$
e_{\mathrm{eq}}
=
x_{\mathrm{eq}}-x^\star,
$$

and therefore

$$
u_{\mathrm{eq}}
=
-K_p e_{\mathrm{eq}}.
$$

Combining these relations gives

$$
x_{\mathrm{eq}}
=
\frac{K_p}{\gamma+K_p}x^\star,
$$

and

$$
e_{\mathrm{eq}}
=
-\frac{\gamma}{\gamma+K_p}x^\star.
$$

Hence, for

$$
\gamma>0,
\qquad
x^\star\neq0,
$$

the implemented controller generally has a nonzero steady-state tracking error.

Therefore, this repository makes **no exact-target-tracking claim**.

---

## 6. Affine Controller Representation

The controller can be rewritten as an affine function of $x_k$ and $e_{k-1}$:

$$
u_k
=
a_x x_k
+
a_e e_{k-1}
+
a_0.
$$

with

$$
a_x
=
-K_p-\frac{K_d}{\Delta t},
$$

$$
a_e
=
\frac{K_d}{\Delta t},
$$

and

$$
a_0
=
\left(
K_p+\frac{K_d}{\Delta t}
\right)x^\star.
$$

V1 verifies that this affine representation is numerically equivalent to the direct controller implementation.

---

## 7. Finite-Dimensional Augmented System

For $D>0$, define

$$
z_k
=
\begin{bmatrix}
x_k\\
e_{k-1}\\
u_{k-D}\\
u_{k-D+1}\\
\vdots\\
u_{k-1}
\end{bmatrix}.
$$

The delayed system then becomes a finite-dimensional affine stochastic recurrence,

$$
z_{k+1}
=
Az_k+b+G\xi_k.
$$

For the reference configuration $D=5$, the state dimension is

$$
n=D+2=7.
$$

The transition matrix is

$$
A=
\begin{pmatrix}
1-\gamma\Delta t & 0 & \Delta t & 0 & \cdots & 0\\
1 & 0 & 0 & 0 & \cdots & 0\\
0 & 0 & 0 & 1 & \cdots & 0\\
0 & 0 & 0 & 0 & \ddots & 0\\
\vdots & \vdots & \vdots & \vdots & \ddots & 1\\
a_x & a_e & 0 & 0 & \cdots & 0
\end{pmatrix},
$$

with

$$
b=
\begin{bmatrix}
0\\
-x^\star\\
0\\
\vdots\\
0\\
a_0
\end{bmatrix},
$$

and

$$
G=
\begin{bmatrix}
\sigma\sqrt{\Delta t}\\
0\\
\vdots\\
0
\end{bmatrix}.
$$

The $D=0$ case is handled explicitly in the implementation.

---

## 8. Exact Linear-Gaussian Moment Oracle

Because the augmented system is affine with Gaussian innovations, its first two moments satisfy exact recurrences.

Define

$$
\mu_k
=
\mathbb{E}[z_k],
$$

and

$$
P_k
=
\operatorname{Cov}(z_k).
$$

Then

$$
\mu_{k+1}
=
A\mu_k+b,
$$

and

$$
P_{k+1}
=
AP_kA^\top
+
GG^\top.
$$

For an affine observable

$$
y_k=c^\top z_k+d,
$$

the exact second moment is

$$
\mathbb{E}[y_k^2]
=
\left(
c^\top\mu_k+d
\right)^2
+
c^\top P_kc.
$$

The error and control are affine observables, so

$$
\mathbb{E}[\ell_k]
=
q\,\mathbb{E}[e_k^2]
+
r\,\mathbb{E}[u_k^2]
$$

can be evaluated analytically.

The resulting oracle is exact for the finite-horizon discrete model implemented in this repository.

---

# 9. Verification Architecture

The benchmark is divided into eight verification stages.

## V1 — Structural / Invariant Audit

V1 checks:

1. direct-controller versus affine-controller equivalence;
2. initial-state definition;
3. structural consistency of the implementation.

**Status: `PASS`**

---

## V2A — Deterministic One-Step Equivalence

The direct numerical update is compared with the augmented-state transition

$$
z_{k+1}=Az_k+b+G\xi_k.
$$

For the reported configuration,

$$
\max|\Delta z|
=
4.44\times10^{-16}.
$$

**Status: `PASS`**

---

## V2B — Deterministic Finite-Horizon Oracle

The complete deterministic trajectory is compared between:

1. the direct simulator;
2. the augmented-state oracle.

The reported maximum discrepancies are

$$
\max|\Delta z|
=
7.77\times10^{-16},
$$

$$
\max|\Delta e|
=
7.77\times10^{-16},
$$

$$
\max|\Delta u|
=
2.998\times10^{-15},
$$

and

$$
|\Delta J|
=
1.249\times10^{-16}.
$$

These values are consistent with floating-point roundoff.

**Status: `PASS`**

---

## V2C — Monte Carlo Mean Versus Exact Oracle

For $N=4096$ trajectories, the reported values are

$$
J_{\mathrm{oracle}}
=
0.1191530250,
$$

and

$$
J_{\mathrm{MC}}
=
0.1194963562.
$$

The standardized discrepancy is

$$
z\approx0.422.
$$

The result is statistically consistent with the exact discrete linear-Gaussian expectation.

**Status: `PASS`**

---

## V2D — Gaussian Mean and Covariance Audit

The terminal empirical moments are compared against the analytical Gaussian moments.

The reported standardized discrepancies were

$$
\max |Z_{\mathrm{mean}}|
\approx1.14,
$$

and

$$
\max |Z_{\mathrm{cov}}|
\approx1.53.
$$

The empirical covariance is also checked for:

- symmetry;
- positive semidefiniteness;
- finite entries.

**Status: `PASS`**

---

## V2E — Exact Moment Decomposition

The implementation verifies

$$
\mathbb{E}[Y^2]
=
\left(\mathbb{E}[Y]\right)^2
+
\operatorname{Var}(Y)
$$

for the affine error and control observables.

The reported maximum decomposition residuals are approximately

$$
1.25\times10^{-16}
$$

for the error and

$$
2.78\times10^{-17}
$$

for the control.

**Status: `PASS`**

---

## 10. Automatic Differentiation Verification

### V3 — AD Versus Central Finite Difference

The JAX automatic gradient is compared with central finite differences,

$$
\frac{\partial J}{\partial\theta_i}
\approx
\frac{
J(\theta+h e_i)
-
J(\theta-h e_i)
}{
2h
},
$$

using common random numbers.

The tested step sizes are

$$
h\in
\left\{
10^{-2},
3\times10^{-3},
10^{-3},
3\times10^{-4},
10^{-4},
3\times10^{-5},
10^{-5}
\right\}.
$$

At

$$
h=10^{-5},
$$

the relative $L^2$ discrepancy was

$$
8.96\times10^{-11}.
$$

This provides strong numerical evidence of consistency between the implemented computational graph and its JAX derivative.

It is not a general mathematical proof of automatic-differentiation correctness.

**Status: `PASS`**

---

## 11. Monte Carlo Scaling

### V4 — Empirical $N^{-1/2}$ Scaling

The Monte Carlo sample sizes are

$$
N=
128,\;256,\;512,\;1024,\;2048,\;4096.
$$

The empirical relationship

$$
\operatorname{SD}(\widehat J)
\propto
N^\alpha
$$

was fitted on a log-log scale.

The empirical exponent is

$$
\alpha=-0.4746,
$$

with

$$
R^2=0.9624.
$$

The bootstrap confidence interval for the exponent contains

$$
-\frac12.
$$

Thus the observed scaling is consistent with the standard Monte Carlo prediction over the tested finite sample range.

This is an empirical numerical observation, not a proof of asymptotic Monte Carlo convergence or a central-limit theorem.

**Status: `PASS`**

---

## 12. Timestep Refinement

### V5 — Fixed Physical Delay and Horizon

The physical delay is held fixed:

$$
\tau=0.05.
$$

The physical simulation horizon is also held fixed.

The tested timesteps are

$$
\Delta t
\in
\left\{
0.01,\;
0.005,\;
0.0025,\;
0.00125
\right\},
$$

with reference timestep

$$
\Delta t_{\mathrm{ref}}=0.000625.
$$

The corresponding delay-grid sizes are

$$
D=
5,\;10,\;20,\;40,\;80.
$$

Nested Brownian refinement is used so that coarse increments are constructed consistently from the finer Brownian realization.

The observed errors relative to the reference decrease monotonically:

$$
2.297\times10^{-3},
\quad
1.832\times10^{-3},
\quad
1.472\times10^{-3},
\quad
9.897\times10^{-4}.
$$

The empirical global order is approximately

$$
p=0.396,
$$

with

$$
R^2=0.977.
$$

This supports numerical refinement toward the selected finer-resolution reference along the tested realization.

It does **not** establish a formal Euler–Maruyama convergence theorem.

**Status: `PASS`**

---

## 13. Discrete Stability

### V6 — Augmented-System Stability and Stationary Moments

For the predefined baseline

$$
(K_p,K_d)=(1,0.1),
$$

the augmented transition matrix has spectral radius

$$
\rho(A)=0.9858055971.
$$

Since

$$
\rho(A)<1,
$$

the tested discrete augmented system is Schur-stable.

The stability margin is

$$
1-\rho(A)
=
0.0141944029.
$$

The stationary mean agrees with the structural equilibrium:

$$
\mu_x
=
0.6666666667,
$$

$$
\mu_e
=
-0.3333333333,
$$

$$
\mu_u
\approx
0.3333333333.
$$

The stationary covariance satisfies the discrete Lyapunov equation

$$
P_\infty
=
AP_\infty A^\top
+
GG^\top
$$

to numerical tolerance.

**Status: `PASS`**

### Stability Claim Boundary

This result establishes Schur stability only for the **implemented discrete augmented system at the tested parameter configuration**.

It does not establish:

- stability for arbitrary $(K_p,K_d)$;
- continuous-time delay stability;
- nonlinear stochastic stability;
- robustness under model uncertainty;
- physical-system stability.

---

## 14. Stochastic Optimization

### V7 — Projected Stochastic Gradient Optimization

The controller parameters are optimized using

$$
\theta_{n+1}
=
\Pi_\Theta
\left[
\theta_n
-
\eta\widehat{\nabla J}(\theta_n)
\right],
$$

where $\Pi_\Theta$ is the explicit projection onto the predefined feasible parameter domain.

The optimization protocol uses:

- multiple independent seeds;
- fresh Monte Carlo noise during optimization;
- a fixed evaluation batch;
- an independent diagnostic batch;
- JAX automatic differentiation;
- explicit parameter projection;
- a fixed optimization budget.

The final gains across the four optimization seeds cluster around

$$
K_p\approx1.413,
$$

and

$$
K_d\approx0.112.
$$

The fixed evaluation batch showed approximately $35.6\%-35.9\%$ improvement relative to the baseline for the four final iterates.

The projected-gradient mapping decreased during optimization.

However, the final projected-gradient norms remain above the predefined stationarity tolerance.

Therefore:

> The optimization is reported as a successful numerical improvement procedure, not as a proof of convergence to a stationary point or global optimum.

**Status: `PASS`**

---

## 15. Independent Out-of-Sample Evaluation

### V8 — Frozen Candidate Evaluation

After the optimization stage, the candidate controller is frozen before the independent evaluation.

The selected candidate is

$$
\boxed{
(K_p,K_d)
=
(1.4143864328,\;0.1157534854)
}.
$$

The predefined baseline is

$$
(K_p,K_d)_{\mathrm{base}}
=
(1.0,\;0.1).
$$

The independent OOS evaluation uses

$$
N=8192
$$

trajectories with a separate random seed.

The baseline and candidate are evaluated on exactly the same noise realizations, producing a paired comparison.

The measured mean costs are

$$
J_{\mathrm{base}}
=
0.1191735230,
$$

and

$$
J_{\mathrm{cand}}
=
0.0748907557.
$$

The paired mean difference is

$$
\Delta J
=
J_{\mathrm{cand}}
-
J_{\mathrm{base}}
=
-0.0442827673.
$$

The relative improvement is therefore

$$
\boxed{
37.16\%
}.
$$

The paired 95% normal confidence interval is

$$
[-0.0446596,\,-0.0439059],
$$

and the bootstrap 95% confidence interval is

$$
[-0.0446636,\,-0.0439019].
$$

Both intervals exclude zero.

The candidate won on all $8192$ paired trajectories in the reported run.

---

## 16. Independent Analytical Cross-Check

The exact finite-horizon linear-Gaussian oracle independently gives

$$
J_{\mathrm{base}}^{\mathrm{oracle}}
=
0.1191530250,
$$

and

$$
J_{\mathrm{cand}}^{\mathrm{oracle}}
=
0.0748660888.
$$

The corresponding oracle improvement is approximately

$$
37.17\%.
$$

The agreement between the OOS Monte Carlo result and the analytical oracle provides an independent consistency check on the reported direction and magnitude of the improvement.

**Status: `PASS`**

---

## 17. Summary of Verification Results

| Stage | Status | Primary purpose |
|---|---|---|
| V1 | PASS | Structural and invariant consistency |
| V2A | PASS | One-step augmented-system equivalence |
| V2B | PASS | Deterministic finite-horizon oracle |
| V2C | PASS | Monte Carlo versus exact oracle |
| V2D | PASS | Gaussian mean/covariance consistency |
| V2E | PASS | Exact second-moment decomposition |
| V3 | PASS | AD versus finite-difference gradient |
| V4 | PASS | Empirical Monte Carlo scaling |
| V5 | PASS | Timestep refinement |
| V6 | PASS | Discrete stability and stationary moments |
| V7 | PASS | Stochastic optimization reproducibility |
| V8 | PASS | Independent frozen-candidate evaluation |

---

## 18. Main Numerical Result

For the explicitly specified model and numerical protocol, the frozen candidate

$$
\boxed{
(K_p,K_d)
=
(1.4143864328,\;0.1157534854)
}
$$

achieved a lower independent OOS tail cost than the predefined baseline

$$
(K_p,K_d)=(1,0.1).
$$

The observed improvement was

$$
\boxed{37.16\%}.
$$

The same qualitative result is independently reproduced by the exact finite-horizon linear-Gaussian oracle.

---

## 19. Claim Boundary

The numerical evidence supports the following claim:

> **Under the explicitly defined delayed stochastic model, discretization, controller class, feasible parameter domain, objective functional, optimization protocol, and independent OOS evaluation procedure, the frozen candidate controller achieved a lower mean tail cost than the predefined baseline.**

The study does **not** establish:

- global optimality;
- uniqueness of the candidate;
- universal superiority;
- continuous-time optimality;
- convergence of the optimization algorithm to a global optimum;
- a general continuous-time delay-stability theorem;
- nonlinear stochastic stability;
- robustness to arbitrary model uncertainty;
- physical validity of the model;
- experimental validation;
- exact target tracking.

The reported $37.16\%$ improvement is therefore a result for the **specified model and computational protocol**, not a universal control-law claim.

---

## 20. Reproducibility

The benchmark explicitly records:

- JAX version;
- 64-bit configuration;
- model parameters;
- timestep;
- delay;
- physical horizon;
- tail-window definition;
- controller parameters;
- random seeds;
- Monte Carlo sample sizes;
- optimization configuration;
- verification tolerances;
- OOS evaluation configuration.

The implementation is designed as a self-contained JAX workflow suitable for a Google Colab environment.

The numerical pipeline avoids:

- silent numerical fallback;
- hidden parameter correction;
- undocumented physical correction;
- post-hoc modification of the selected candidate;
- cosmetic replacement of failed diagnostics.

The parameter projection

$$
\Pi_\Theta:\mathbb{R}^2\rightarrow\Theta
$$

is an explicit part of the optimization algorithm rather than a hidden numerical correction.

---

## 21. Computational Environment

The reported benchmark was executed using:

```text
JAX version : 0.11.1
x64 enabled : True
