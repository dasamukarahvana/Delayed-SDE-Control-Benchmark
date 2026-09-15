

# Delayed SDE Control Benchmark

## Rigorous Numerical Verification of a Delayed Stochastic Feedback System

A reproducible JAX/X64 numerical benchmark for a delayed stochastic first-order dynamical system with discrete PD control.

The project is designed as a **numerical verification and benchmarking study**, not as a proof of global optimality or a general stability theorem.

The central principle is:

> **What is claimed must be exactly what is implemented and tested.**

---

## 1. Research Question

Consider the delayed stochastic system

$$
dx_t =
\left(
-\gamma x_t + u_{t-\tau}
\right)dt
+
\sigma\,dW_t,
$$

with a discrete proportional-derivative controller

$$
u_k
=
-K_p e_k
-
K_d
\frac{e_k-e_{k-1}}{\Delta t},
$$

where

$$
e_k = x_k-x^\star.
$$

The numerical study asks:

1. Does the implemented controller have an internally consistent affine representation?
2. Does the finite-dimensional augmented-state representation reproduce the direct simulation?
3. Does the exact finite-horizon moment recursion agree with Monte Carlo estimates?
4. Does automatic differentiation agree with an independent finite-difference gradient calculation?
5. Does Monte Carlo error exhibit the expected approximately
   $$N^{-1/2}$$
   scaling in the tested regime?
6. Does timestep refinement provide numerical evidence of refinement toward a finer-resolution reference?
7. Does the discrete augmented system satisfy the tested deterministic Schur-stability criterion?
8. Can stochastic projected-gradient optimization produce a candidate controller that improves the predefined baseline on an independent out-of-sample evaluation?

The study deliberately does **not** attempt to establish a global optimum, universal controller superiority, or physical validation.

---

# 2. Mathematical Model

## 2.1 Continuous-Time Model

The stochastic delayed system is

$$
dx_t =
\left(
-\gamma x_t + u_{t-\tau}
\right)dt
+
\sigma\,dW_t.
$$

Parameters:

- $\gamma$ — damping coefficient,
- $\sigma$ — diffusion amplitude,
- $\tau$ — physical control delay,
- $W_t$ — standard Wiener process.

The target state is

$$
x^\star.
$$

The tracking error is

$$
e_t=x_t-x^\star.
$$

---

## 2.2 Discrete Controller

The controller is evaluated at discrete times

$$
t_k=k\Delta t.
$$

The error is

$$
e_k=x_k-x^\star.
$$

The discrete PD controller is

$$
u_k
=
-K_p e_k
-
K_d
\frac{e_k-e_{k-1}}{\Delta t}.
$$

Equivalently,

$$
u_k
=
\left(
-K_p-\frac{K_d}{\Delta t}
\right)x_k
+
\frac{K_d}{\Delta t}e_{k-1}
+
\left(
K_p+\frac{K_d}{\Delta t}
\right)x^\star.
$$

Define

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

Then

$$
u_k
=
a_xx_k+a_ee_{k-1}+a_0.
$$

This affine representation is used by the augmented-state formulation.

---

# 3. Numerical Discretization

Euler-Maruyama discretization gives

$$
x_{k+1}
=
x_k
+
\left(
-\gamma x_k+u_{k-D}
\right)\Delta t
+
\sigma\sqrt{\Delta t}\,\xi_k,
$$

where

$$
\xi_k\sim\mathcal N(0,1),
$$

independently across time steps.

The discrete delay is

$$
\tau=D\Delta t,
$$

with integer delay index

$$
D\in\mathbb N.
$$

The implementation explicitly distinguishes the zero-delay case $D=0$ from the delayed case $D>0$.

---

# 4. Objective Function

The instantaneous quadratic cost is

$$
\ell_k
=
q e_k^2
+
r u_k^2.
$$

The reported objective is the mean cost over a fixed tail window:

$$
J
=
\frac{1}{N_{\mathrm{tail}}}
\sum_{k\in\mathrm{tail}}
\ell_k.
$$

The tail window is fixed before optimization and evaluation.

This objective measures the numerical control performance of the specified discrete model.

It is **not** an experimentally measured physical performance metric.

---

# 5. Structural Equilibrium

For constant deterministic equilibrium,

$$
x_{k+1}=x_k=x_{\rm eq},
$$

and

$$
u_k=u_{\rm eq}.
$$

At equilibrium the derivative term vanishes, giving

$$
u_{\rm eq}
=
-K_p(x_{\rm eq}-x^\star).
$$

The plant equilibrium condition is

$$
0=-\gamma x_{\rm eq}+u_{\rm eq}.
$$

Therefore,

$$
x_{\rm eq}
=
\frac{K_p}{\gamma+K_p}x^\star.
$$

The equilibrium error is

$$
e_{\rm eq}
=
-\frac{\gamma}{\gamma+K_p}x^\star.
$$

Thus, for

$$
\gamma>0,
\qquad
x^\star\neq0,
$$

the implemented PD controller does **not** generally achieve exact target tracking.

This benchmark therefore makes no claim of exact steady-state tracking.

---

# 6. Augmented-State Representation

For delayed dynamics, define

$$
z_k
=
\begin{bmatrix}
x_k\\
e_{k-1}\\
B_k
\end{bmatrix},
$$

where the delay buffer contains the previously computed controls.

For $D>0$,

$$
B_k
=
\begin{bmatrix}
u_{k-D}\\
u_{k-D+1}\\
\vdots\\
u_{k-1}
\end{bmatrix}.
$$

The augmented dimension is

$$
n=D+2.
$$

The state transition can be written as

$$
z_{k+1}
=
Az_k+b+G\xi_k.
$$

For $D>0$, the implemented matrix has the structure

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

with affine vector

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

and noise vector

$$
G=
\begin{bmatrix}
\sigma\sqrt{\Delta t}\\
0\\
\vdots\\
0
\end{bmatrix}.
$$

The implementation computes observables from $z_k$ **before** applying the transition to $z_{k+1}$.

This ordering is important for consistency between direct simulation and the augmented representation.

---

# 7. Exact Finite-Horizon Linear-Gaussian Oracle

Because the implemented augmented system is affine and Gaussian,

$$
z_{k+1}=Az_k+b+G\xi_k,
$$

its finite-horizon first and second moments satisfy the exact recursions

$$
\mu_{k+1}
=
A\mu_k+b,
$$

and

$$
P_{k+1}
=
AP_kA^\mathsf T+GG^\mathsf T.
$$

For an affine observable

$$
y_k=c^\mathsf Tz_k+d,
$$

the second moment is

$$
\mathbb E[y_k^2]
=
\left(
c^\mathsf T\mu_k+d
\right)^2
+
c^\mathsf TP_kc.
$$

Therefore,

$$
\mathbb E[\ell_k]
=
q\,\mathbb E[e_k^2]
+
r\,\mathbb E[u_k^2].
$$

The finite-horizon expected cost is obtained by summing these expected stage costs over the same tail window used by the Monte Carlo calculation.

### Important qualification

The oracle is **exact for the implemented discrete affine linear-Gaussian model over the finite simulated horizon**.

It is not an exact solution of the original continuous-time delayed stochastic differential equation.

---

# 8. Verification Protocol

The benchmark contains eight verification stages.

---

## V1 — Structural Verification

V1 checks:

- direct controller versus affine controller equivalence;
- initial-state definition;
- numerical residuals against predefined tolerance.

The test uses multiple controller configurations rather than relying on a single parameter point.

### Interpretation

A PASS establishes consistency of the tested algebraic implementation.

It does not constitute a formal dimensional-analysis proof of the entire project.

---

# V2 — Analytical Linear-Gaussian Oracle

V2 contains several independent consistency checks.

### V2A — Moment recursion

The deterministic mean/covariance recursion is checked against independently constructed expectations.

### V2B — Augmented-state consistency

The direct trajectory and augmented-state representation are compared at matching time indices.

The comparison includes:

- state,
- tracking error,
- control,
- stage cost,
- accumulated objective.

### V2C — Monte Carlo versus oracle

Monte Carlo estimates are compared against the exact finite-horizon expectation of the implemented discrete model.

### V2D — Terminal moments

Monte Carlo estimates of terminal mean and covariance are compared against the analytical moments.

### V2E — Cost decomposition

The direct cost calculation is compared against the decomposition into error and control contributions.

---

# V3 — Automatic Differentiation versus Finite Differences

The objective is differentiated using JAX automatic differentiation.

An independent central finite-difference estimate is

$$
\frac{\partial J}{\partial\theta_i}
\approx
\frac{
J(\theta_i+h)-J(\theta_i-h)
}{
2h
}.
$$

The same random-number realization is used for the paired evaluations so that the comparison is not dominated by independent Monte Carlo noise.

The tested relative step sizes are

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

### Observed behavior

The coarsest step,

$$
h=10^{-2},
$$

does **not** satisfy the predefined discrepancy tolerance.

For the smaller steps,

$$
3\times10^{-3}
\le h\le
10^{-5},
$$

the discrepancy decreases systematically and the tested cases pass.

At

$$
h=10^{-5},
$$

the maximum relative component discrepancy is approximately

$$
8.96\times10^{-11}.
$$

### Interpretation

V3 provides numerical evidence that the implemented automatic-differentiation gradient agrees with central finite differences in the tested finite-difference regime.

The coarse-step failure is retained as part of the verification record rather than being hidden.

---

# V4 — Monte Carlo Scaling

For an estimator with finite variance, the standard error is expected to scale approximately as

$$
\operatorname{SEM}\propto N^{-1/2}.
$$

The benchmark tests sample sizes

$$
N=
128,\,
256,\,
512,\,
1024,\,
2048,\,
4096.
$$

The fitted log-log slope is approximately

$$
\alpha=-0.475,
$$

with

$$
R^2\approx0.962.
$$

A bootstrap confidence interval for the fitted exponent contains the reference value

$$
-\frac12.
$$

### Interpretation

This provides numerical evidence that the observed Monte Carlo error scaling is compatible with the expected

$$
N^{-1/2}
$$

behavior in the tested regime.

It is not a proof of the central limit theorem or of asymptotic scaling for all sample sizes.

---

# V5 — Timestep Refinement

The physical delay is held fixed:

$$
\tau=0.05.
$$

Therefore the delay index changes with timestep:

$$
D=\frac{\tau}{\Delta t}.
$$

The tested timesteps are

$$
\Delta t
\in
\left\{
0.01,\,
0.005,\,
0.0025,\,
0.00125
\right\},
$$

with a finer reference calculation at

$$
\Delta t_{\rm ref}=0.000625.
$$

The physical simulation horizon is held fixed.

Nested Brownian refinement is used so that finer resolutions are constructed consistently from the same underlying Brownian increments.

The observed errors relative to the selected finer reference decrease monotonically across the tested resolutions.

The fitted empirical global order is approximately

$$
p\approx0.396,
$$

with

$$
R^2\approx0.977.
$$

### Interpretation

V5 provides **numerical evidence of refinement toward the selected finer-resolution reference**.

The measured empirical order is an observation of this numerical experiment.

It is not presented as a formal convergence theorem for the continuous-time delayed SDE.

---

# V6 — Discrete Stability and Stationary Moments

The deterministic augmented matrix $A$ is analyzed through its spectral radius

$$
\rho(A)
=
\max_i|\lambda_i(A)|.
$$

The tested baseline configuration gives

$$
\rho(A)\approx0.985806<1.
$$

For the tested discrete augmented system, this satisfies the Schur-stability criterion.

The stationary mean is obtained from

$$
\mu_\infty
=
A\mu_\infty+b,
$$

or equivalently

$$
(I-A)\mu_\infty=b.
$$

The stationary covariance satisfies the discrete Lyapunov equation

$$
P_\infty
=
AP_\infty A^\mathsf T
+
GG^\mathsf T.
$$

The implementation verifies:

- convergence of the covariance fixed-point iteration;
- Lyapunov residual;
- covariance symmetry;
- positive semidefiniteness within numerical precision;
- transient convergence of mean and covariance.

For the baseline configuration,

$$
x_\infty
=
0.6666666667,
$$

$$
e_\infty
=
-0.3333333333,
$$

and

$$
u_\infty
=
0.3333333333.
$$

These values agree with the structural equilibrium calculation.

### Interpretation

V6 supports consistency between the implemented discrete augmented dynamics, its spectral-radius calculation, and its stationary moment calculation.

It does not establish a general stability theorem for arbitrary parameters, nonlinear systems, or the original continuous-time delayed stochastic system.

---

# V7 — Stochastic Projected-Gradient Optimization

The controller parameters are

$$
\theta=
\begin{bmatrix}
K_p\\
K_d
\end{bmatrix}.
$$

Optimization uses projected stochastic gradient descent,

$$
\theta_{n+1}
=
\Pi_\Theta
\left[
\theta_n-\eta\widehat{\nabla J}_n
\right],
$$

where $\Pi_\Theta$ projects onto the predefined feasible parameter set.

The corresponding projected-gradient mapping is

$$
G_\eta(\theta)
=
\frac{
\theta-
\Pi_\Theta
\left(
\theta-\eta\nabla J(\theta)
\right)
}{
\eta
}.
$$

The optimization protocol uses:

- four independent optimization seeds;
- fresh Monte Carlo noise during optimization;
- a fixed evaluation batch that is not used for optimization;
- the final iterate from each optimization run;
- no best-iterate selection during the optimization trajectory.

The candidate-selection rule is predefined:

> Select the final iterate with the lowest fixed-batch evaluation objective among the four independent optimization seeds.

After selection, the candidate parameters are frozen.

A separate diagnostic evaluation is then performed before the final OOS test.

### Important interpretation

A V7 PASS means the optimization procedure satisfied the predefined numerical benchmark criteria.

It does **not** prove:

- global optimality;
- uniqueness;
- convergence to a global optimum;
- convergence to a stationary point in the mathematical optimization sense;
- optimality for the continuous-time control problem.

---

# V8 — Independent Out-of-Sample Evaluation

The predefined baseline is

$$
(K_p,K_d)_{\rm base}
=
(1.0,0.1).
$$

The frozen candidate selected by V7 is

$$
(K_p,K_d)_{\rm cand}
=
(1.4143864328,\,
0.1157534854).
$$

The final evaluation uses:

- independent OOS random seed;
- $N=8192$ trajectories;
- the same exact noise realization for baseline and candidate;
- paired trajectory-wise comparison;
- normal and bootstrap confidence intervals;
- win-rate calculation;
- independent analytical-oracle comparison.

The OOS Monte Carlo results are

$$
J_{\rm base}
=
0.1191735230,
$$

and

$$
J_{\rm cand}
=
0.0748907557.
$$

The paired mean difference is

$$
\Delta J
=
J_{\rm cand}-J_{\rm base}
=
-0.0442827673.
$$

The corresponding relative improvement is approximately

$$
37.16\%.
$$

The normal 95% confidence interval for the paired mean difference is

$$
[-0.0446596,\,-0.0439059].
$$

The bootstrap 95% confidence interval is

$$
[-0.0446636,\,-0.0439019].
$$

Both intervals exclude zero.

The candidate wins all tested paired trajectories:

$$
\text{win rate}=1.0.
$$

The independent oracle gives

$$
J_{\rm base}^{\rm oracle}
=
0.1191530250,
$$

and

$$
J_{\rm cand}^{\rm oracle}
=
0.0748660888.
$$

Thus,

$$
\Delta J_{\rm oracle}
=
-0.0442869362,
$$

corresponding to an oracle improvement of approximately

$$
37.17\%.
$$

The Monte Carlo and oracle results therefore agree closely for the tested finite-horizon discrete model.

---

# 9. Main Numerical Result

Under the explicitly defined model, timestep, delay, horizon, controller parameterization, optimization protocol, and independent OOS evaluation:

$$
(K_p,K_d)_{\rm cand}
=
(1.4143864328,\,
0.1157534854)
$$

produces a lower mean tail cost than the predefined baseline

$$
(K_p,K_d)_{\rm base}
=
(1.0,0.1).
$$

The measured OOS improvement is approximately

$$
37.16\%.
$$

The independent analytical finite-horizon oracle gives a consistent improvement of approximately

$$
37.17\%.
$$

This is the principal numerical finding of the benchmark.

---

# 10. Claim Boundary

## Supported by this benchmark

The study supports the following claim:

> **Under the explicitly implemented discrete delayed stochastic model and the predefined numerical protocol, the frozen candidate controller obtained from the stochastic projected-gradient procedure achieves lower mean tail cost than the predefined baseline on an independent out-of-sample Monte Carlo evaluation. The direction and magnitude of the result are independently consistent with the exact finite-horizon moment oracle of the implemented affine linear-Gaussian model.**

---

## Not established

The benchmark does **not** establish:

- global optimality;
- uniqueness of the reported controller;
- universal superiority over other controllers;
- optimality for the continuous-time control problem;
- a general convergence theorem for the delayed SDE;
- a general stochastic stability theorem;
- stability for arbitrary parameter choices;
- nonlinear stochastic stability;
- robustness to model uncertainty;
- robustness to disturbances outside the tested model;
- experimental or physical validity;
- real-world control performance;
- exact target tracking.

The reported controller should therefore be interpreted as a **numerically validated candidate under a specified benchmark**, not as a universally optimal controller.

---

# 11. Verification Status

| Test | Purpose | Status |
|---|---|---|
| V1 | Structural/controller consistency | PASS |
| V2A | Analytical moment recursion | PASS |
| V2B | Augmented-state consistency | PASS |
| V2C | Monte Carlo vs analytical expectation | PASS |
| V2D | Terminal moment consistency | PASS |
| V2E | Cost decomposition | PASS |
| V3 | AD vs finite difference | PASS* |
| V4 | Monte Carlo scaling | PASS |
| V5 | Timestep refinement | PASS |
| V6 | Discrete stability/stationary moments | PASS |
| V7 | Stochastic projected optimization | PASS |
| V8 | Independent OOS evaluation | PASS |

### V3 qualification

V3 is marked PASS because the predefined overall verification criterion is satisfied at the finer finite-difference steps.

However,

$$
h=10^{-2}
$$

fails the predefined discrepancy tolerance.

This failure is intentionally retained in the verification record.

A PASS status therefore does not mean that every individual sub-test at every resolution passed.

---

# 12. Reproducibility

The benchmark is implemented in JAX with 64-bit floating-point arithmetic enabled.

The principal configuration is:

| Parameter | Value |
|---|---:|
| $\gamma$ | $0.5$ |
| $\sigma$ | $0.15$ |
| $x^\star$ | $1.0$ |
| $q$ | $1.0$ |
| $r$ | $0.005$ |
| $\Delta t$ | $0.01$ |
| $D$ | $5$ |
| $\tau$ | $0.05$ |
| Horizon | $600$ steps |
| Tail window | $0.5$ |
| Baseline $K_p$ | $1.0$ |
| Baseline $K_d$ | $0.1$ |
| Optimization seeds | $101,202,303,404$ |
| OOS seed | $909090$ |
| OOS trajectories | $8192$ |
| Arithmetic | JAX float64 |

The implementation performs explicit finite-value validation.

There is no use of:

- `nan_to_num` as a hidden numerical correction;
- silent fallback algorithms;
- post-hoc modification of the selected controller;
- hidden objective substitution;
- cosmetic metric substitution;
- undisclosed parameter correction.

Projection during optimization is an explicit part of the predefined feasible-set constraint.

---

# 13. Numerical Environment

The reported run used:

```text
JAX: 0.11.1
x64: True
```

The code is intended to run in a standard Python/JAX environment and is compatible with Google Colab when the required dependencies are available.


---

14. Scientific Scope

This repository is best classified as a:

rigorous numerical verification / reproducible computational benchmark.

It combines:

stochastic differential-equation discretization;

delayed feedback control;

affine state-space representation;

exact finite-horizon linear-Gaussian moments;

automatic differentiation;

independent finite differences;

Monte Carlo verification;

timestep refinement;

discrete stability analysis;

stochastic optimization;

independent out-of-sample testing.


The analytical oracle is especially useful because the numerical Monte Carlo and optimization pipeline can be checked against an independently derived expectation for the implemented discrete model.


---

15. Limitations

Several limitations are deliberate.

Continuous-time versus discrete-time

The simulation uses Euler-Maruyama discretization.

Therefore, the numerical oracle is exact for the discrete implementation, not for the original continuous-time delayed SDE.

Linear model

The plant is linear in $x$ and $u$.

The results do not establish performance for nonlinear plants.

Fixed controller structure

Only the specified discrete PD controller is optimized.

The benchmark does not compare against arbitrary controller classes.

Fixed parameter domain

Optimization occurs inside a predefined feasible parameter region.

No claim is made about parameters outside that region.

Finite horizon

All reported performance results refer to the specified finite simulation horizon and tail window.

Model-specific result

The 37.16% improvement is a result for the explicitly defined benchmark configuration.

It should not be interpreted as a universal percentage improvement for delayed stochastic control.


---

16. Scientific Contract

The project follows a simple scientific contract:

> Write what is done. Do what is written.



Accordingly:

derivations are distinguished from numerical checks;

numerical evidence is distinguished from proof;

simulation observations are distinguished from analytical results;

empirical improvements are distinguished from optimality claims;

failed sub-tests are not silently removed;

numerical tolerances are predefined;

the candidate is frozen before OOS evaluation;

independent random seeds are used for final evaluation;

analytical and numerical routes are compared whenever possible.


The goal is not to make the benchmark appear stronger than it is.

The goal is to make the boundary between what has been demonstrated and what has not explicit.


---

17. Conclusion

The benchmark provides a reproducible numerical study of a delayed stochastic PD-controlled system.

The implemented pipeline passes the structural, analytical-oracle, automatic-differentiation, Monte Carlo, refinement, stability, optimization, and independent OOS checks defined by the project.

The strongest numerical result is that the frozen candidate

\[
(K_p,K_d)
=
(1.4143864328,\,
0.1157534854)
\]

reduces the mean tail cost relative to the predefined baseline

\[
(K_p,K_d)
=
(1.0,0.1)
\]

by approximately

\[
37.16\%
\]

on the independent OOS evaluation.

The corresponding analytical finite-horizon oracle predicts approximately

\[
37.17\%
\]

improvement.

These results support the claim of numerical superiority over the predefined baseline under the tested model and protocol.

They do not establish global optimality, universal superiority, continuous-time optimality, or physical validity.

