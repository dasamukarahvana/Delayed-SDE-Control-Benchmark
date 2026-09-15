

# Delayed-SDE Control Benchmark

Reproducible Numerical Verification and Stochastic Optimization of a Delayed SDE Control System

This repository provides a reproducible numerical benchmark for a linear stochastic differential system with discrete delayed proportional-derivative (PD) control.

The primary objective is not to establish a general optimal-control theorem, but to construct and numerically audit a complete computational pipeline in which the implemented dynamics, delayed controller, finite-dimensional augmented representation, analytical moment oracle, automatic differentiation, Monte Carlo estimation, timestep refinement, stability diagnostics, stochastic optimization, and independent out-of-sample evaluation are mutually consistent.

The implementation is written in JAX with 64-bit floating-point arithmetic and is designed to make the numerical assumptions, verification procedures, and claim boundaries explicit.

---

### 1. Model

We consider the delayed stochastic system

$$
dx_t = \left( -\gamma x_t + u_{t-\tau} \right) dt + \sigma \, dW_t,
$$

where

- $x_t$ is the scalar system state,
- $u_t$ is the control input,
- $\gamma > 0$ is the linear damping coefficient,
- $\sigma \geq 0$ is the noise amplitude,
- $W_t$ is a standard Wiener process,
- $\tau \geq 0$ is the control delay.

The tracking error is defined as

$$
e_k = x_k - x^\star,
$$

where $x^\star$ is the prescribed reference target.

The discrete PD controller is

$$
u_k = -K_p e_k - K_d \frac{e_k - e_{k-1}}{\Delta t}.
$$

The delay is represented on the numerical grid by

$$
\tau = D\Delta t, \qquad D \in \mathbb{N}.
$$

---

### 2. Numerical Discretization

The continuous-time model is discretized using Euler--Maruyama:

$$
x_{k+1} = x_k + \left( -\gamma x_k + u_{k-D} \right)\Delta t + \sigma\sqrt{\Delta t} \, \xi_k,
$$

with

$$
\xi_k \sim \mathcal{N}(0, 1).
$$

The implementation therefore defines a precise discrete-time stochastic model. Analytical results labelled as exact in this repository refer to this implemented finite-dimensional discrete model, not to the original continuous-time delayed SDE.

---

### 3. Objective Functional

The instantaneous quadratic cost is

$$
\ell_k = q e_k^2 + r u_k^2,
$$

with $q > 0$ and $r \geq 0$.

Only a fixed terminal portion of the trajectory is used for the reported objective. If the tail contains $N_{\mathrm{tail}}$ steps,

$$
J(\theta) = \frac{1}{N_{\mathrm{tail}}} \sum_{k=k_{\mathrm{tail}}}^{H-1} \ell_k,
$$

where

$$
\theta = (K_p, K_d).
$$

The benchmark therefore evaluates the controller according to its mean tail cost rather than an arbitrarily selected transient value.

---

### 4. Important Equilibrium Observation

Because the model contains damping but no integral action or explicit feed-forward compensation, the controller does not, in general, produce exact tracking of a nonzero target.

For the deterministic equilibrium,

$$
0 = -\gamma x_{\mathrm{eq}} + u_{\mathrm{eq}},
$$

while

$$
u_{\mathrm{eq}} = -K_p(x_{\mathrm{eq}} - x^\star).
$$

Consequently,

$$
x_{\mathrm{eq}} = \frac{K_p}{\gamma + K_p}x^\star,
$$

and

$$
e_{\mathrm{eq}} = -\frac{\gamma}{\gamma + K_p}x^\star.
$$

Thus, for

$$
\gamma > 0, \qquad x^\star \neq 0,
$$

the equilibrium tracking error is generally nonzero.

This is an explicit model property and is not treated as a numerical defect.

Accordingly, this repository makes no claim of exact target tracking.

---

### 5. Finite-Dimensional Delayed Representation

The delayed system is converted into a finite-dimensional Markov representation.

For $D > 0$, define

$$
z_k = \begin{bmatrix}
x_k \\
e_{k-1} \\
u_{k-D} \\
u_{k-D+1} \\
\vdots \\
u_{k-1}
\end{bmatrix}.
$$

The affine controller can be written as

$$
u_k = a_x x_k + a_e e_{k-1} + a_0,
$$

where

$$
a_x = -K_p - \frac{K_d}{\Delta t},
$$

$$
a_e = \frac{K_d}{\Delta t},
$$

and

$$
a_0 = \left( K_p + \frac{K_d}{\Delta t} \right) x^\star.
$$

The augmented system therefore has the form

$$
z_{k+1} = A z_k + b + G \xi_k.
$$

For $D > 0$,

$$
A = \begin{pmatrix}
1 - \gamma\Delta t & 0 & \Delta t & 0 & \cdots & 0 \\
1 & 0 & 0 & 0 & \cdots & 0 \\
0 & 0 & 0 & 1 & \cdots & 0 \\
0 & 0 & 0 & 0 & \ddots & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & 1 \\
a_x & a_e & 0 & 0 & \cdots & 0
\end{pmatrix},
$$

with

$$
b = \begin{bmatrix}
0 \\
-x^\star \\
0 \\
\vdots \\
0 \\
a_0
\end{bmatrix},
$$

and

$$
G = \begin{bmatrix}
\sigma\sqrt{\Delta t} \\
0 \\
\vdots \\
0
\end{bmatrix}.
$$

The $D=0$ case is handled explicitly rather than being treated as a degenerate buffer operation.

---

### 6. Exact Discrete Linear-Gaussian Oracle

For fixed controller parameters, the augmented system is affine and Gaussian.

If

$$
\mu_k = \mathbb{E}[z_k], \qquad P_k = \operatorname{Cov}(z_k),
$$

then

$$
\mu_{k+1} = A\mu_k + b,
$$

and

$$
P_{k+1} = A P_k A^\top + G G^\top.
$$

For any affine observable

$$
y_k = c^\top z_k + d,
$$

its second moment is exactly

$$
\mathbb{E}[y_k^2] = (c^\top \mu_k + d)^2 + c^\top P_k c.
$$

In particular,

$$
e_k = x_k - x^\star
$$

and

$$
u_k = a_x x_k + a_e e_{k-1} + a_0
$$

are affine observables of the augmented state.

Therefore,

$$
\mathbb{E}[\ell_k] = q \, \mathbb{E}[e_k^2] + r \, \mathbb{E}[u_k^2]
$$

can be evaluated without Monte Carlo sampling.

The resulting finite-horizon oracle is an exact analytical expectation for the implemented discrete linear-Gaussian system.

It is not an exact solution of the continuous-time delayed SDE.

---

### 7. Verification Pipeline

The benchmark is organized as a sequence of independently interpretable verification stages.

#### V1 — Structural and Invariant Audit

Checks include:

- direct controller versus affine controller equivalence,
- initial-state definition,
- dimensional/state representation consistency,
- finite-value checks.

Result: "PASS"

---

#### V2A — Deterministic One-Step Equivalence

The direct simulator and the augmented matrix representation are compared for one deterministic step.

For the reference configuration,

$$
\gamma = 0.5, \qquad \sigma = 0.15, \qquad x^\star = 1, \qquad \Delta t = 0.01, \qquad D = 5,
$$

the implementation obtains

$$
\max |\Delta z| \approx 4.44 \times 10^{-16}.
$$

Result: "PASS"

---

#### V2B — Deterministic Finite-Horizon Oracle

The complete deterministic trajectory generated by the direct implementation is compared against the augmented-state oracle.

For the reported configuration,

$$
\max |\Delta z| = 7.77 \times 10^{-16},
$$

$$
\max |\Delta e| = 7.77 \times 10^{-16},
$$

$$
\max |\Delta u| = 2.998 \times 10^{-15},
$$

and

$$
|\Delta J| = 1.249 \times 10^{-16}.
$$

The agreement is at floating-point roundoff level.

Result: "PASS"

---

#### V2C — Monte Carlo Versus Exact Discrete Oracle

The Monte Carlo estimator is compared with the exact finite-horizon expectation obtained from the linear-Gaussian moment recursion.

For $N = 4096$,

$$
J_{\mathrm{oracle}} = 0.1191530250,
$$

while the Monte Carlo estimate in the reported run was

$$
J_{\mathrm{MC}} = 0.1194963562.
$$

The standardized discrepancy was

$$
z \approx 0.422.
$$

Thus the Monte Carlo result is statistically consistent with the analytical discrete oracle.

Result: "PASS"

---

#### V2D — Gaussian Mean and Covariance Audit

The terminal empirical mean and covariance are compared against the exact Gaussian moments.

The covariance comparison uses the finite-sample Gaussian covariance scale

$$
\operatorname{Var}(\widehat P_{ij}) \approx \frac{P_{ii}P_{jj} + P_{ij}^2}{N - 1}.
$$

The reported maximum standardized covariance discrepancy was approximately

$$
1.53.
$$

The empirical covariance was also checked for symmetry and positive semidefiniteness.

Result: "PASS"

---

#### V2E — Exact Moment Decomposition

For each affine observable, the numerical implementation verifies

$$
\mathbb{E}[Y^2] = \left(\mathbb{E}[Y]\right)^2 + \operatorname{Var}(Y).
$$

The maximum decomposition residuals in the reported run were approximately

$$
1.25 \times 10^{-16}
$$

for the error and

$$
2.78 \times 10^{-17}
$$

for the control.

Result: "PASS"

---

### 8. Automatic Differentiation Verification

#### V3 — AD Versus Central Finite Difference

The JAX automatic gradient is independently compared with a central finite-difference approximation using the same Monte Carlo realization.

For step sizes

$$
h \in \{10^{-2}, 3\times10^{-3}, 10^{-3}, 3\times10^{-4}, 10^{-4}, 3\times10^{-5}, 10^{-5}\},
$$

the relative discrepancy decreases into the numerical roundoff regime.

At

$$
h = 10^{-5},
$$

the reported relative $L^2$ discrepancy was

$$
8.96 \times 10^{-11}.
$$

This provides numerical evidence that the implemented computational graph is differentiated consistently.

It does not constitute a general mathematical proof of automatic differentiation correctness.

Result: "PASS"

---

### 9. Monte Carlo Scaling

#### V4 — Empirical $N^{-1/2}$ Scaling

The Monte Carlo estimator was evaluated over the predefined sample sizes

$$
N \in \{128, 256, 512, 1024, 2048, 4096\}.
$$

The empirical regression

$$
\log(\operatorname{SD}) = \alpha\log(N) + c
$$

gave

$$
\alpha \approx -0.475,
$$

with

$$
R^2 \approx 0.962.
$$

The bootstrap confidence interval for the fitted exponent contained

$$
-\frac{1}{2}.
$$

This is consistent with the expected Monte Carlo scaling

$$
\operatorname{SD} \propto N^{-1/2}
$$

over the tested finite sample range.

It is not a proof of an asymptotic central-limit theorem or universal $N^{-1/2}$ behavior.

Result: "PASS"

---

### 10. Timestep Refinement

#### V5 — Fixed Physical Delay and Horizon

The timestep was refined while keeping the physical delay

$$
\tau = 0.05
$$

and physical simulation horizon fixed.

The tested timesteps were

$$
\Delta t \in \{0.01, 0.005, 0.0025, 0.00125\},
$$

with reference timestep

$$
\Delta t_{\mathrm{ref}} = 0.000625.
$$

The corresponding delay-grid sizes were

$$
D \in \{5, 10, 20, 40, 80\}.
$$

Nested Brownian increments were used so that coarse Brownian increments are constructed consistently from the finer realization.

The reported objective errors decreased monotonically with refinement:

$$
2.297 \times 10^{-3}, \quad 1.832 \times 10^{-3}, \quad 1.472 \times 10^{-3}, \quad 9.897 \times 10^{-4}.
$$

The global log-log empirical order was approximately

$$
p \approx 0.396,
$$

with

$$
R^2 \approx 0.977.
$$

This demonstrates numerical refinement toward the selected finer-resolution reference along the tested realization.

It does not constitute a formal Euler--Maruyama convergence proof.

Result: "PASS"

---

### 11. Discrete Stability and Stationary Moments

#### V6 — Augmented-System Stability

For the baseline controller,

$$
(K_p, K_d) = (1, 0.1),
$$

the spectral radius of the augmented transition matrix was

$$
\rho(A) = 0.9858055971.
$$

Since

$$
\rho(A) < 1,
$$

the tested discrete augmented system is Schur-stable.

The corresponding stability margin is

$$
1 - \rho(A) = 0.0141944029.
$$

The stationary mean agrees with the structural equilibrium:

$$
\mu_x = 0.6666666667,
$$

$$
\mu_e = -0.3333333333,
$$

$$
\mu_u \approx 0.3333333333.
$$

The covariance recursion converges to a numerically consistent fixed point satisfying the discrete Lyapunov equation

$$
P_\infty = A P_\infty A^\top + G G^\top.
$$

Result: "PASS"

**Scope of the stability result**

This verifies the implemented discrete augmented linear system for the tested parameter configuration.

It does not establish:

- stability for arbitrary controller gains,
- a continuous-time delay stability theorem,
- nonlinear stochastic stability,
- robustness under model uncertainty,
- or physical-system stability.

---

### 12. Stochastic Projected Optimization

#### V7 — Projected Stochastic Gradient Benchmark

The controller parameters

$$
\theta = (K_p, K_d)
$$

are optimized numerically according to

$$
\theta_{n+1} = \Pi_{\Theta} \left[ \theta_n - \eta\widehat{\nabla J}(\theta_n) \right],
$$

where $\Pi_\Theta$ projects onto the predefined feasible parameter domain.

The benchmark uses:

- multiple independent optimization seeds,
- fresh Monte Carlo noise during optimization,
- a fixed evaluation batch,
- an independent diagnostic batch,
- JAX automatic differentiation,
- explicit parameter projection,
- no post-hoc parameter correction,
- no hidden objective modification.

The four optimization seeds produced final gains clustered around

$$
K_p \approx 1.413,
$$

$$
K_d \approx 0.112.
$$

The fixed evaluation batch showed approximately $35.6\% - 35.9\%$ improvement relative to the baseline.

The projected-gradient mapping decreased over the optimization budget, but its final value remained substantially above the predefined stationarity tolerance.

Therefore, the optimization result is not described as a proven stationary point or global optimum.

Result: "PASS"

---

### 13. Independent Out-of-Sample Evaluation

#### V8 — Frozen Candidate OOS Test

After optimization, the candidate controller is frozen before the independent evaluation.

The selected candidate was

$$
\boxed{
K_p = 1.4143864328, \qquad K_d = 0.1157534854
}
$$

and was selected from the predefined fixed evaluation procedure.

The independent OOS test uses

$$
N = 8192
$$

trajectories with seed

$$
909090.
$$

The baseline and candidate use exactly the same OOS noise realizations, enabling a paired comparison.

The resulting mean costs were

$$
J_{\mathrm{baseline}} = 0.1191735230,
$$

and

$$
J_{\mathrm{candidate}} = 0.0748907557.
$$

The paired mean difference was

$$
\Delta J = J_{\mathrm{candidate}} - J_{\mathrm{baseline}} = -0.0442827673.
$$

The relative improvement was

$$
\boxed{37.16\%}
$$

and the candidate won on all $8192$ paired trajectories in the reported run.

The normal 95% confidence interval for the paired mean difference was

$$
[-0.0446596, -0.0439059],
$$

while the bootstrap 95% interval was

$$
[-0.0446636, -0.0439019].
$$

Both intervals exclude zero.

The exact finite-horizon linear-Gaussian oracle independently predicts

$$
J_{\mathrm{baseline}}^{\mathrm{oracle}} = 0.1191530250,
$$

$$
J_{\mathrm{candidate}}^{\mathrm{oracle}} = 0.0748660888,
$$

with an oracle relative improvement of approximately

$$
37.17\%.
$$

Thus, the independent Monte Carlo result and the analytical discrete oracle agree in direction and magnitude.

Result: "PASS"

---

### 14. Verification Summary

| Verification | Result | Interpretation |
| :--- | :--- | :--- |
| V1 Structural / invariant audit | PASS | Direct and structural definitions agree |
| V2A One-step equivalence | PASS | Augmented transition agrees with direct dynamics |
| V2B Deterministic oracle | PASS | Finite-horizon trajectory agrees to roundoff |
| V2C MC vs analytical oracle | PASS | MC estimate is statistically consistent |
| V2D Gaussian moments | PASS | Empirical moments agree with Gaussian prediction |
| V2E Moment decomposition | PASS | Second-moment identity verified |
| V3 AD vs finite difference | PASS | Numerical gradient consistency demonstrated |
| V4 Monte Carlo scaling | PASS | Empirical $N^{-1/2}$ scaling supported |
| V5 Timestep refinement | PASS | Objective approaches finer reference |
| V6 Stability / stationary moments | PASS | Tested discrete system is Schur-stable |
| V7 Stochastic optimization | PASS | Reproducible improvement across seeds |
| V8 Independent OOS test | PASS | Frozen candidate improves independent evaluation |

---

### 15. Main Numerical Result

For the tested model and numerical configuration,

$$
\boxed{
(K_p, K_d) = (1.4143864328, 0.1157534854)
}
$$

produced a lower independent OOS tail cost than the predefined baseline

$$
(K_p, K_d)_{\mathrm{baseline}} = (1, 0.1).
$$

The measured OOS improvement was

$$
\boxed{37.16\%}.
$$

The result is independently supported by the exact finite-horizon linear-Gaussian oracle for the implemented discrete model.

---

### 16. Claim Boundary

The results support the following claim:

«Under the explicitly defined delayed stochastic model, discretization, controller class, parameter domain, objective functional, optimization protocol, and independent OOS evaluation procedure, the frozen candidate controller achieved a lower mean tail cost than the predefined baseline.»

The evidence does not establish:

- global optimality;
- uniqueness of the reported controller;
- universal superiority over the baseline;
- optimality for the continuous-time delayed SDE;
- convergence of the stochastic optimization algorithm to a global optimum;
- a general continuous-time delay-stability theorem;
- nonlinear stochastic stability;
- robustness to arbitrary model uncertainty;
- physical validity of the model;
- experimental validation;
- exact target tracking.

The reported $37.16\%$ improvement should therefore be interpreted strictly as a numerical result for the tested model and protocol.

---

### 17. Reproducibility

The benchmark uses deterministic seed management and explicitly records:

- JAX version;
- 64-bit configuration;
- model parameters;
- timestep;
- delay;
- horizon;
- tail-window definition;
- controller parameters;
- optimization seeds;
- Monte Carlo seeds;
- OOS seed;
- sample sizes;
- verification tolerances;
- numerical diagnostics.

The reference implementation is intended to run as a self-contained JAX workflow without requiring hidden state from previous notebook cells.

The computational pipeline does not use:

- "nan_to_num" masking;
- silent numerical fallback;
- hidden parameter corrections;
- undocumented hard-coded physical corrections;
- post-hoc candidate modification;
- cosmetic metric substitution.

Parameter projection is used only where explicitly defined as part of the optimization problem.

---

### 18. Computational Environment

The reference benchmark was executed with:

JAX version : 0.11.1  
x64 enabled : True  

The implementation uses:

- JAX
- NumPy
- Python standard library

and is intended to be compatible with a Google Colab-style environment.

---

### 19. Scientific Scope

This repository should be understood as a numerical verification and benchmarking study, rather than a claim of a new control-theoretic theorem.

Its main methodological purpose is to demonstrate a reproducible chain:

$$
\boxed{
\text{Model} \rightarrow \text{Discretization} \rightarrow \text{Implementation} \rightarrow \text{Analytical Oracle} \rightarrow \text{Numerical Verification} \rightarrow \text{Optimization} \rightarrow \text{Independent OOS Test}
}
$$

with explicit separation between what is analytically established, what is numerically verified, what is empirically observed, and what remains outside the scope of the study.

---

### 20. Status

Full verification pipeline: "PASS"

All predefined verification stages V1--V8 passed in the reported run.

The appropriate interpretation is:

«A reproducible numerical benchmark demonstrating internal consistency, analytical-oracle agreement, numerical differentiation consistency, empirical Monte Carlo scaling, timestep refinement, tested discrete stability, stochastic optimization reproducibility, and independent out-of-sample improvement for the specified delayed stochastic control model.»

It is not a proof of globally optimal stochastic control and should not be interpreted as such.
