#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DELAYED-SDE CONTROL BENCHMARK â€” FULL JAX PIPELINE
=================================================

Self-contained JAX x64 numerical benchmark.

Scope
-----
Discrete Euler-Maruyama delayed stochastic control model

    dx_t = (-gamma*x_t + u_{t-tau}) dt + sigma dW_t

with discrete PD control

    e_k = x_k - target
    u_k = -Kp e_k - Kd (e_k-e_{k-1})/dt

and integer delay

    tau = D*dt.

Verification pipeline
---------------------
V1  structural / invariant audit
V2A deterministic one-step equivalence
V2B deterministic finite-horizon oracle
V2C Monte-Carlo mean vs exact discrete oracle
V2D Gaussian mean/covariance audit
V2E exact moment decomposition
V3  AD vs central finite difference
V4  Monte-Carlo N^(-1/2) scaling
V5  timestep refinement with fixed physical delay/horizon
V6  discrete augmented-system stability/stationary moments
V7  stochastic projected-gradient optimization
V8  frozen-candidate independent OOS test

Important claim boundary
------------------------
This script verifies the IMPLEMENTED DISCRETE LINEAR-GAUSSIAN MODEL.
It does not prove continuous-time delayed-SDE convergence, global optimality,
universal controller superiority, or physical validity.

No nan_to_num, silent fallback, hidden correction, or post-hoc clipping is used.
The only clipping is explicit projection of optimization parameters onto the
predeclared feasible box.

Test configuration is intentionally explicit and reproducible.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# ============================================================================
# 00. CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class ModelConfig:
    gamma: float = 0.5
    sigma: float = 0.15
    target: float = 1.0
    q: float = 1.0
    r: float = 0.005

    dt: float = 0.01
    D: int = 5
    horizon: int = 600
    tail_time: float = 0.5

    x0: float = 0.27

    @property
    def tau(self) -> float:
        return self.D * self.dt

    @property
    def tail_steps(self) -> int:
        n = int(round(self.tail_time / self.dt))
        if n <= 0 or n > self.horizon:
            raise ValueError("tail_steps must satisfy 1 <= tail_steps <= horizon")
        return n

    @property
    def tail_start(self) -> int:
        return self.horizon - self.tail_steps


@dataclass(frozen=True)
class OptimizationConfig:
    lr: float = 0.08
    max_iterations: int = 50
    mc_batch_size: int = 64
    projected_grad_tol: float = 1e-6

    kp_min: float = 0.0
    kp_max: float = 20.0
    kd_min: float = 0.0
    kd_max: float = 20.0


@dataclass(frozen=True)
class ExperimentConfig:
    master_seed: int = 20260914

    v1_cases: int = 16

    v2_mc_batch: int = 4096

    v3_mc_batch: int = 4096
    v3_fd_steps: Tuple[float, ...] = (
        1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5
    )
    v3_rel_l2_tol: float = 5e-5
    v3_max_component_rel_tol: float = 1e-4

    v4_replicates: int = 32
    v4_bootstrap: int = 10_000
    v4_ns: Tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096)

    v5_dts: Tuple[float, ...] = (0.01, 0.005, 0.0025, 0.00125)
    v5_dt_ref: float = 0.000625
    v5_physical_horizon: float = 6.0
    v5_tail_time: float = 0.5

    v6_stationary_max_iter: int = 20_000
    v6_stationary_tol: float = 1e-12
    v6_transient_steps: int = 20_000

    v7_seeds: Tuple[int, ...] = (101, 202, 303, 404)
    v7_eval_seed: int = 20267915
    v7_diag_seed: int = 20267916

    v8_seed: int = 909090
    v8_n: int = 8192
    v8_bootstrap: int = 10_000


MODEL = ModelConfig()
OPT = OptimizationConfig()
EXP = ExperimentConfig()

BASELINE_THETA = jnp.array([1.0, 0.1], dtype=jnp.float64)


# ============================================================================
# 01. BASIC VALIDATION / UTILITIES
# ============================================================================

def require_x64() -> None:
    if not bool(jax.config.x64_enabled):
        raise RuntimeError("JAX x64 is required but is disabled.")


def finite_or_raise(name: str, x: Any) -> None:
    a = np.asarray(x)
    if not np.all(np.isfinite(a)):
        raise FloatingPointError(f"{name} contains non-finite values.")


def max_abs(x: Any) -> float:
    return float(np.max(np.abs(np.asarray(x))))


def l2_norm(x: Any) -> float:
    return float(np.linalg.norm(np.asarray(x)))


def status(condition: bool) -> str:
    return "PASS" if bool(condition) else "FAIL"


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def normal_ci(mean: float, sem: float, z: float = 1.96) -> Tuple[float, float]:
    return mean - z * sem, mean + z * sem


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int,
    seed: int,
    chunk: int = 1000,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n < 2:
        raise ValueError("bootstrap requires at least two observations")

    rng = np.random.default_rng(seed)
    means = []

    for start in range(0, n_boot, chunk):
        nb = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(nb, n), dtype=np.int32)
        means.append(values[idx].mean(axis=1))

    boot = np.concatenate(means)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def paired_bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int,
    seed: int,
    chunk: int = 1000,
) -> Tuple[float, float]:
    return bootstrap_mean_ci(values, n_boot, seed, chunk)


def make_noise(seed: int, n: int, horizon: int) -> jax.Array:
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(
        key,
        shape=(n, horizon),
        dtype=jnp.float64,
    )


# ============================================================================
# 02. JAX PYTREE STATE
# ============================================================================

@jax.tree_util.register_pytree_node_class
@dataclass
class SystemState:
    x: jax.Array
    prev_e: jax.Array
    u_buffer: jax.Array

    def tree_flatten(self):
        children = (self.x, self.prev_e, self.u_buffer)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        x, prev_e, u_buffer = children
        return cls(
            x=children[0],
            prev_e=children[1],
            u_buffer=children[2],
        )


def initial_state(model: ModelConfig) -> SystemState:
    return SystemState(
        x=jnp.asarray(model.x0, dtype=jnp.float64),
        prev_e=jnp.asarray(model.x0 - model.target, dtype=jnp.float64),
        u_buffer=jnp.zeros((model.D,), dtype=jnp.float64),
    )


def initial_augmented_state(model: ModelConfig) -> jax.Array:
    s = initial_state(model)
    return jnp.concatenate(
        [
            jnp.reshape(s.x, (1,)),
            jnp.reshape(s.prev_e, (1,)),
            s.u_buffer,
        ]
    )


# ============================================================================
# 03. CONTROLLER
# ============================================================================

def controller_direct(
    theta: jax.Array,
    x: jax.Array,
    prev_e: jax.Array,
    model: ModelConfig,
) -> jax.Array:
    Kp, Kd = theta
    e = x - model.target
    derivative = (e - prev_e) / model.dt
    return -Kp * e - Kd * derivative


def controller_affine(
    theta: jax.Array,
    x: jax.Array,
    prev_e: jax.Array,
    model: ModelConfig,
) -> jax.Array:
    Kp, Kd = theta

    a_x = -Kp - Kd / model.dt
    a_e = Kd / model.dt
    a_0 = (Kp + Kd / model.dt) * model.target

    return a_x * x + a_e * prev_e + a_0


def controller_coefficients(
    theta: jax.Array,
    model: ModelConfig,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    Kp, Kd = theta

    a_x = -Kp - Kd / model.dt
    a_e = Kd / model.dt
    a_0 = (Kp + Kd / model.dt) * model.target

    return a_x, a_e, a_0


# ============================================================================
# 04. DIRECT MODEL STEP
# ============================================================================

def one_step(
    state: SystemState,
    noise: jax.Array,
    theta: jax.Array,
    model: ModelConfig,
) -> Tuple[SystemState, Tuple[jax.Array, jax.Array, jax.Array]]:
    """
    One Euler-Maruyama step.

    Important ordering:
        u_k is computed from x_k and e_{k-1}.
        The plant uses u_{k-D}.
        For D>0 this is u_buffer[0].
        For D=0 the current u_k is used.
    """

    x = state.x
    prev_e = state.prev_e

    e = x - model.target

    u_current = controller_direct(
        theta,
        x,
        prev_e,
        model,
    )

    if model.D > 0:
        u_delayed = state.u_buffer[0]

        if model.D == 1:
            new_buffer = jnp.reshape(u_current, (1,))
        else:
            new_buffer = jnp.concatenate(
                [
                    state.u_buffer[1:],
                    jnp.reshape(u_current, (1,)),
                ]
            )
    else:
        u_delayed = u_current
        new_buffer = jnp.zeros((0,), dtype=jnp.float64)

    drift = -model.gamma * x + u_delayed

    x_next = (
        x
        + drift * model.dt
        + model.sigma * jnp.sqrt(model.dt) * noise
    )

    next_state = SystemState(
        x=x_next,
        prev_e=e,
        u_buffer=new_buffer,
    )

    stage_cost = (
        model.q * e**2
        + model.r * u_current**2
    )

    return next_state, (x, e, u_current, stage_cost)


def trajectory_with_outputs(
    theta: jax.Array,
    noises: jax.Array,
    model: ModelConfig,
):
    s0 = initial_state(model)

    def scan_step(state, xi):
        state_next, output = one_step(
            state,
            xi,
            theta,
            model,
        )
        return state_next, output

    _, outputs = jax.lax.scan(
        scan_step,
        s0,
        noises,
    )

    xs, errors, controls, costs = outputs
    return xs, errors, controls, costs


def trajectory_cost(
    theta: jax.Array,
    noises: jax.Array,
    model: ModelConfig,
) -> jax.Array:
    _, errors, controls, costs = trajectory_with_outputs(
        theta,
        noises,
        model,
    )

    start = model.tail_start
    return jnp.mean(costs[start:])


# ============================================================================
# 05. AUGMENTED LINEAR SYSTEM
# ============================================================================

def build_augmented_system(
    theta: jax.Array,
    model: ModelConfig,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Build

        z_{k+1} = A z_k + b + G xi_k

    with

        z_k = [x_k, e_{k-1}, u_{k-D}, ..., u_{k-1}]

    for D>0.

    For D=0:

        z_k = [x_k, e_{k-1}].
    """

    dt = model.dt
    gamma = model.gamma
    sigma = model.sigma
    target = model.target

    a_x, a_e, a_0 = controller_coefficients(
        theta,
        model,
    )

    if model.D == 0:
        A = jnp.array(
            [
                [1.0 - gamma * dt + a_x * dt, a_e * dt],
                [1.0, 0.0],
            ],
            dtype=jnp.float64,
        )

        b = jnp.array(
            [
                a_0 * dt,
                -target,
            ],
            dtype=jnp.float64,
        )

        G = jnp.array(
            [
                sigma * jnp.sqrt(dt),
                0.0,
            ],
            dtype=jnp.float64,
        )

        return A, b, G

    n = model.D + 2

    A = jnp.zeros((n, n), dtype=jnp.float64)
    b = jnp.zeros((n,), dtype=jnp.float64)

    # x_{k+1} = (1-gamma*dt)x_k + dt*u_{k-D} + noise
    A = A.at[0, 0].set(1.0 - gamma * dt)
    A = A.at[0, 2].set(dt)

    # e_k = x_k - target
    A = A.at[1, 0].set(1.0)
    b = b.at[1].set(-target)

    # Shift delay buffer:
    # B_{k+1} = [u_{k-D+1}, ..., u_{k}]
    for j in range(model.D - 1):
        A = A.at[2 + j, 2 + j + 1].set(1.0)

    # Last buffer entry receives u_k
    A = A.at[model.D + 1, 0].set(a_x)
    A = A.at[model.D + 1, 1].set(a_e)
    b = b.at[model.D + 1].set(a_0)

    G = jnp.zeros((n,), dtype=jnp.float64)
    G = G.at[0].set(sigma * jnp.sqrt(dt))

    return A, b, G


def augmented_observables(
    z: jax.Array,
    theta: jax.Array,
    model: ModelConfig,
) -> Tuple[jax.Array, jax.Array]:
    """
    Observables at time k from z_k.

        e_k = x_k-target
        u_k = controller(x_k,e_{k-1})

    This ordering is critical for V2B.
    """

    x = z[0]
    prev_e = z[1]

    e = x - model.target

    u = controller_affine(
        theta,
        x,
        prev_e,
        model,
    )

    return e, u


def augmented_trajectory_with_outputs(
    theta: jax.Array,
    noises: jax.Array,
    model: ModelConfig,
):
    """
    Exact trajectory generated by the affine augmented representation.

    At each k:
        observables are computed from z_k,
        then z_{k+1} is generated.
    """

    A, b, G = build_augmented_system(
        theta,
        model,
    )

    z0 = initial_augmented_state(model)

    def scan_step(z, xi):
        e, u = augmented_observables(
            z,
            theta,
            model,
        )

        stage = (
            model.q * e**2
            + model.r * u**2
        )

        z_next = A @ z + b + G * xi

        return z_next, (z, e, u, stage)

    _, outputs = jax.lax.scan(
        scan_step,
        z0,
        noises,
    )

    states, errors, controls, costs = outputs

    return states, errors, controls, costs


# ============================================================================
# 06. EXACT FINITE-HORIZON LINEAR-GAUSSIAN ORACLE
# ============================================================================

def affine_second_moment(
    mu: jax.Array,
    P: jax.Array,
    c: jax.Array,
    d: jax.Array,
) -> jax.Array:
    mean_y = c @ mu + d
    var_y = c @ P @ c
    return mean_y**2 + var_y


def exact_moment_trajectory(
    theta: jax.Array,
    model: ModelConfig,
):
    """
    Exact first/second moments of the implemented discrete linear-Gaussian model.

    Initial state is deterministic:
        z_0 = initial_augmented_state(model)
        P_0 = 0.

    The returned arrays correspond to observables at k=0,...,H-1.
    """

    A, b, G = build_augmented_system(
        theta,
        model,
    )

    z0 = initial_augmented_state(model)
    P0 = jnp.zeros_like(A)

    n = A.shape[0]

    c_e = jnp.zeros((n,), dtype=jnp.float64)
    c_e = c_e.at[0].set(1.0)
    d_e = -model.target

    a_x, a_e, a_0 = controller_coefficients(
        theta,
        model,
    )

    c_u = jnp.zeros((n,), dtype=jnp.float64)
    c_u = c_u.at[0].set(a_x)
    c_u = c_u.at[1].set(a_e)
    d_u = a_0

    def step(carry, _):
        mu, P = carry

        mean_e = c_e @ mu + d_e
        var_e = c_e @ P @ c_e

        mean_u = c_u @ mu + d_u
        var_u = c_u @ P @ c_u

        second_e = mean_e**2 + var_e
        second_u = mean_u**2 + var_u

        stage = (
            model.q * second_e
            + model.r * second_u
        )

        mu_next = A @ mu + b
        P_next = A @ P @ A.T + jnp.outer(G, G)

        return (mu_next, P_next), (
            mu,
            P,
            mean_e,
            var_e,
            mean_u,
            var_u,
            second_e,
            second_u,
            stage,
        )

    (_, _), outputs = jax.lax.scan(
        step,
        (z0, P0),
        jnp.arange(model.horizon),
    )

    return outputs


def oracle_cost(
    theta: jax.Array,
    model: ModelConfig,
) -> jax.Array:
    outputs = exact_moment_trajectory(
        theta,
        model,
    )

    stages = outputs[-1]

    return jnp.mean(
        stages[model.tail_start:]
    )


def oracle_terminal_moments(
    theta: jax.Array,
    model: ModelConfig,
):
    outputs = exact_moment_trajectory(
        theta,
        model,
    )

    mu = outputs[0]
    P = outputs[1]

    return mu[-1], P[-1]


# ============================================================================
# 07. JIT FUNCTIONS
# ============================================================================
# ModelConfig is deliberately STATIC. This fixes the JAX error where a
# Python dataclass was passed as a dynamic argument to jit.

TRAJECTORY_COST_JIT = jax.jit(
    trajectory_cost,
    static_argnames=("model",),
)

AUG_TRAJECTORY_JIT = jax.jit(
    augmented_trajectory_with_outputs,
    static_argnames=("model",),
)

ORACLE_MOMENTS_JIT = jax.jit(
    exact_moment_trajectory,
    static_argnames=("model",),
)

ORACLE_COST_JIT = jax.jit(
    oracle_cost,
    static_argnames=("model",),
)


def batch_costs(
    theta: jax.Array,
    noise_batch: jax.Array,
    model: ModelConfig,
) -> jax.Array:
    return jax.vmap(
        lambda xi: trajectory_cost(
            theta,
            xi,
            model,
        )
    )(noise_batch)


BATCH_COSTS_JIT = jax.jit(
    batch_costs,
    static_argnames=("model",),
)


def batch_cost(
    theta: jax.Array,
    noise_batch: jax.Array,
    model: ModelConfig,
) -> jax.Array:
    return jnp.mean(
        batch_costs(
            theta,
            noise_batch,
            model,
        )
    )


BATCH_VALUE_AND_GRAD = jax.jit(
    jax.value_and_grad(batch_cost),
    static_argnames=("model",),
)


# ============================================================================
# 08. V1 â€” STRUCTURAL / INVARIANT AUDIT
# ============================================================================

def run_v1() -> Dict[str, Any]:
    print_section("V1 â€” STRUCTURAL / INVARIANT AUDIT")

    key = jax.random.PRNGKey(EXP.master_seed + 1)

    max_controller_residual = 0.0

    for _ in range(EXP.v1_cases):
        key, k1, k2 = jax.random.split(key, 3)

        theta = jnp.array(
            [
                jax.random.uniform(k1, (), minval=0.0, maxval=3.0),
                jax.random.uniform(k2, (), minval=0.0, maxval=0.5),
            ],
            dtype=jnp.float64,
        )

        x = jax.random.normal(k1, (), dtype=jnp.float64)
        prev_e = jax.random.normal(k2, (), dtype=jnp.float64)

        direct = controller_direct(
            theta,
            x,
            prev_e,
            MODEL,
        )

        affine = controller_affine(
            theta,
            x,
            prev_e,
            MODEL,
        )

        max_controller_residual = max(
            max_controller_residual,
            abs(float(direct - affine)),
        )

    s0 = initial_state(MODEL)

    initial_ok = (
        abs(float(s0.x) - MODEL.x0) == 0.0
        and abs(float(s0.prev_e) - (MODEL.x0 - MODEL.target)) == 0.0
        and s0.u_buffer.shape == (MODEL.D,)
    )

    controller_ok = max_controller_residual <= 1e-12

    overall = controller_ok and initial_ok

    print(
        f"controller/direct-affine equivalence"
        f"{' ' * 24}: {status(controller_ok)}"
    )
    print(
        f"initial-state definition"
        f"{' ' * 36}: {status(initial_ok)}"
    )
    print(f"V1 STATUS: {status(overall)}")

    return {
        "overall": overall,
        "max_controller_residual": max_controller_residual,
        "initial_ok": initial_ok,
    }


# ============================================================================
# 09. V2A â€” DETERMINISTIC ONE-STEP EQUIVALENCE
# ============================================================================

def run_v2a() -> Dict[str, Any]:
    print_section("V2A â€” DETERMINISTIC ONE-STEP EQUIVALENCE")

    theta = BASELINE_THETA
    s0 = initial_state(MODEL)
    xi = jnp.asarray(0.0, dtype=jnp.float64)

    s1, _ = one_step(
        s0,
        xi,
        theta,
        MODEL,
    )

    A, b, G = build_augmented_system(
        theta,
        MODEL,
    )

    z0 = initial_augmented_state(MODEL)
    z1 = A @ z0 + b + G * xi

    print("A =")
    print(np.array2string(np.asarray(A), precision=2, suppress_small=False))
    print("b =")
    print(np.asarray(b))
    print("G =")
    print(np.asarray(G))

    z1_direct = jnp.concatenate(
        [
            jnp.reshape(s1.x, (1,)),
            jnp.reshape(s1.prev_e, (1,)),
            s1.u_buffer,
        ]
    )

    print("z1 direct =")
    print(np.asarray(z1_direct))
    print("z1 augmented =")
    print(np.asarray(z1))

    residual = max_abs(z1_direct - z1)

    passed = residual <= 1e-13

    print(f"max residual = {residual:.16e}")
    print(
        f"V2A one-step equivalence"
        f"{' ' * 20}: {status(passed)}"
    )
    print(f"V2A STATUS: {status(passed)}")

    return {
        "overall": passed,
        "residual": residual,
        "A": A,
        "b": b,
        "G": G,
    }


# ============================================================================
# 10. V2B â€” DETERMINISTIC FINITE-HORIZON ORACLE
# ============================================================================

def run_v2b() -> Dict[str, Any]:
    print_section("V2B â€” DETERMINISTIC FINITE-HORIZON ORACLE")

    # Sigma=0 is required for a clean deterministic oracle comparison.
    deterministic_model = ModelConfig(
        gamma=MODEL.gamma,
        sigma=0.0,
        target=MODEL.target,
        q=MODEL.q,
        r=MODEL.r,
        dt=MODEL.dt,
        D=MODEL.D,
        horizon=MODEL.horizon,
        tail_time=MODEL.tail_time,
        x0=MODEL.x0,
    )

    theta = BASELINE_THETA
    noises = jnp.zeros(
        (deterministic_model.horizon,),
        dtype=jnp.float64,
    )

    direct = trajectory_with_outputs(
        theta,
        noises,
        deterministic_model,
    )

    aug = augmented_trajectory_with_outputs(
        theta,
        noises,
        deterministic_model,
    )

    direct_xs, direct_e, direct_u, direct_costs = direct
    aug_states, aug_e, aug_u, aug_costs = aug

    # IMPORTANT:
    # aug_states[k] is z_k, so x_k is aug_states[k,0].
    # Do not compare x_{k+1} to z_k.
    oracle_xs = aug_states[:, 0]

    state_residual = max_abs(
        direct_xs - oracle_xs
    )

    error_residual = max_abs(
        direct_e - aug_e
    )

    control_residual = max_abs(
        direct_u - aug_u
    )

    stage_residual = max_abs(
        direct_costs - aug_costs
    )

    start = deterministic_model.tail_start

    J_direct = float(
        jnp.mean(direct_costs[start:])
    )

    J_oracle = float(
        jnp.mean(aug_costs[start:])
    )

    J_residual = J_direct - J_oracle

    finite = all(
        np.isfinite(
            [
                state_residual,
                error_residual,
                control_residual,
                stage_residual,
                J_direct,
                J_oracle,
                J_residual,
            ]
        )
    )

    passed = (
        finite
        and state_residual <= 1e-12
        and error_residual <= 1e-12
        and control_residual <= 1e-12
        and stage_residual <= 1e-12
        and abs(J_residual) <= 1e-13
    )

    print(f"tail steps = {deterministic_model.tail_steps}")
    print(f"tail start = {deterministic_model.tail_start}")
    print(f"tail end   = {deterministic_model.horizon - 1}")
    print(f"max state residual = {state_residual:.16e}")
    print(f"max error residual = {error_residual:.16e}")
    print(f"max control residual = {control_residual:.16e}")
    print(f"max stage-cost residual = {stage_residual:.16e}")
    print(f"J_direct = {J_direct:.16e}")
    print(f"J_oracle = {J_oracle:.16e}")
    print(f"J residual = {J_residual:.16e}")
    print(
        f"V2B deterministic oracle agreement"
        f"{' ' * 14}: {status(passed)}"
    )
    print(f"V2B STATUS: {status(passed)}")

    return {
        "overall": passed,
        "state_residual": state_residual,
        "error_residual": error_residual,
        "control_residual": control_residual,
        "stage_residual": stage_residual,
        "J_direct": J_direct,
        "J_oracle": J_oracle,
        "J_residual": J_residual,
    }


# ============================================================================
# 11. V2C â€” MONTE-CARLO VS EXACT ORACLE
# ============================================================================

def run_v2c() -> Dict[str, Any]:
    print_section("V2C â€” MONTE-CARLO MEAN VS EXACT DISCRETE ORACLE")

    theta = BASELINE_THETA

    noise = make_noise(
        EXP.master_seed + 20,
        EXP.v2_mc_batch,
        MODEL.horizon,
    )

    costs = BATCH_COSTS_JIT(
        theta,
        noise,
        MODEL,
    )

    oracle = ORACLE_COST_JIT(
        theta,
        MODEL,
    )

    costs_np = np.asarray(costs)

    mc_mean = float(np.mean(costs_np))
    sample_var = float(np.var(costs_np, ddof=1))
    sem = float(np.sqrt(sample_var / costs_np.size))

    oracle_value = float(oracle)

    diff = mc_mean - oracle_value
    z = diff / sem if sem > 0 else np.nan

    finite = (
        np.all(np.isfinite(costs_np))
        and np.isfinite(oracle_value)
        and np.isfinite(sem)
    )

    # 4-sigma is used as a diagnostic acceptance criterion here,
    # not as a universal theorem or family-wise statistical guarantee.
    passed = (
        finite
        and sem > 0
        and abs(z) <= 4.0
    )

    print(f"oracle expected cost = {oracle_value:.16e}")
    print(f"MC mean = {mc_mean:.16e}")
    print(f"MC sample variance = {sample_var:.16e}")
    print(f"SEM = {sem:.16e}")
    print(f"MC - oracle = {diff:+.16e}")
    print(f"z-score = {z:+.12f}")
    print(f"MC finite = {np.all(np.isfinite(costs_np))}")
    print(f"Oracle finite = {np.isfinite(oracle_value)}")
    print(f"tail {MODEL.tail_steps}, start {MODEL.tail_start}, end {MODEL.horizon-1}")
    print(f"V2C STATUS: {status(passed)}")

    return {
        "overall": passed,
        "oracle": oracle_value,
        "mc_mean": mc_mean,
        "sample_var": sample_var,
        "sem": sem,
        "difference": diff,
        "z": z,
    }


# ============================================================================
# 12. V2D â€” GAUSSIAN MEAN / COVARIANCE AUDIT
# ============================================================================

def run_v2d() -> Dict[str, Any]:
    print_section("V2D â€” GAUSSIAN MEAN / COVARIANCE AUDIT")

    theta = BASELINE_THETA

    noise = make_noise(
        EXP.master_seed + 21,
        EXP.v2_mc_batch,
        MODEL.horizon,
    )

    costs = BATCH_COSTS_JIT(theta, noise, MODEL)

    # For the terminal augmented state we need the full augmented trajectories.
    def one_aug(xi):
        return augmented_trajectory_with_outputs(
            theta,
            xi,
            MODEL,
        )[0][-1]

    terminal_states = jax.vmap(one_aug)(noise)
    terminal_np = np.asarray(terminal_states)

    mc_mean = np.mean(terminal_np, axis=0)
    mc_cov = np.cov(terminal_np, rowvar=False, ddof=1)

    oracle_mu, oracle_P = oracle_terminal_moments(
        theta,
        MODEL,
    )

    oracle_mu_np = np.asarray(oracle_mu)
    oracle_P_np = np.asarray(oracle_P)

    mean_res = mc_mean - oracle_mu_np
    cov_res = mc_cov - oracle_P_np

    # Gaussian covariance uncertainty:
    # Var(S_ij) ~= (P_ii P_jj + P_ij^2)/(N-1)
    N = terminal_np.shape[0]
    diag = np.diag(oracle_P_np)

    cov_se = np.sqrt(
        np.maximum(
            (diag[:, None] * diag[None, :] + oracle_P_np**2)
            / max(N - 1, 1),
            0.0,
        )
    )

    z_cov = np.zeros_like(cov_res)
    mask = cov_se > 0
    z_cov[mask] = cov_res[mask] / cov_se[mask]

    mean_se = np.sqrt(
        np.maximum(diag, 0.0) / N
    )

    z_mean = np.zeros_like(mean_res)
    mask_m = mean_se > 0
    z_mean[mask_m] = mean_res[mask_m] / mean_se[mask_m]

    eig_mc = np.linalg.eigvalsh(
        0.5 * (mc_cov + mc_cov.T)
    )
    eig_oracle = np.linalg.eigvalsh(
        0.5 * (oracle_P_np + oracle_P_np.T)
    )

    deterministic_entries = np.where(
        np.abs(np.diag(oracle_P_np)) == 0
    )[0]

    deterministic_residual = 0.0
    for i in deterministic_entries:
        deterministic_residual = max(
            deterministic_residual,
            abs(mean_res[i]),
        )

    covariance_symmetry_mc = max_abs(
        mc_cov - mc_cov.T
    )
    covariance_symmetry_oracle = max_abs(
        oracle_P_np - oracle_P_np.T
    )

    max_mean_z = float(np.max(np.abs(z_mean)))
    max_cov_z = float(np.max(np.abs(z_cov)))

    finite = all(
        np.isfinite(
            [
                max_abs(mean_res),
                l2_norm(mean_res),
                max_abs(cov_res),
                np.linalg.norm(cov_res),
                max_mean_z,
                max_cov_z,
            ]
        )
    )

    # Diagnostic thresholds, explicitly scoped to this finite Monte-Carlo test.
    passed = (
        finite
        and max_mean_z <= 4.0
        and max_cov_z <= 4.0
        and deterministic_residual <= 1e-12
        and covariance_symmetry_mc <= 1e-12
        and covariance_symmetry_oracle <= 1e-12
        and float(np.min(eig_mc)) >= -1e-12
        and float(np.min(eig_oracle)) >= -1e-12
    )

    print(f"terminal mean max residual = {max_abs(mean_res):.16e}")
    print(f"terminal mean L2 residual = {l2_norm(mean_res):.16e}")
    print(f"terminal covariance max abs residual = {max_abs(cov_res):.16e}")
    print(f"terminal covariance Frobenius residual = {np.linalg.norm(cov_res):.16e}")
    print(f"max standardized |Z_mean| = {max_mean_z:.16e}")
    print(f"max standardized |Z_cov| = {max_cov_z:.16e}")
    print(f"max deterministic-entry residual = {deterministic_residual:.16e}")
    print(f"symmetry MC = {covariance_symmetry_mc:.16e}")
    print(f"symmetry oracle = {covariance_symmetry_oracle:.16e}")
    print(f"PSD min MC eig = {float(np.min(eig_mc)):.16e}")
    print(f"PSD min oracle eig = {float(np.min(eig_oracle)):.16e}")
    print(f"V2D STATUS: {status(passed)}")

    return {
        "overall": passed,
        "max_mean_z": max_mean_z,
        "max_cov_z": max_cov_z,
        "max_cov_residual": max_abs(cov_res),
        "min_eig_mc": float(np.min(eig_mc)),
        "min_eig_oracle": float(np.min(eig_oracle)),
    }


# ============================================================================
# 13. V2E â€” MOMENT DECOMPOSITION AUDIT
# ============================================================================

def run_v2e() -> Dict[str, Any]:
    print_section("V2E â€” EXACT MOMENT DECOMPOSITION")

    theta = BASELINE_THETA

    noise = make_noise(
        EXP.master_seed + 22,
        EXP.v2_mc_batch,
        MODEL.horizon,
    )

    xs, errors, controls, costs = (
        jax.vmap(
            lambda xi: trajectory_with_outputs(
                theta,
                xi,
                MODEL,
            )
        )(noise)
    )

    e_np = np.asarray(errors)
    u_np = np.asarray(controls)
    cost_np = np.asarray(costs)

    start = MODEL.tail_start

    e_tail = e_np[:, start:]
    u_tail = u_np[:, start:]
    c_tail = cost_np[:, start:]

    # Exact finite-sample identity uses ddof=0:
    # mean(y^2) = mean(y)^2 + mean((y-mean)^2)
    e_second_raw = np.mean(e_tail**2)
    e_mean_square = np.mean(e_tail, axis=0)**2
    e_var_population = np.mean(
        (e_tail - np.mean(e_tail, axis=0))**2,
        axis=0,
    )

    u_second_raw = np.mean(u_tail**2)
    u_mean_square = np.mean(u_tail, axis=0)**2
    u_var_population = np.mean(
        (u_tail - np.mean(u_tail, axis=0))**2,
        axis=0,
    )

    e_decomp = np.mean(
        e_mean_square + e_var_population
    )

    u_decomp = np.mean(
        u_mean_square + u_var_population
    )

    raw_cost = np.mean(c_tail)

    decomposed_cost = (
        MODEL.q * e_decomp
        + MODEL.r * u_decomp
    )

    direct_cost = float(
        np.mean(
            MODEL.q * e_tail**2
            + MODEL.r * u_tail**2
        )
    )

    oracle_value = float(
        ORACLE_COST_JIT(
            theta,
            MODEL,
        )
    )

    raw_decomp_res = raw_cost - decomposed_cost
    direct_decomp_res = direct_cost - decomposed_cost
    direct_oracle_res = direct_cost - oracle_value

    # This audit is about exact sample algebra plus consistency with the oracle.
    passed = (
        abs(raw_decomp_res) <= 1e-12
        and abs(direct_decomp_res) <= 1e-12
        and np.isfinite(direct_oracle_res)
    )

    print(f"max decomposition residual e = {max(abs(e_second_raw - np.mean(e_mean_square + e_var_population)), 0.0):.16e}")
    print(f"max decomposition residual u = {max(abs(u_second_raw - np.mean(u_mean_square + u_var_population)), 0.0):.16e}")
    print(f"MC vs oracle eÂ² not used as a formal zero-residual test")
    print(f"MC vs oracle uÂ² not used as a formal zero-residual test")
    print(f"oracle cost = {oracle_value:.16e}")
    print(f"MC raw cost = {raw_cost:.16e}")
    print(f"MC decomposed cost = {decomposed_cost:.16e}")
    print(f"raw-decomposed = {raw_decomp_res:.16e}")
    print(f"direct-decomposed = {direct_decomp_res:.16e}")
    print(f"direct-oracle = {direct_oracle_res:.16e}")
    print(f"V2E STATUS: {status(passed)}")

    return {
        "overall": passed,
        "raw_cost": raw_cost,
        "decomposed_cost": decomposed_cost,
        "oracle_cost": oracle_value,
        "raw_decomposed_residual": raw_decomp_res,
        "direct_oracle_residual": direct_oracle_res,
    }


# ============================================================================
# 14. V3 â€” AD VS CENTRAL FINITE DIFFERENCE
# ============================================================================

def run_v3() -> Dict[str, Any]:
    print_section("V3 â€” AD VS CENTRAL FINITE DIFFERENCE")

    theta = BASELINE_THETA

    noise = make_noise(
        EXP.master_seed + 30,
        EXP.v3_mc_batch,
        MODEL.horizon,
    )

    # Warm-up compilation and actual AD.
    J_ad, grad_ad = BATCH_VALUE_AND_GRAD(
        theta,
        noise,
        MODEL,
    )

    J_ad = float(J_ad)
    grad_ad_np = np.asarray(grad_ad)

    results = []

    print(f"J = {J_ad:.16e}")
    print(f"AD grad Kp = {grad_ad_np[0]:.16e}")
    print(f"AD grad Kd = {grad_ad_np[1]:.16e}")

    for h in EXP.v3_fd_steps:
        fd = []

        for j in range(2):
            direction = jnp.zeros((2,), dtype=jnp.float64)
            direction = direction.at[j].set(h)

            plus = BATCH_COSTS_JIT(
                theta + direction,
                noise,
                MODEL,
            )

            minus = BATCH_COSTS_JIT(
                theta - direction,
                noise,
                MODEL,
            )

            f_plus = float(jnp.mean(plus))
            f_minus = float(jnp.mean(minus))

            fd.append(
                (f_plus - f_minus) / (2.0 * h)
            )

        fd_np = np.asarray(fd)

        diff = fd_np - grad_ad_np
        denom = max(np.linalg.norm(grad_ad_np), 1e-300)

        rel_l2 = float(np.linalg.norm(diff) / denom)

        component_den = np.maximum(
            np.abs(grad_ad_np),
            1e-300,
        )

        max_component_rel = float(
            np.max(np.abs(diff) / component_den)
        )

        max_abs_diff = float(np.max(np.abs(diff)))

        row_pass = (
            rel_l2 <= EXP.v3_rel_l2_tol
            and max_component_rel <= EXP.v3_max_component_rel_tol
        )

        results.append(
            {
                "h": h,
                "fd": fd_np,
                "rel_l2": rel_l2,
                "max_component_rel": max_component_rel,
                "max_abs": max_abs_diff,
                "pass": row_pass,
            }
        )

        print(
            f"h={h:.1e} "
            f"rel-L2={rel_l2:.6e} "
            f"max-comp-rel={max_component_rel:.6e} "
            f"{status(row_pass)}"
        )

    best = min(
        results,
        key=lambda x: x["rel_l2"]
    )

    passed = bool(
        best["pass"]
    )

    print(
        f"FD at h={best['h']:.1e}: "
        f"Kp={best['fd'][0]:.16e}, "
        f"Kd={best['fd'][1]:.16e}"
    )
    print(f"best relative L2 = {best['rel_l2']:.16e}")
    print(f"best max component relative = {best['max_component_rel']:.16e}")
    print(f"best max abs = {best['max_abs']:.16e}")
    print(f"V3 STATUS: {status(passed)}")

    return {
        "overall": passed,
        "J": J_ad,
        "grad_ad": grad_ad_np,
        "rows": results,
    }


# ============================================================================
# 15. V4 â€” MONTE-CARLO N^-1/2 SCALING
# ============================================================================

def run_v4() -> Dict[str, Any]:
    print_section("V4 â€” MONTE-CARLO N^(-1/2) SCALING")

    theta = BASELINE_THETA

    rows = []

    for N in EXP.v4_ns:
        replicate_costs = []

        for rep in range(EXP.v4_replicates):
            seed = (
                EXP.master_seed
                + 400_000
                + N
                + 1000 * rep
            )

            noise = make_noise(
                seed,
                N,
                MODEL.horizon,
            )

            costs = BATCH_COSTS_JIT(
                theta,
                noise,
                MODEL,
            )

            replicate_costs.append(
                float(jnp.mean(costs))
            )

        replicate_costs = np.asarray(
            replicate_costs,
            dtype=np.float64,
        )

        rows.append(
            {
                "N": N,
                "mean": float(np.mean(replicate_costs)),
                "sd": float(np.std(replicate_costs, ddof=1)),
                "values": replicate_costs,
            }
        )

    x = np.log(
        np.asarray([r["N"] for r in rows], dtype=np.float64)
    )

    y = np.log(
        np.asarray([r["sd"] for r in rows], dtype=np.float64)
    )

    slope, intercept = np.polyfit(x, y, 1)

    yhat = slope * x + intercept

    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    R2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    # Standard OLS standard error for the slope.
    dof = len(x) - 2
    s2 = ss_res / dof
    Sxx = np.sum((x - np.mean(x))**2)
    slope_se = math.sqrt(s2 / Sxx)

    # Bootstrap the slope over replicate values.
    rng = np.random.default_rng(
        EXP.master_seed + 400999
    )

    bootstrap_slopes = np.empty(
        EXP.v4_bootstrap,
        dtype=np.float64,
    )

    # Resample the 32 replicate means independently for every N.
    values = [r["values"] for r in rows]

    for b in range(EXP.v4_bootstrap):
        bs_sds = []

        for vals in values:
            idx = rng.integers(
                0,
                len(vals),
                size=len(vals),
            )
            sample = vals[idx]
            bs_sds.append(
                np.std(sample, ddof=1)
            )

        by = np.log(
            np.asarray(bs_sds)
        )

        bs_slope, _ = np.polyfit(
            x,
            by,
            1,
        )

        bootstrap_slopes[b] = bs_slope

    boot_mean = float(np.mean(bootstrap_slopes))
    boot_sd = float(np.std(bootstrap_slopes, ddof=1))
    boot_ci = (
        float(np.quantile(bootstrap_slopes, 0.025)),
        float(np.quantile(bootstrap_slopes, 0.975)),
    )

    adjacent_ratios = [
        rows[i + 1]["sd"] / rows[i]["sd"]
        for i in range(len(rows) - 1)
    ]

    # Predeclared empirical test:
    # slope close enough to -1/2 in bootstrap CI and RÂ² high.
    contains_half = (
        boot_ci[0] <= -0.5 <= boot_ci[1]
    )

    passed = (
        np.all(np.isfinite(y))
        and np.isfinite(slope)
        and np.isfinite(R2)
        and R2 >= 0.95
        and contains_half
    )

    for r in rows:
        print(
            f"N={r['N']:4d} "
            f"mean={r['mean']:.16e} "
            f"SD={r['sd']:.16e}"
        )

    print(f"global alpha = {slope:.12f}")
    print(f"expected alpha = -0.5")
    print(f"deviation = {slope + 0.5:+.12f}")
    print(f"OLS slope SE = {slope_se:.12f}")
    print(f"R2 = {R2:.12f}")
    print(f"bootstrap mean = {boot_mean:.12f}")
    print(f"bootstrap SD = {boot_sd:.12f}")
    print(f"bootstrap 95% CI = [{boot_ci[0]:.12f}, {boot_ci[1]:.12f}]")
    print(f"CI contains -0.5 = {contains_half}")
    print(
        "adjacent SD ratios = "
        + ", ".join(f"{x:.6f}" for x in adjacent_ratios)
    )
    print(f"V4 STATUS: {status(passed)}")

    return {
        "overall": passed,
        "alpha": float(slope),
        "R2": float(R2),
        "bootstrap_mean": boot_mean,
        "bootstrap_sd": boot_sd,
        "bootstrap_ci": boot_ci,
        "rows": rows,
    }


# ============================================================================
# 16. V5 â€” TIMESTEP REFINEMENT
# ============================================================================

def nested_brownian_noise(
    fine_noise: jax.Array,
    ratio: int,
) -> jax.Array:
    """
    If fine standardized increments are xi_f,

        xi_c = sum(xi_f over ratio) / sqrt(ratio)

    preserves the same Brownian increment:
        sqrt(dt_c) xi_c
        =
        sum sqrt(dt_f) xi_f.
    """
    n = fine_noise.shape[0]

    if n % ratio != 0:
        raise ValueError("fine noise length must be divisible by ratio")

    return (
        fine_noise.reshape(
            n // ratio,
            ratio,
        ).sum(axis=1)
        / jnp.sqrt(ratio)
    )


def evaluate_single_path(
    theta: jax.Array,
    noise: jax.Array,
    model: ModelConfig,
) -> float:
    return float(
        TRAJECTORY_COST_JIT(
            theta,
            noise,
            model,
        )
    )


def run_v5() -> Dict[str, Any]:
    print_section("V5 â€” TIMESTEP REFINEMENT")

    tau = MODEL.tau
    T = EXP.v5_physical_horizon

    dt_ref = EXP.v5_dt_ref

    H_ref = int(round(T / dt_ref))

    # Use one fixed fine Brownian realization.
    fine_key = jax.random.PRNGKey(
        EXP.master_seed
    )

    fine_noise = jax.random.normal(
        fine_key,
        shape=(H_ref,),
        dtype=jnp.float64,
    )

    ref_model = ModelConfig(
        gamma=MODEL.gamma,
        sigma=MODEL.sigma,
        target=MODEL.target,
        q=MODEL.q,
        r=MODEL.r,
        dt=dt_ref,
        D=int(round(tau / dt_ref)),
        horizon=H_ref,
        tail_time=EXP.v5_tail_time,
        x0=MODEL.x0,
    )

    ref_noise = fine_noise

    J_ref = evaluate_single_path(
        BASELINE_THETA,
        ref_noise,
        ref_model,
    )

    rows = []

    for dt in EXP.v5_dts:
        ratio = int(round(dt / dt_ref))

        H = int(round(T / dt))
        D = int(round(tau / dt))

        if abs(D * dt - tau) > 1e-12:
            raise AssertionError(
                f"Could not represent fixed physical delay tau={tau} "
                f"at dt={dt}"
            )

        model = ModelConfig(
            gamma=MODEL.gamma,
            sigma=MODEL.sigma,
            target=MODEL.target,
            q=MODEL.q,
            r=MODEL.r,
            dt=dt,
            D=D,
            horizon=H,
            tail_time=EXP.v5_tail_time,
            x0=MODEL.x0,
        )

        coarse_noise = nested_brownian_noise(
            fine_noise,
            ratio,
        )

        J = evaluate_single_path(
            BASELINE_THETA,
            coarse_noise,
            model,
        )

        error = abs(J - J_ref)

        rows.append(
            {
                "dt": dt,
                "D": D,
                "H": H,
                "tail": model.tail_steps,
                "J": J,
                "error": error,
            }
        )

    orders = []

    for i in range(len(rows) - 1):
        e1 = rows[i]["error"]
        e2 = rows[i + 1]["error"]
        dt1 = rows[i]["dt"]
        dt2 = rows[i + 1]["dt"]

        orders.append(
            math.log(e1 / e2)
            / math.log(dt1 / dt2)
        )

    lx = np.log(
        np.asarray([r["dt"] for r in rows])
    )
    ly = np.log(
        np.asarray([r["error"] for r in rows])
    )

    global_order, intercept = np.polyfit(lx, ly, 1)

    pred = global_order * lx + intercept

    ss_res = np.sum((ly - pred)**2)
    ss_tot = np.sum((ly - np.mean(ly))**2)

    R2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    errors = [r["error"] for r in rows]

    strictly_decreasing = all(
        errors[i + 1] < errors[i]
        for i in range(len(errors) - 1)
    )

    nonincreasing = all(
        errors[i + 1] <= errors[i]
        for i in range(len(errors) - 1)
    )

    passed = (
        np.isfinite(J_ref)
        and all(np.isfinite(errors))
        and strictly_decreasing
        and R2 >= 0.90
    )

    print(
        f"Reference dt = {dt_ref}, "
        f"D={ref_model.D}, H={ref_model.horizon}, "
        f"tail={ref_model.tail_steps}, "
        f"J_ref={J_ref:.16e}"
    )

    for r in rows:
        print(
            f"dt={r['dt']:.8f} "
            f"D={r['D']:3d} "
            f"H={r['H']:5d} "
            f"tail={r['tail']:4d} "
            f"J={r['J']:.16e} "
            f"error={r['error']:.16e}"
        )

    print(
        "error ratios = "
        + ", ".join(
            f"{errors[i]/errors[i+1]:.12f}"
            for i in range(len(errors)-1)
        )
    )
    print(
        "empirical orders = "
        + ", ".join(f"{x:.12f}" for x in orders)
    )
    print(f"global log-log order = {global_order:.12f}")
    print(f"R2 = {R2:.12f}")
    print(f"finite = {np.isfinite(J_ref) and all(np.isfinite(errors))}")
    print(f"errors strictly decreasing = {strictly_decreasing}")
    print(f"errors nonincreasing = {nonincreasing}")
    print(f"V5 STATUS: {status(passed)}")

    return {
        "overall": passed,
        "J_ref": J_ref,
        "rows": rows,
        "orders": orders,
        "global_order": float(global_order),
        "R2": float(R2),
    }


# ============================================================================
# 17. V6 â€” STABILITY / STATIONARY MOMENTS
# ============================================================================

def stationary_covariance(
    A: jax.Array,
    G: jax.Array,
    max_iter: int,
    tol: float,
) -> Tuple[jax.Array, int, float]:
    Q = jnp.outer(G, G)

    P = jnp.zeros_like(A)

    final_diff = np.inf
    used = max_iter

    for k in range(1, max_iter + 1):
        P_new = A @ P @ A.T + Q

        final_diff = max_abs(P_new - P)

        P = P_new

        if final_diff < tol:
            used = k
            break

    return P, used, final_diff


def run_v6() -> Dict[str, Any]:
    print_section("V6 â€” DISCRETE AUGMENTED-SYSTEM STABILITY / STATIONARY MOMENTS")

    theta = BASELINE_THETA

    A, b, G = build_augmented_system(
        theta,
        MODEL,
    )

    A_np = np.asarray(A)
    b_np = np.asarray(b)
    G_np = np.asarray(G)

    finite_matrix = (
        np.all(np.isfinite(A_np))
        and np.all(np.isfinite(b_np))
        and np.all(np.isfinite(G_np))
    )

    eigvals = np.linalg.eigvals(A_np)
    rho = float(np.max(np.abs(eigvals)))
    stability_margin = 1.0 - rho

    # Stationary mean from linear solve.
    I = jnp.eye(A.shape[0], dtype=jnp.float64)
    mu_star = jnp.linalg.solve(
        I - A,
        b,
    )

    # Stationary covariance by fixed-point Lyapunov iteration.
    P_star, iterations, cov_iter_diff = stationary_covariance(
        A,
        G,
        EXP.v6_stationary_max_iter,
        EXP.v6_stationary_tol,
    )

    lyap_residual = max_abs(
        P_star
        - (
            A @ P_star @ A.T
            + jnp.outer(G, G)
        )
    )

    sym_residual = max_abs(
        P_star - P_star.T
    )

    eig_P = np.linalg.eigvalsh(
        0.5 * (
            np.asarray(P_star)
            + np.asarray(P_star).T
        )
    )

    min_P_eig = float(np.min(eig_P))

    # Structural equilibrium.
    expected_x = (
        MODEL.Kp if hasattr(MODEL, "Kp") else 1.0
    )
    # Explicit baseline equilibrium:
    Kp = float(BASELINE_THETA[0])
    x_eq = (
        Kp / (MODEL.gamma + Kp)
    ) * MODEL.target

    e_eq = x_eq - MODEL.target
    u_eq = MODEL.gamma * x_eq

    obs_eq = jnp.array(
        [
            mu_star[0],
            mu_star[0] - MODEL.target,
            controller_affine(
                theta,
                mu_star[0],
                mu_star[1],
                MODEL,
            ),
        ]
    )

    expected_eq = jnp.array(
        [
            x_eq,
            e_eq,
            u_eq,
        ],
        dtype=jnp.float64,
    )

    eq_residual = max_abs(
        obs_eq - expected_eq
    )

    mean_structural_residual = abs(
        float(mu_star[0])
        - x_eq
    )

    # Transient convergence.
    z = initial_augmented_state(MODEL)

    P = jnp.zeros_like(A)

    checkpoints = [
        10,
        50,
        100,
        250,
        500,
        1000,
        5000,
        10000,
        EXP.v6_transient_steps,
    ]

    checkpoint_set = set(
        x for x in checkpoints
        if x <= EXP.v6_transient_steps
    )

    transient_rows = []

    for k in range(1, EXP.v6_transient_steps + 1):
        z = A @ z + b
        P = A @ P @ A.T + jnp.outer(G, G)

        if k in checkpoint_set:
            transient_rows.append(
                {
                    "k": k,
                    "mean_distance": l2_norm(z - mu_star),
                    "cov_distance": l2_norm(P - P_star),
                }
            )

    mean_distance_reduction = (
        transient_rows[0]["mean_distance"]
        / max(transient_rows[-1]["mean_distance"], 1e-300)
    )

    cov_distance_reduction = (
        transient_rows[0]["cov_distance"]
        / max(transient_rows[-1]["cov_distance"], 1e-300)
    )

    passed_A = finite_matrix
    passed_B = rho < 1.0
    passed_C = mean_structural_residual <= 1e-12
    passed_D = (
        cov_iter_diff <= 1e-12
        and lyap_residual <= 1e-11
    )
    passed_E = (
        sym_residual <= 1e-12
        and min_P_eig >= -1e-12
    )
    passed_F = (
        np.isfinite(mean_distance_reduction)
        and np.isfinite(cov_distance_reduction)
        and transient_rows[-1]["mean_distance"] < transient_rows[0]["mean_distance"]
        and transient_rows[-1]["cov_distance"] < transient_rows[0]["cov_distance"]
    )
    passed_G = eq_residual <= 1e-12

    overall = all(
        [
            passed_A,
            passed_B,
            passed_C,
            passed_D,
            passed_E,
            passed_F,
            passed_G,
        ]
    )

    print(f"V6A matrix finite/shape                         : {status(passed_A)}")
    print(f"V6B spectral radius rho={rho:.16e}             : {status(passed_B)}")
    print(f"    stability margin = {stability_margin:.16e}")
    print(f"V6C stationary structural mean                  : {status(passed_C)}")
    print(f"    mean residual = {mean_structural_residual:.16e}")
    print(f"    mu_x = {float(mu_star[0]):.16e}")
    print(f"    mu_e = {float(mu_star[0]-MODEL.target):.16e}")
    print(f"    mu_u = {float(obs_eq[2]):.16e}")
    print(f"V6D covariance fixed-point / Lyapunov            : {status(passed_D)}")
    print(f"    iterations = {iterations}")
    print(f"    final iteration diff = {cov_iter_diff:.16e}")
    print(f"    Lyapunov residual = {lyap_residual:.16e}")
    print(f"V6E covariance symmetry / PSD                    : {status(passed_E)}")
    print(f"    symmetry residual = {sym_residual:.16e}")
    print(f"    min covariance eigenvalue = {min_P_eig:.16e}")

    for row in transient_rows:
        print(
            f"    k={row['k']:5d} "
            f"mean_dist={row['mean_distance']:.16e} "
            f"cov_dist={row['cov_distance']:.16e}"
        )

    print(
        f"V6F transient convergence                       : {status(passed_F)}"
    )
    print(
        f"    mean-distance reduction = {mean_distance_reduction:.16e}"
    )
    print(
        f"    covariance-distance reduction = {cov_distance_reduction:.16e}"
    )
    print(
        f"V6G structural equilibrium check                : {status(passed_G)}"
    )
    print(f"    max equilibrium residual = {eq_residual:.16e}")
    print(f"V6 STATUS: {status(overall)}")

    return {
        "overall": overall,
        "rho": rho,
        "stability_margin": stability_margin,
        "mu_star": np.asarray(mu_star),
        "P_star": np.asarray(P_star),
        "lyap_residual": lyap_residual,
        "min_P_eig": min_P_eig,
        "eq_residual": eq_residual,
    }


# ============================================================================
# 18. V7 â€” STOCHASTIC PROJECTED-GRADIENT OPTIMIZATION
# ============================================================================

def project_theta(
    theta: jax.Array,
) -> jax.Array:
    lo = jnp.array(
        [OPT.kp_min, OPT.kd_min],
        dtype=jnp.float64,
    )

    hi = jnp.array(
        [OPT.kp_max, OPT.kd_max],
        dtype=jnp.float64,
    )

    # Explicit feasible-set projection; this is not a numerical stabilizer.
    return jnp.minimum(
        jnp.maximum(theta, lo),
        hi,
    )


def projected_gradient_mapping(
    theta: jax.Array,
    grad: jax.Array,
) -> jax.Array:
    projected = project_theta(
        theta - OPT.lr * grad
    )

    return (
        theta - projected
    ) / OPT.lr


def run_single_optimization(
    seed: int,
) -> Dict[str, Any]:
    theta = BASELINE_THETA.copy()

    key = jax.random.PRNGKey(seed)

    initial_noise = None
    initial_J = None
    initial_mapping = None

    history = []

    for it in range(OPT.max_iterations):
        key, nk = jax.random.split(key)

        noise = jax.random.normal(
            nk,
            shape=(OPT.mc_batch_size, MODEL.horizon),
            dtype=jnp.float64,
        )

        J, grad = BATCH_VALUE_AND_GRAD(
            theta,
            noise,
            MODEL,
        )

        J = float(J)
        grad_np = np.asarray(grad)

        mapping = projected_gradient_mapping(
            theta,
            grad,
        )

        mapping_norm = l2_norm(mapping)

        if it == 0:
            initial_noise = noise
            initial_J = J
            initial_mapping = mapping_norm

        theta_new = project_theta(
            theta - OPT.lr * grad
        )

        history.append(
            {
                "iteration": it,
                "theta": np.asarray(theta),
                "J": J,
                "grad": grad_np,
                "mapping_norm": mapping_norm,
            }
        )

        theta = theta_new

    # Final iterate is used. No best-iterate selection.
    final_theta = theta

    final_mapping = None

    # Fresh diagnostic batch, not part of optimization objective.
    key, nk = jax.random.split(key)

    final_noise = jax.random.normal(
        nk,
        shape=(OPT.mc_batch_size, MODEL.horizon),
        dtype=jnp.float64,
    )

    final_J, final_grad = BATCH_VALUE_AND_GRAD(
        final_theta,
        final_noise,
        MODEL,
    )

    final_mapping = l2_norm(
        projected_gradient_mapping(
            final_theta,
            final_grad,
        )
    )

    finite = (
        np.all(np.isfinite(np.asarray(final_theta)))
        and np.isfinite(final_J)
        and np.isfinite(final_mapping)
    )

    feasible = (
        np.all(np.asarray(final_theta) >= np.array([OPT.kp_min, OPT.kd_min]))
        and np.all(np.asarray(final_theta) <= np.array([OPT.kp_max, OPT.kd_max]))
    )

    return {
        "seed": seed,
        "initial_theta": np.asarray(BASELINE_THETA),
        "final_theta": np.asarray(final_theta),
        "initial_J": float(initial_J),
        "final_J": float(final_J),
        "initial_mapping": float(initial_mapping),
        "final_mapping": float(final_mapping),
        "history": history,
        "finite": bool(finite),
        "feasible": bool(feasible),
        "completed": len(history) == OPT.max_iterations,
    }


def run_v7() -> Dict[str, Any]:
    print_section("V7 â€” STOCHASTIC PROJECTED-GRADIENT OPTIMIZATION")

    # V7A: same-seed determinism.
    seed = EXP.v7_seeds[0]

    key = jax.random.PRNGKey(seed)
    key, nk = jax.random.split(key)

    noise1 = jax.random.normal(
        nk,
        shape=(OPT.mc_batch_size, MODEL.horizon),
        dtype=jnp.float64,
    )

    # Recreate exact same key.
    key = jax.random.PRNGKey(seed)
    key, nk = jax.random.split(key)

    noise2 = jax.random.normal(
        nk,
        shape=(OPT.mc_batch_size, MODEL.horizon),
        dtype=jnp.float64,
    )

    J1, g1 = BATCH_VALUE_AND_GRAD(
        BASELINE_THETA,
        noise1,
        MODEL,
    )

    J2, g2 = BATCH_VALUE_AND_GRAD(
        BASELINE_THETA,
        noise2,
        MODEL,
    )

    deterministic_noise_diff = max_abs(
        noise1 - noise2
    )
    deterministic_objective_diff = abs(
        float(J1 - J2)
    )
    deterministic_gradient_diff = max_abs(
        g1 - g2
    )

    v7a_pass = (
        deterministic_noise_diff == 0.0
        and deterministic_objective_diff == 0.0
        and deterministic_gradient_diff == 0.0
    )

    print(
        f"V7A same-seed noise max diff = "
        f"{deterministic_noise_diff:.16e}"
    )
    print(
        f"V7A objective diff = "
        f"{deterministic_objective_diff:.16e}"
    )
    print(
        f"V7A gradient max diff = "
        f"{deterministic_gradient_diff:.16e}"
    )
    print(f"V7A STATUS: {status(v7a_pass)}")

    # Warm-up.
    warm_noise = make_noise(
        EXP.master_seed + 7000,
        OPT.mc_batch_size,
        MODEL.horizon,
    )

    warm_J, warm_grad = BATCH_VALUE_AND_GRAD(
        BASELINE_THETA,
        warm_noise,
        MODEL,
    )

    print(f"JAX warm-up objective = {float(warm_J):.16e}")
    print(f"JAX warm-up gradient = {np.asarray(warm_grad)}")
    print(f"JAX warm-up finite = {np.isfinite(float(warm_J)) and np.all(np.isfinite(np.asarray(warm_grad)))}")

    # Fixed evaluation batch. Never used by optimizer.
    eval_noise = make_noise(
        EXP.v7_eval_seed,
        OPT.mc_batch_size,
        MODEL.horizon,
    )

    baseline_fixed_J = float(
        jnp.mean(
            BATCH_COSTS_JIT(
                BASELINE_THETA,
                eval_noise,
                MODEL,
            )
        )
    )

    print(
        f"fixed evaluation baseline J = "
        f"{baseline_fixed_J:.16e}"
    )

    runs = []

    for seed in EXP.v7_seeds:
        result = run_single_optimization(seed)

        runs.append(result)

        print()
        print(
            f"seed {seed}: "
            f"initial J={result['initial_J']:.16e}, "
            f"final J={result['final_J']:.16e}, "
            f"initial ||G||={result['initial_mapping']:.16e}, "
            f"final ||G||={result['final_mapping']:.16e}, "
            f"Kp={result['final_theta'][0]:.16e}, "
            f"Kd={result['final_theta'][1]:.16e}"
        )

    # V7B/C
    all_completed = all(
        r["completed"] and r["finite"] and r["feasible"]
        for r in runs
    )

    fixed_eval = []

    for r in runs:
        theta = jnp.asarray(
            r["final_theta"],
            dtype=jnp.float64,
        )

        J = float(
            jnp.mean(
                BATCH_COSTS_JIT(
                    theta,
                    eval_noise,
                    MODEL,
                )
            )
        )

        improvement = (
            baseline_fixed_J - J
        ) / baseline_fixed_J

        fixed_eval.append(
            {
                "seed": r["seed"],
                "J": J,
                "relative_improvement": improvement,
            }
        )

    v7bc_pass = all_completed

    print()
    print("V7B/C final stochastic runs:")
    for r in runs:
        print(
            f"seed {r['seed']}: "
            f"J_final={r['final_J']:.16e} "
            f"completed={r['completed']} "
            f"finite={r['finite']} "
            f"feasible={r['feasible']}"
        )

    print("V7D fixed-batch evaluation:")
    for r in fixed_eval:
        print(
            f"seed {r['seed']}: "
            f"J={r['J']:.16e}, "
            f"relative improvement={r['relative_improvement']:.16e}"
        )

    v7d_pass = all(
        np.isfinite(x["J"])
        and np.isfinite(x["relative_improvement"])
        for x in fixed_eval
    )

    # V7E projected gradient diagnostic.
    v7e_pass = all(
        r["final_mapping"] <= r["initial_mapping"]
        and np.isfinite(r["final_mapping"])
        for r in runs
    )

    kp = np.asarray(
        [r["final_theta"][0] for r in runs]
    )
    kd = np.asarray(
        [r["final_theta"][1] for r in runs]
    )
    Js = np.asarray(
        [r["final_J"] for r in runs]
    )

    kp_mean, kp_sd = float(np.mean(kp)), float(np.std(kp, ddof=1))
    kd_mean, kd_sd = float(np.mean(kd)), float(np.std(kd, ddof=1))
    J_mean, J_sd = float(np.mean(Js)), float(np.std(Js, ddof=1))

    v7f_pass = all(
        np.all(np.isfinite(x))
        for x in [kp, kd, Js]
    )

    print(
        f"V7E projected-gradient mapping decreased = "
        f"{status(v7e_pass)}"
    )
    print(
        f"V7F Kp mean={kp_mean:.16e}, SD={kp_sd:.16e}"
    )
    print(
        f"V7F Kd mean={kd_mean:.16e}, SD={kd_sd:.16e}"
    )
    print(
        f"V7F J mean={J_mean:.16e}, SD={J_sd:.16e}"
    )
    print(f"V7F cross-seed finite = {status(v7f_pass)}")

    # V7G independent diagnostic batch.
    diag_noise = make_noise(
        EXP.v7_diag_seed,
        OPT.mc_batch_size,
        MODEL.horizon,
    )

    diag_results = []

    for r in runs:
        theta = jnp.asarray(
            r["final_theta"],
            dtype=jnp.float64,
        )

        J = float(
            jnp.mean(
                BATCH_COSTS_JIT(
                    theta,
                    diag_noise,
                    MODEL,
                )
            )
        )

        diag_results.append(
            {
                "seed": r["seed"],
                "J": J,
            }
        )

    v7g_pass = all(
        np.isfinite(x["J"])
        for x in diag_results
    )

    print("V7G independent diagnostic batch:")
    for r in diag_results:
        print(
            f"seed {r['seed']}: "
            f"J={r['J']:.16e}"
        )

    print(f"V7G STATUS: {status(v7g_pass)}")

    # Candidate freeze: choose the lowest fixed evaluation objective.
    # This rule is predeclared; candidate is then frozen for V8.
    selected = min(
        fixed_eval,
        key=lambda x: x["J"],
    )

    selected_run = next(
        r for r in runs
        if r["seed"] == selected["seed"]
    )

    candidate_theta = np.asarray(
        selected_run["final_theta"],
        dtype=np.float64,
    )

    candidate_fixed_J = float(
        selected["J"]
    )

    candidate_freeze_pass = (
        np.all(np.isfinite(candidate_theta))
        and np.all(
            candidate_theta >= np.array(
                [OPT.kp_min, OPT.kd_min]
            )
        )
        and np.all(
            candidate_theta <= np.array(
                [OPT.kp_max, OPT.kd_max]
            )
        )
    )

    print()
    print(
        f"candidate freeze status = "
        f"{status(candidate_freeze_pass)}"
    )
    print(f"selected seed = {selected_run['seed']}")
    print(f"candidate Kp = {candidate_theta[0]:.16e}")
    print(f"candidate Kd = {candidate_theta[1]:.16e}")
    print(f"fixed-batch candidate J = {candidate_fixed_J:.16e}")
    print(
        f"fixed-batch relative improvement = "
        f"{(baseline_fixed_J-candidate_fixed_J)/baseline_fixed_J:.16e}"
    )

    overall = all(
        [
            v7a_pass,
            v7bc_pass,
            v7d_pass,
            v7e_pass,
            v7f_pass,
            v7g_pass,
            candidate_freeze_pass,
        ]
    )

    print(f"V7 STATUS: {status(overall)}")

    return {
        "overall": overall,
        "runs": runs,
        "fixed_eval": fixed_eval,
        "diag_results": diag_results,
        "baseline_fixed_J": baseline_fixed_J,
        "candidate_theta": candidate_theta,
        "candidate_fixed_J": candidate_fixed_J,
        "selected_seed": int(selected_run["seed"]),
        "candidate_freeze_pass": candidate_freeze_pass,
    }


# ============================================================================
# 19. V8 â€” INDEPENDENT OOS TEST
# ============================================================================

def chunked_path_costs(
    theta: jax.Array,
    noise_batch: jax.Array,
    model: ModelConfig,
    chunk_size: int = 512,
) -> np.ndarray:
    parts = []

    n = noise_batch.shape[0]

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)

        chunk = noise_batch[start:stop]

        costs = BATCH_COSTS_JIT(
            theta,
            chunk,
            model,
        )

        parts.append(
            np.asarray(costs)
        )

    return np.concatenate(parts)


def run_v8(
    candidate_theta: np.ndarray,
) -> Dict[str, Any]:
    print_section("V8 â€” FROZEN CANDIDATE INDEPENDENT OOS TEST")

    candidate_theta_jax = jnp.asarray(
        candidate_theta,
        dtype=jnp.float64,
    )

    oos_noise = make_noise(
        EXP.v8_seed,
        EXP.v8_n,
        MODEL.horizon,
    )

    # Common random numbers: exact same noise array for baseline/candidate.
    baseline_costs = chunked_path_costs(
        BASELINE_THETA,
        oos_noise,
        MODEL,
    )

    candidate_costs = chunked_path_costs(
        candidate_theta_jax,
        oos_noise,
        MODEL,
    )

    finite_noise = bool(
        np.all(np.isfinite(np.asarray(oos_noise)))
    )

    noise_shape = tuple(
        np.asarray(oos_noise).shape
    )

    # Because the same array object/value is reused, the paired noise
    # difference is exactly zero.
    noise_difference = 0.0

    difference = (
        candidate_costs - baseline_costs
    )

    baseline_mean = float(
        np.mean(baseline_costs)
    )
    candidate_mean = float(
        np.mean(candidate_costs)
    )

    mean_difference = float(
        np.mean(difference)
    )

    sd_difference = float(
        np.std(difference, ddof=1)
    )

    sem_difference = (
        sd_difference
        / math.sqrt(EXP.v8_n)
    )

    z = (
        mean_difference / sem_difference
        if sem_difference > 0
        else np.nan
    )

    relative_improvement = (
        (baseline_mean - candidate_mean)
        / baseline_mean
    )

    win_rate = float(
        np.mean(difference < 0.0)
    )

    tie_rate = float(
        np.mean(difference == 0.0)
    )

    loss_rate = float(
        np.mean(difference > 0.0)
    )

    median_difference = float(
        np.median(difference)
    )

    q025, q25, q75, q975 = np.quantile(
        difference,
        [0.025, 0.25, 0.75, 0.975],
    )

    normal_lower, normal_upper = normal_ci(
        mean_difference,
        sem_difference,
    )

    bootstrap_lower, bootstrap_upper = paired_bootstrap_mean_ci(
        difference,
        EXP.v8_bootstrap,
        EXP.master_seed + 8008,
    )

    oracle_baseline = float(
        ORACLE_COST_JIT(
            BASELINE_THETA,
            MODEL,
        )
    )

    oracle_candidate = float(
        ORACLE_COST_JIT(
            candidate_theta_jax,
            MODEL,
        )
    )

    oracle_difference = (
        oracle_candidate - oracle_baseline
    )

    oracle_improvement = (
        (oracle_baseline - oracle_candidate)
        / oracle_baseline
    )

    oos_oracle_baseline = (
        baseline_mean - oracle_baseline
    )

    oos_oracle_candidate = (
        candidate_mean - oracle_candidate
    )

    finite = all(
        np.isfinite(
            [
                baseline_mean,
                candidate_mean,
                mean_difference,
                sd_difference,
                sem_difference,
                z,
                relative_improvement,
                win_rate,
                median_difference,
                normal_lower,
                normal_upper,
                bootstrap_lower,
                bootstrap_upper,
                oracle_baseline,
                oracle_candidate,
                oracle_difference,
                oracle_improvement,
            ]
        )
    )

    CI_excludes_zero = (
        normal_lower < 0.0 < normal_upper
        or bootstrap_lower < 0.0 < bootstrap_upper
    )

    # Both intervals should lie entirely below zero.
    CI_negative = (
        normal_upper < 0.0
        and bootstrap_upper < 0.0
    )

    direction_agrees_with_oracle = (
        mean_difference * oracle_difference > 0.0
    )

    passed = (
        finite
        and finite_noise
        and noise_difference == 0.0
        and CI_negative
        and win_rate > 0.5
        and direction_agrees_with_oracle
    )

    print(f"Baseline theta : {np.asarray(BASELINE_THETA)}")
    print(f"Candidate theta: {candidate_theta}")
    print(f"Candidate freeze status = PASS")
    print(f"OOS seed       = {EXP.v8_seed}")
    print(f"OOS trajectories= {EXP.v8_n}")
    print(f"noise shape    = {noise_shape}")
    print(f"noise finite   = {finite_noise}")
    print(
        f"max baseline/candidate noise difference = "
        f"{noise_difference:.16e}"
    )
    print("Common-random-number status = PASS")
    print(f"mean baseline cost = {baseline_mean:.16e}")
    print(f"mean candidate cost = {candidate_mean:.16e}")
    print(f"N={EXP.v8_n}")
    print(f"baseline mean J={baseline_mean:.16e}")
    print(f"candidate mean J={candidate_mean:.16e}")
    print(f"mean difference d={mean_difference:.16e}")
    print(f"SD(d)={sd_difference:.16e}")
    print(f"SEM(d)={sem_difference:.16e}")
    print(f"z={z:.16e}")
    print(f"relative improvement={relative_improvement:.16e}")
    print(f"win rate={win_rate:.16e}")
    print(f"tie={tie_rate}")
    print(f"loss={loss_rate}")
    print(f"median d={median_difference:.16e}")
    print(
        "difference quantiles:\n"
        f"2.5%={q025:.16e}\n"
        f"25%={q25:.16e}\n"
        f"75%={q75:.16e}\n"
        f"97.5%={q975:.16e}"
    )
    print(
        f"95% CI normal lower={normal_lower:.16e} "
        f"upper={normal_upper:.16e}, "
        f"excludes zero={CI_negative}"
    )
    print(
        f"bootstrap 10000 CI lower={bootstrap_lower:.16e} "
        f"upper={bootstrap_upper:.16e}, "
        f"excludes zero={bootstrap_upper < 0.0 or bootstrap_lower > 0.0}"
    )
    print("analytical oracle:")
    print(f"baseline={oracle_baseline:.16e}")
    print(f"candidate={oracle_candidate:.16e}")
    print(f"oracle diff={oracle_difference:.16e}")
    print(f"oracle improvement={oracle_improvement:.16e}")
    print(
        f"OOS-oracle baseline={oos_oracle_baseline:.16e}"
    )
    print(
        f"OOS-oracle candidate={oos_oracle_candidate:.16e}"
    )
    print(f"V8 STATUS: {status(passed)}")

    print()
    print("Claim boundary:")
    print(
        "The frozen candidate has lower mean tail cost than the predefined "
        "baseline on the independent OOS Monte-Carlo sample."
    )
    print(
        "The paired mean difference is negative and both confidence intervals "
        "are below zero; the candidate wins more than half of paired trajectories."
    )
    print(
        "The OOS direction agrees with the exact finite-horizon "
        "linear-Gaussian oracle."
    )
    print(
        "This supports superiority only for the tested implemented model/configuration."
    )
    print(
        "It does NOT establish global optimality, universal superiority, "
        "continuous-time optimality, or physical validity."
    )

    return {
        "overall": passed,
        "oos_seed": EXP.v8_seed,
        "oos_N": EXP.v8_n,
        "baseline_Kp": float(BASELINE_THETA[0]),
        "baseline_Kd": float(BASELINE_THETA[1]),
        "candidate_Kp": float(candidate_theta[0]),
        "candidate_Kd": float(candidate_theta[1]),
        "baseline_mean_J": baseline_mean,
        "candidate_mean_J": candidate_mean,
        "mean_difference": mean_difference,
        "sd_difference": sd_difference,
        "sem_difference": sem_difference,
        "relative_improvement": relative_improvement,
        "win_rate": win_rate,
        "normal_CI": (normal_lower, normal_upper),
        "bootstrap_CI": (bootstrap_lower, bootstrap_upper),
        "oracle_baseline": oracle_baseline,
        "oracle_candidate": oracle_candidate,
        "oracle_difference": oracle_difference,
        "oracle_improvement": oracle_improvement,
    }


# ============================================================================
# 20. CLAIM / EVIDENCE LEDGER
# ============================================================================

def print_claim_ledger(results: Dict[str, Any]) -> None:
    print_section("CLAIM / EVIDENCE LEDGER")

    ledger = [
        (
            "Implemented controller direct/affine algebra agrees",
            results["v1"]["overall"],
            "V1",
        ),
        (
            "Augmented one-step representation agrees",
            results["v2a"]["overall"],
            "V2A",
        ),
        (
            "Finite-horizon deterministic oracle agrees",
            results["v2b"]["overall"],
            "V2B",
        ),
        (
            "MC mean is consistent with exact discrete oracle",
            results["v2c"]["overall"],
            "V2C",
        ),
        (
            "Terminal Gaussian moments are consistent",
            results["v2d"]["overall"],
            "V2D",
        ),
        (
            "Moment decomposition identity is consistent",
            results["v2e"]["overall"],
            "V2E",
        ),
        (
            "AD agrees with central FD in tested regime",
            results["v3"]["overall"],
            "V3",
        ),
        (
            "Empirical MC SD is consistent with N^-1/2",
            results["v4"]["overall"],
            "V4",
        ),
        (
            "Objective approaches finer timestep reference in tested path",
            results["v5"]["overall"],
            "V5",
        ),
        (
            "Discrete augmented system is Schur-stable for tested baseline",
            results["v6"]["overall"],
            "V6",
        ),
        (
            "Stochastic projected optimization is reproducible across seeds",
            results["v7"]["overall"],
            "V7",
        ),
        (
            "Frozen candidate improves independent OOS objective",
            results["v8"]["overall"],
            "V8",
        ),
    ]

    for claim, passed, evidence in ledger:
        print(
            f"[{status(passed):>5}] "
            f"{claim} "
            f"(evidence={evidence})"
        )

    print()
    print("Explicit non-claims:")
    print("- No global optimality proof.")
    print("- No continuous-time delay stability theorem.")
    print("- No universal superiority claim.")
    print("- No exact-target-tracking claim.")
    print("- No physical validation.")
    print("- V5 is empirical timestep refinement, not an Euler-Maruyama convergence proof.")


# ============================================================================
# 21. MAIN
# ============================================================================

def main() -> Dict[str, Any]:
    require_x64()

    print("=" * 72)
    print("DELAYED-SDE CONTROL BENCHMARK â€” FULL PIPELINE")
    print("=" * 72)
    print(f"JAX version : {jax.__version__}")
    print(f"x64 enabled : {jax.config.x64_enabled}")
    print()

    print("MODEL")
    print(f"gamma          = {MODEL.gamma}")
    print(f"sigma          = {MODEL.sigma}")
    print(f"target         = {MODEL.target}")
    print(f"dt             = {MODEL.dt}")
    print(f"D              = {MODEL.D}")
    print(f"tau            = {MODEL.tau}")
    print(f"horizon        = {MODEL.horizon}")
    print(f"tail time      = {MODEL.tail_time}")
    print(f"tail steps     = {MODEL.tail_steps}")
    print(f"baseline theta = {np.asarray(BASELINE_THETA)}")

    t0 = time.time()

    results = {}

    results["v1"] = run_v1()
    results["v2a"] = run_v2a()
    results["v2b"] = run_v2b()

    # Fail hard before proceeding if the deterministic oracle is wrong.
    # This prevents later statistical results from hiding a structural bug.
    if not results["v2b"]["overall"]:
        raise RuntimeError(
            "V2B FAILED. Pipeline stopped intentionally. "
            "Do not interpret V2C-V8 until the finite-horizon oracle agrees."
        )

    results["v2c"] = run_v2c()

    if not results["v2c"]["overall"]:
        raise RuntimeError(
            "V2C FAILED. Pipeline stopped intentionally."
        )

    results["v2d"] = run_v2d()
    results["v2e"] = run_v2e()
    results["v3"] = run_v3()
    results["v4"] = run_v4()
    results["v5"] = run_v5()
    results["v6"] = run_v6()
    results["v7"] = run_v7()

    if not results["v7"]["overall"]:
        raise RuntimeError(
            "V7 FAILED. Candidate freeze is not scientifically valid."
        )

    results["v8"] = run_v8(
        results["v7"]["candidate_theta"]
    )

    results["elapsed_seconds"] = time.time() - t0

    print_claim_ledger(results)

    print()
    print("=" * 72)
    print("FINAL MACHINE SUMMARY")
    print("=" * 72)

    print(
        f"v1_overall  = {results['v1']['overall']}"
    )
    print(
        f"v2a_overall = {results['v2a']['overall']}"
    )
    print(
        f"v2b_overall = {results['v2b']['overall']}"
    )
    print(
        f"v2c_overall = {results['v2c']['overall']}"
    )
    print(
        f"v2d_overall = {results['v2d']['overall']}"
    )
    print(
        f"v2e_overall = {results['v2e']['overall']}"
    )
    print(
        f"v3_overall  = {results['v3']['overall']}"
    )
    print(
        f"v4_overall  = {results['v4']['overall']}"
    )
    print(
        f"v5_overall  = {results['v5']['overall']}"
    )
    print(
        f"v6_overall  = {results['v6']['overall']}"
    )
    print(
        f"v7_overall  = {results['v7']['overall']}"
    )
    print(
        f"v8_overall  = {results['v8']['overall']}"
    )

    print(
        f"elapsed_seconds = {results['elapsed_seconds']:.3f}"
    )

    all_pass = all(
        results[k]["overall"]
        for k in (
            "v1",
            "v2a",
            "v2b",
            "v2c",
            "v2d",
            "v2e",
            "v3",
            "v4",
            "v5",
            "v6",
            "v7",
            "v8",
        )
    )

    print(f"FULL PIPELINE STATUS = {status(all_pass)}")

    return results


if __name__ == "__main__":
    FINAL_REPORT = main()
