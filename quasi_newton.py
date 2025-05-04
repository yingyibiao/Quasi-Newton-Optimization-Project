# -*- coding: utf-8 -*-
"""
Quasi‑Newton Optimization Project
=================================
Implementation of:
 - BFGS
 - L‑BFGS (two‑loop recursion, memory m)
 - DFP
 - SR1
 - Broyden rank‑1
and baselines:
 - Gradient Descent
 - Newton's Method

Experiments:
 1. Convex quadratic minimization (varying condition number & dimension)
 2. Rosenbrock function minimization
 3. Logistic regression (binary, Iris dataset)

Dependencies: numpy, matplotlib, scikit‑learn (for Iris dataset only)
"""

from __future__ import annotations

import abc
import dataclasses
import math
from collections import deque
from typing import Callable, Tuple, List, Deque, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Line Search (strong Wolfe)
def strong_wolfe(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    p: np.ndarray,
    f_x: float,
    g_x: np.ndarray,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 50,
) -> float:
    """Backtracking line search satisfying strong Wolfe conditions.

    Returns chosen step length alpha ∈ (0, 1].
    """
    alpha = 1.0
    phi0 = f_x
    derphi0 = g_x @ p

    # Check descent direction
    if derphi0 >= 0:
        # Not a descent direction – fallback to steepest descent
        return 0.0

    # Bracketing phase (simple backtracking / doubling)
    for i in range(max_iter):
        x_new = x + alpha * p
        phi = f(x_new)
        if phi > phi0 + c1 * alpha * derphi0 or (i > 0 and phi >= prev_phi):
            return _zoom(
                f,
                grad,
                x,
                p,
                f_x,
                g_x,
                0.0,
                alpha,
                c1,
                c2,
            )
        g_new = grad(x_new)
        if abs(g_new @ p) <= -c2 * derphi0:
            return alpha
        if g_new @ p >= 0:
            return _zoom(
                f,
                grad,
                x,
                p,
                f_x,
                g_x,
                alpha,
                0.0,
                c1,
                c2,
            )
        prev_phi = phi
        alpha *= 2.0  # try larger step
    return alpha


def _zoom(
    f,
    grad,
    x,
    p,
    f_x,
    g_x,
    alo,
    ahi,
    c1,
    c2,
):
    derphi0 = g_x @ p
    max_iter = 25
    for _ in range(max_iter):
        alpha = 0.5 * (alo + ahi)
        phi = f(x + alpha * p)
        phi_lo = f(x + alo * p)
        if phi > f_x + c1 * alpha * derphi0 or phi >= phi_lo:
            ahi = alpha
        else:
            g_new = grad(x + alpha * p)
            if abs(g_new @ p) <= -c2 * derphi0:
                return alpha
            if g_new @ p * (ahi - alo) >= 0:
                ahi = alo
            alo = alpha
    return alpha


# Base Optimizer
@dataclasses.dataclass
class OptResult:
    x: np.ndarray
    history: Dict[str, List]


class Optimizer(abc.ABC):
    """Abstract base class for deterministic optimizers."""

    def __init__(self, f, grad, max_iter=100, tol=1e-6):
        self.f = f
        self.grad = grad
        self.max_iter = max_iter
        self.tol = tol

    def optimize(self, x0: np.ndarray) -> OptResult:
        x = x0.copy()
        history = {"f": [], "grad_norm": [], "alpha": [], "x": []}  # Added "x" for trajectory tracking
        history["x"].append(x.copy())  # Store initial point
        for k in range(self.max_iter):
            f_x = self.f(x)
            g_x = self.grad(x)
            history["f"].append(f_x)
            history["grad_norm"].append(np.linalg.norm(g_x))
            if np.linalg.norm(g_x) < self.tol:
                break
            p = self.direction(x, g_x)
            alpha = strong_wolfe(self.f, self.grad, x, p, f_x, g_x)
            if alpha == 0:
                # fallback to steepest descent direction
                p = -g_x
                alpha = strong_wolfe(self.f, self.grad, x, p, f_x, g_x)
            x = x + alpha * p
            history["x"].append(x.copy())  # Store current point
            history["alpha"].append(alpha)
        return OptResult(x, history)

    @abc.abstractmethod
    def direction(self, x: np.ndarray, g_x: np.ndarray) -> np.ndarray:
        """Return a descent direction."""


# BFGS 
class BFGS(Optimizer):
    def __init__(self, f, grad, max_iter=100, tol=1e-6):
        super().__init__(f, grad, max_iter, tol)
        self.H = None  # inverse Hessian approximation

    def direction(self, x, g):
        if self.H is None:
            self.H = np.eye(len(x))
            self._prev_x = x.copy()
            self._prev_g = g.copy()
            return -self.H @ g
        s = x - self._prev_x
        y = g - self._prev_g
        rho = y @ s
        if rho > 1e-10:  # curvature condition
            rho = 1.0 / rho
            I = np.eye(len(x))
            V = I - rho * np.outer(s, y)
            self.H = V @ self.H @ V.T + rho * np.outer(s, s)
        # update caches
        self._prev_x = x.copy()
        self._prev_g = g.copy()
        return -self.H @ g


# L‑BFGS 
class LBFGS(Optimizer):
    def __init__(self, f, grad, m=5, max_iter=100, tol=1e-6):
        super().__init__(f, grad, max_iter, tol)
        self.m = m
        self.S: Deque[np.ndarray] = deque(maxlen=m)
        self.Y: Deque[np.ndarray] = deque(maxlen=m)
        self.rho: Deque[float] = deque(maxlen=m)
        self.prev_x = None
        self.prev_g = None

    def direction(self, x, g):
        if self.prev_x is None:
            self.prev_x = x.copy()
            self.prev_g = g.copy()
            return -g
        s = x - self.prev_x
        y = g - self.prev_g
        ys = y @ s
        if ys > 1e-10:
            self.S.append(s)
            self.Y.append(y)
            self.rho.append(1.0 / ys)
        q = g.copy()
        alphas = []
        # two‑loop recursion backward pass
        for s_i, y_i, rho_i in zip(reversed(self.S), reversed(self.Y), reversed(self.rho)):
            alpha_i = rho_i * (s_i @ q)
            q -= alpha_i * y_i
            alphas.append(alpha_i)
        # initial H0 as scaling of Identity
        if len(self.Y) > 0:
            y_last = self.Y[-1]
            s_last = self.S[-1]
            gamma = (s_last @ y_last) / (y_last @ y_last)
        else:
            gamma = 1.0
        r = gamma * q
        # forward pass
        for s_i, y_i, rho_i, alpha_i in zip(self.S, self.Y, self.rho, reversed(alphas)):
            beta = rho_i * (y_i @ r)
            r += s_i * (alpha_i - beta)
        self.prev_x = x.copy()
        self.prev_g = g.copy()
        return -r


# DFP
class DFP(Optimizer):
    def __init__(self, f, grad, max_iter=100, tol=1e-6):
        super().__init__(f, grad, max_iter, tol)
        self.H = None
        self.prev_x = None
        self.prev_g = None

    def direction(self, x, g):
        if self.H is None:
            self.H = np.eye(len(x))
            self.prev_x = x.copy()
            self.prev_g = g.copy()
            return -self.H @ g
        s = x - self.prev_x
        y = g - self.prev_g
        ys = y @ s
        if ys > 1e-10:
            Hy = self.H @ y
            self.H += np.outer(s, s) / ys - np.outer(Hy, Hy) / (y @ Hy)
        self.prev_x = x.copy()
        self.prev_g = g.copy()
        return -self.H @ g


# SR1
class SR1(Optimizer):
    def __init__(self, f, grad, max_iter=100, tol=1e-6):
        super().__init__(f, grad, max_iter, tol)
        self.H = None
        self.prev_x = None
        self.prev_g = None

    def direction(self, x, g):
        if self.H is None:
            self.H = np.eye(len(x))
            self.prev_x = x.copy()
            self.prev_g = g.copy()
            return -self.H @ g
        s = x - self.prev_x
        y = g - self.prev_g
        v = s - self.H @ y
        denom = v @ y
        if abs(denom) > 1e-8:
            self.H += np.outer(v, v) / denom
        self.prev_x = x.copy()
        self.prev_g = g.copy()
        p = -self.H @ g
        # ensure descent direction
        if p @ g >= 0:
            self.H = np.eye(len(x))  # reset
            p = -g
        return p


# Broyden Rank‑1
class Broyden(Optimizer):
    def __init__(self, f, grad, max_iter=100, tol=1e-6):
        super().__init__(f, grad, max_iter, tol)
        self.H = None
        self.prev_x = None
        self.prev_g = None

    def direction(self, x, g):
        if self.H is None:
            self.H = np.eye(len(x))
            self.prev_x = x.copy()
            self.prev_g = g.copy()
            return -self.H @ g
        s = x - self.prev_x
        y = g - self.prev_g
        Hy = self.H @ y
        denom = y @ y
        if denom > 1e-10:
            self.H += np.outer(s - Hy, y) / denom  # unsymmetric rank‑1
        self.prev_x = x.copy()
        self.prev_g = g.copy()
        p = -self.H @ g
        if p @ g >= 0:
            p = -g  # fallback
        return p


# Baselines: Gradient Descent
def gradient_descent(f, grad, x0, max_iter=100, tol=1e-6):
    x = x0.copy()
    history = {"f": [], "grad_norm": [], "alpha": [], "x": []}  # Added "x" for trajectory tracking
    history["x"].append(x.copy())  # Store initial point
    for _ in range(max_iter):
        f_x = f(x)
        g = grad(x)
        history["f"].append(f_x)
        history["grad_norm"].append(np.linalg.norm(g))
        if np.linalg.norm(g) < tol:
            break
        p = -g
        alpha = strong_wolfe(f, grad, x, p, f_x, g)
        x = x + alpha * p
        history["x"].append(x.copy())  # Store current point
        history["alpha"].append(alpha)
    return OptResult(x, history)


# Baselines: Newton
def newton_method(f, grad, hess, x0, max_iter=100, tol=1e-6):
    x = x0.copy()
    history = {"f": [], "grad_norm": [], "alpha": [], "x": []}  # Added "x" for trajectory tracking
    history["x"].append(x.copy())  # Store initial point
    for _ in range(max_iter):
        f_x = f(x)
        g = grad(x)
        history["f"].append(f_x)
        history["grad_norm"].append(np.linalg.norm(g))
        if np.linalg.norm(g) < tol:
            break
        H = hess(x)
        try:
            p = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            p = -g  # fallback
        alpha = strong_wolfe(f, grad, x, p, f_x, g)
        x = x + alpha * p
        history["x"].append(x.copy())  # Store current point
        history["alpha"].append(alpha)
    return OptResult(x, history)


# Test Function: Quadratic f(x) = 0.5 x^T Q x
def quadratic(Q):
    def f(x):
        return 0.5 * x @ Q @ x

    def grad(x):
        return Q @ x

    def hess(_x):
        return Q

    return f, grad, hess


# Test Function: Rosenbrock (2‑D)
def rosenbrock():
    def f(z):
        x, y = z
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    def grad(z):
        x, y = z
        return np.array([
            -2 * (1 - x) - 400 * x * (y - x ** 2),
            200 * (y - x ** 2),
        ])

    def hess(z):
        x, y = z
        return np.array([
            [2 - 400 * (y - 3 * x ** 2), -400 * x],
            [-400 * x, 200],
        ])

    return f, grad, hess


# Logistic Regression Model
class LogisticRegressionModel:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X  # shape (N, d)
        self.y = y  # shape (N,)

    def f(self, w: np.ndarray) -> float:
        z = self.X @ w
        log_exp = np.logaddexp(0, -z * self.y)
        return np.mean(log_exp)

    def grad(self, w: np.ndarray) -> np.ndarray:
        z = self.X @ w
        sig = 1.0 / (1.0 + np.exp(-z * self.y))
        coeff = -self.y * (1 - sig)
        return (self.X.T @ coeff) / len(self.y)

    def hess(self, w: np.ndarray) -> np.ndarray:
        z = self.X @ w
        sig = 1.0 / (1.0 + np.exp(-z))
        S = sig * (1 - sig)
        return (self.X.T * S) @ self.X / len(self.y)


# Visualization Utilities
def plot_loss(history_dict: Dict[str, Dict[str, List]], title: str, outfile: str):
    # --- NO CHANGE NEEDED HERE ---
    plt.figure()
    for label, hist in history_dict.items():
        plt.semilogy(hist["f"], label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Objective value (log scale)")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def plot_trajectory(
    f,
    history_dict: Dict[str, Dict[str, List]],
    levels: int = 50,
    outfile: str = "traj.png",
    title: str = "Trajectories on Rosenbrock"
    ):
    plt.figure(figsize=(8, 6))
    # generate contour grid
    xs = np.linspace(-2, 2, 400)
    ys = np.linspace(-1, 3, 400)
    Xg, Yg = np.meshgrid(xs, ys)
    Z = np.vectorize(lambda a, b: f(np.array([a, b])))(Xg, Yg)
    levels_log = np.logspace(np.log10(max(Z.min(), 1e-2)), np.log10(Z.max()), levels)
    plt.contour(Xg, Yg, Z, levels=levels_log, cmap="jet", alpha=0.7)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(history_dict))) # Get distinct colors
    for i, (name, history) in enumerate(history_dict.items()):
        traj = np.array(history["x"])
        if traj.ndim == 2 and traj.shape[1] == 2:
            plt.plot(traj[:, 0], traj[:, 1], "-o",
                     markersize=2, linewidth=1.0,
                     label=name, color=colors[i], alpha=0.9) # Add label & color

    plt.scatter([1], [1], c="black", marker="*", s=100, label="Minimum", zorder=5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title) # Use title parameter
    plt.grid(True, alpha=0.3) # Optional: Adjust grid alpha
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


# Experiments
def run_quadratic_tests():
    np.random.seed(42)
    n = 10
    # varying condition numbers
    for kappa in [10, 100, 1000]:
        # Generate SPD matrix with desired condition
        U, _ = np.linalg.qr(np.random.randn(n, n))
        eigs = np.logspace(0, math.log10(kappa), n)
        Q = U @ np.diag(eigs) @ U.T
        f, grad, hess = quadratic(Q)
        x0 = np.random.randn(n)
        methods = {
            "BFGS": BFGS(f, grad),
            "L-BFGS": LBFGS(f, grad, m=5),
            "DFP": DFP(f, grad),
            "SR1": SR1(f, grad),
            "Broyden": Broyden(f, grad),
            "GD": lambda x0=x0, max_iter=100: gradient_descent(f, grad, x0, max_iter=max_iter),
            "Newton": lambda x0=x0, max_iter=100: newton_method(f, grad, hess, x0, max_iter=max_iter),
        }
        results = {}
        for name, opt in methods.items():
            if callable(opt):
                res = opt()
            else:
                res = opt.optimize(x0)
            results[name] = res.history
        plot_loss(results, f"Quadratic κ={kappa}", f"quadratic_kappa{kappa}.png")


def run_rosenbrock():
    np.random.seed(42)
    f, grad, hess = rosenbrock()
    x0 = np.array([-1.2, 1.0])
    methods = {
        "BFGS": BFGS(f, grad),
        "L-BFGS": LBFGS(f, grad, m=5),
        "DFP": DFP(f, grad),
        "SR1": SR1(f, grad),
        "Broyden": Broyden(f, grad),
        "GD": lambda x0=x0, max_iter=100: gradient_descent(f, grad, x0, max_iter=max_iter), # Increased GD iter
        "Newton": lambda x0=x0, max_iter=100: newton_method(f, grad, hess, x0, max_iter=max_iter),
    }
    results = {}
    traj_results_to_plot = {} # Store histories specifically for the combined plot

    print("Running Rosenbrock Optimizers...")
    for name, opt in methods.items():
        print(f"  Running {name}...")
        if callable(opt):
            res = opt()
        else:
            res = opt.optimize(x0)
        results[name] = res.history # Store history for loss plot

        if name in ["BFGS", "L-BFGS", "DFP", "SR1", "Broyden"]:
            traj_results_to_plot[name] = res.history

    print("Plotting combined loss...")
    plot_loss(results, "Rosenbrock Function Loss", "rosenbrock_loss.png") # Unified title

    print("Plotting combined trajectories...")
    if traj_results_to_plot: # Only plot if we collected some results
        plot_trajectory(
            f,
            traj_results_to_plot, # Pass the dictionary with multiple histories
            outfile="rosenbrock_traj_COMBINED.png", # New output filename
            title="Rosenbrock Trajectories (Combined)" # New title
        )


def run_logistic_regression():
    np.random.seed(42)
    iris = load_iris()
    X = iris.data
    y = iris.target
    mask = y != 0
    X = X[mask]
    y = y[mask]
    y = np.where(y == 1, 1, -1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = np.hstack([np.ones((X_train_orig.shape[0], 1)), X_train_orig])
    X_test = np.hstack([np.ones((X_test_orig.shape[0], 1)), X_test_orig])
    model = LogisticRegressionModel(X_train, y_train)

    x0 = np.zeros(X.shape[1])
    x0 = np.zeros(X_train.shape[1])
    methods = {
        "BFGS": BFGS(model.f, model.grad),
        "L-BFGS": LBFGS(model.f, model.grad, m=5),
        "DFP": DFP(model.f, model.grad),
        "SR1": SR1(model.f, model.grad),
        "Broyden": Broyden(model.f, model.grad),
        "GD": lambda: gradient_descent(model.f, model.grad, x0),
        "Newton": lambda: newton_method(model.f, model.grad, model.hess, x0),
    }

    results = {}
    final_acc = {}
    for name, opt in methods.items():
        if callable(opt):
            res = opt()
        else:
            res = opt.optimize(x0)
        results[name] = res.history
        w_opt = res.x
        preds = np.sign(X_test @ w_opt)

        acc = np.mean(preds == y_test) * 100
        final_acc[name] = acc
        print(f"{name}: Test Accuracy = {acc:.2f}%")

    plot_loss(results, "Logistic Regression", "logistic_loss.png")


# Main
if __name__ == "__main__":
    run_quadratic_tests()
    run_rosenbrock()
    run_logistic_regression()
    print("Experiments completed. Plots saved to current directory.")
