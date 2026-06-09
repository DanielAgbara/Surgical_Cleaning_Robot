"""
Position-only redundancy resolution IK solver.

This keeps the function name numerical_inverse_kinematics_position
so existing scripts continue to work.

Primary task:
    minimize end-effector position error using J_v

Secondary task:
    maximize manipulability in the null space
"""

import numpy as np
from robot_control.utilities.fk import *
from robot_control.utilities.se3 import *
from robot_control.utilities.jacobian import *


def finite_difference_grad(w_func, theta, eps=1e-6):
    """
    Central-difference gradient of a scalar function w(theta).
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)
    grad = np.zeros_like(theta)

    for i in range(len(theta)):
        t1 = theta.copy()
        t2 = theta.copy()
        t1[i] += eps
        t2[i] -= eps
        grad[i] = (w_func(t1) - w_func(t2)) / (2.0 * eps)

    return grad


def manipulability_measure(B_list, theta):
    """
    Position-only manipulability based on J_v.
    """
    J_b = body_jacobian(B_list, theta)
    J_v = J_b[3:, :]
    val = np.linalg.det(J_v @ J_v.T)
    return np.sqrt(max(val, 0.0))


def damped_least_squares_pinv(J, damping=1e-2):
    """
    Damped least-squares pseudoinverse:
        J# = J^T (J J^T + lambda^2 I)^-1
    """
    m, _ = J.shape
    return J.T @ np.linalg.inv(J @ J.T + (damping ** 2) * np.eye(m))


def numerical_inverse_kinematics_position(
    M_ee: np.ndarray,
    B_list: list,
    theta_init: np.ndarray,
    p_des: np.ndarray,
    max_iters: int = 500,
    tol_converge: float = 1e-6,
    tol_manipulability: float = 1e-3,
    q_min: np.ndarray | None = None,
    q_max: np.ndarray | None = None,
    k_null: float = 0.1,
    k_damping: float = 0.01,
    print_iterations: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Position-only redundancy resolution IK.

    Uses:
        dq = J_v# e + (I - J_v# J_v) * k_null * grad_w

    where grad_w is the gradient of manipulability.
    """
    theta = np.asarray(theta_init, dtype=float).reshape(-1)
    p_des = np.asarray(p_des, dtype=float).reshape(3)

    theta_history = [theta.copy()]
    count = 0

    for i in range(max_iters):
        # Forward kinematics
        T_ee = body_product_of_exponentials(M_ee, B_list, theta)
        p_ee = T_ee[:3, 3]

        # Position error only
        error = p_des - p_ee
        err_norm = np.linalg.norm(error)

        if print_iterations:
            theta_str = ", ".join(
                f"θ{j+1}={np.degrees(t):.2f}°" for j, t in enumerate(theta)
            )
            print(
                f"Iteration {i}: ({theta_str}), "
                f"(x,y,z)=({p_ee[0]:.3f}, {p_ee[1]:.3f}, {p_ee[2]:.3f}), "
                f"||error||={err_norm:.3e}"
            )

        if err_norm < tol_converge:
            break

        # Position Jacobian only
        J_b = body_jacobian(B_list, theta)
        J_v = J_b[3:, :]

        # Use pseudoinverse or DLS near singularity
        if manipulability(J_v) > tol_manipulability:
            J_dagger = pseudoinverse_jacobian(J_v)
        else:
            J_dagger = damped_least_squares_pinv(J_v, damping=k_damping)

        # Primary task
        dq_task = J_dagger @ error

        # Secondary task: maximize manipulability
        w_func = lambda th: manipulability_measure(B_list, th)
        grad_w = finite_difference_grad(w_func, theta)

        # Null-space projector
        P = np.eye(len(theta)) - J_dagger @ J_v

        # Redundancy resolution update
        dq = dq_task + P @ (k_null * grad_w)

        # Update joints
        theta = theta + dq

        # Joint limits
        if q_min is not None and q_max is not None:
            theta = np.clip(theta, q_min, q_max)

        theta_history.append(theta.copy())
        count += 1

    print(f"No of Iterations {count}")

    return theta, np.array(theta_history)