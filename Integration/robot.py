"""
Script that has all the robot function and classes
"""

import time
import numpy as np
from typing import Callable
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from dataclasses import dataclass
from pathlib import Path
import force_sensing, warnings, sys

# --------------------------------------------------
# SO3 Functions
# --------------------------------------------------
def Rx(a: float) -> np.ndarray:
    """
    Computes:
        Rotation matrix for rotation about x-axis by angle a.

    Inputs:
        a : rotation angle in radians

    Returns:
        R : (3x3) rotation matrix
    """
    c, s = np.cos(a), np.sin(a)
    return np.array(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ]
    )


def Ry(a: float) -> np.ndarray:
    """
    Computes:
        Rotation matrix for rotation about y-axis by angle a.

    Inputs:
        a : rotation angle in radians

    Returns:
        R : (3x3) rotation matrix
    """
    c, s = np.cos(a), np.sin(a)
    return np.array(
        [
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ]
    )


def Rz(a: float) -> np.ndarray:
    """
    Computes:
        Rotation matrix for rotation about z-axis by angle a.

    Inputs:
        a : rotation angle in radians

    Returns:
        R : (3x3) rotation matrix
    """
    c, s = np.cos(a), np.sin(a)
    return np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ]
    )


def rpy_to_R(rpy: np.ndarray) -> np.ndarray:
    """
    Computes:
        R = Rz(y) Ry(p) Rx(r)

    Inputs:
        rpy : (3,) roll-pitch-yaw angles in radians

    Returns:
        R : (3x3) rotation matrix
    """
    r, p, y = rpy
    return Rz(y) @ Ry(p) @ Rx(r)


def RToAxisAngle(R: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, float]:
    """
    Computes:
        Rotation axis and angle from a rotation matrix.

    Inputs:
        R : (3x3) rotation matrix
        eps : numerical tolerance for special cases

    Returns:
        w_hat : (3,) rotation axis (unit vector)
        theta : rotation angle in radians
    """
    R = np.asarray(R, dtype=float)
    tr_R = np.trace(R)
    I = np.eye(3)
    # Case I: R = I
    if np.linalg.norm(R - I) < eps:
        warnings.warn("Rotation axis is undefined", UserWarning)
        return np.array([0.0, 0.0, 1.0]), 0.0

    # Case II: tr(R) = -1
    if abs(np.trace(R) + 1) < eps:
        diag = 1 + np.diag(R)
        i = int(np.argmax(diag))

        theta = np.pi

        w_hat = R[:, i].copy()
        w_hat[i] += 1.0
        w_hat = w_hat / np.sqrt(2 * (1 + R[i, i]))

    else:  # Case III: Otherwise
        theta = np.arccos((tr_R - 1) / 2)
        theta = np.clip(theta, 0, np.pi)

        W = (R - np.transpose(R)) / (2 * np.sin(theta))
        w_hat = np.array([W[2, 1], W[0, 2], W[1, 0]])

    w_hat /= np.linalg.norm(w_hat)
    return w_hat, theta


def AxisAngleToR(w: np.ndarray, theta: float) -> np.ndarray:
    """
    Computes:
        R = exp([w] θ) using Rodrigues formula

    Inputs:
        w : (3,) rotation axis (unit vector)
        theta : rotation angle in radians

    Returns:
        R : (3x3) rotation matrix
    """
    w = np.asarray(w, dtype=float)

    wn = np.linalg.norm(w)

    if wn < 1e-12:
        return np.eye(3)

    w = w / wn
    w_hat = skew(w)

    R = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * (w_hat @ w_hat)
    return R

def RToQuaternion(R, eps=1e-6):
    R = np.asarray(R, dtype=float)

    q0 = np.sqrt(R[0, 0] + R[1, 1] + R[2, 2] + 1) / 2
    q1 = np.sign(R[2, 1] - R[1, 2]) * np.sqrt(R[0, 0] - R[1, 1] - R[2, 2] + 1) / 2
    q2 = np.sign(R[0, 2] - R[2, 0]) * np.sqrt(R[1, 1] - R[2, 2] - R[0, 0] + 1) / 2
    q3 = np.sign(R[1, 0] - R[0, 1]) * np.sqrt(R[2, 2] - R[0, 0] - R[1, 1] + 1) / 2

    Q = np.array([q0, q1, q2, q3])
    Q = Q / np.linalg.norm(Q)
    return Q

def QuaternionToR(Q):
    Q = np.asarray(Q, dtype=float)
    q0, q1, q2, q3 = Q
    R = np.array(
        [
            [
                q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
                2 * (q1 * q2 - q0 * q3),
                2 * (q0 * q2 + q1 * q3),
            ],
            [
                2 * (q0 * q3 + q1 * q2),
                q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
                2 * (q2 * q3 - q0 * q1),
            ],
            [
                2 * (q1 * q3 - q0 * q2),
                2 * (q0 * q1 + q2 * q3),
                q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3,
            ],
        ]
    )

    return R


def skew(w: np.ndarray) -> np.ndarray:
    """
    Computes:
        so(3) hat matrix [w]^.

    Inputs:
        w : (3,) rotation vector

    Returns:
        (3x3) skew-symmetric matrix
    """
    w1, w2, w3 = w
    return np.array([[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]])


def unskew(hat_w: np.ndarray) -> np.ndarray:
    """
    Computes:
        w = [w]^∨ from the so(3) hat matrix.

    Inputs:
        hat_w : (3x3) skew-symmetric matrix

    Returns:
        w : (3,) rotation vector
    """
    return np.array([hat_w[2, 1], hat_w[0, 2], hat_w[1, 0]])


# --------------------------------------------------
# SE3 Functions
# --------------------------------------------------
def screw_axis_from_w_q(w: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Computes:
        Revolute screw axis V = [w; v], where v = -w × q.

    Inputs:
        w : (3,) unit rotation axis
        q : (3,) point on the rotation axis

    Returns:
        V : (6,) screw axis
    """
    w = np.asarray(w, dtype=float).reshape(
        3,
    )
    q = np.asarray(q, dtype=float).reshape(
        3,
    )
    return np.concatenate([w, -skew(w) @ q])


def vec_to_se3(V: np.ndarray) -> np.ndarray:
    """
    Computes:
        V = [[w], [v]] -> [V] = [[w] v; 0 0 0 0]

    Inputs:
        V : (6,) twist coordinates

    Returns:
        hat_V : (4x4) se(3) matrix
    """
    hat_V = np.zeros((4, 4))
    hat_V[:3, :3] = skew(V[:3])
    hat_V[:3, 3] = V[3:]
    return hat_V


def exp_screw_hat(hat_S: np.ndarray, theta: float) -> np.ndarray:
    """
    Computes:
        T = exp([S] θ) ∈ SE(3).

    Inputs:
        hat_S : (4x4) se(3) matrix corresponding to screw axis S
        theta : scalar joint displacement

    Returns:
        T : (4x4) homogeneous transformation
    """
    hat_w = hat_S[:3, :3]
    v = hat_S[:3, 3]

    w = np.array([hat_w[2, 1], hat_w[0, 2], hat_w[1, 0]])
    wn = np.linalg.norm(w)

    T = np.eye(4)

    if wn < 1e-12:
        # pure translation
        T[:3, :3] = np.eye(3)
        T[:3, 3] = v * theta
        return T

    R = np.eye(3) + np.sin(theta) * hat_w + (1 - np.cos(theta)) * (hat_w @ hat_w)
    G = (
        np.eye(3) * theta
        + (1 - np.cos(theta)) * hat_w
        + (theta - np.sin(theta)) * (hat_w @ hat_w)
    )

    T[:3, :3] = R
    T[:3, 3] = G @ v
    return T


def exp_screw_axis(S: np.ndarray, theta: float) -> np.ndarray:
    """
    Computes:
        T = exp([S] θ) ∈ SE(3).

    Inputs:
        S : (6,) screw axis
        theta : scalar joint displacement

    Returns:
        T : (4x4) homogeneous transformation
    """
    hat_S = vec_to_se3(S)
    return exp_screw_hat(hat_S, theta)


def log_screw_axis(T: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Computes:
        T = exp([S] θ) -> S, θ

    Inputs:
        T : (4x4) homogeneous transformation

    Returns:
        S : (6,) screw axis
        theta : scalar joint displacement
    """
    R = T[:3, :3]
    p = T[:3, 3]

    w, theta = RToAxisAngle(R)

    if np.linalg.norm(w * theta) < 1e-12:
        # pure translation
        v = p / np.linalg.norm(p)
        return np.concatenate([np.zeros(3), v]), np.linalg.norm(p)

    G_inv = (
        np.eye(3) / theta
        - 0.5 * skew(w)
        + (1 / theta - 0.5 / np.tan(theta / 2)) * (skew(w) @ skew(w))
    )
    v = G_inv @ p

    return np.concatenate([w, v]), theta


def inv_SE3(T: np.ndarray) -> np.ndarray:
    """
    Computes:
        T^{-1} = [[R^T, -R^T p], [0 0 0 1]]

    Inputs:
        T : (4x4) homogeneous transformation

    Returns:
        T_inv : (4x4) inverse homogeneous transformation
    """
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    p = T[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p

    return T_inv


def adjoint(T: np.ndarray) -> np.ndarray:
    """
    Computes:
        Ad_T = [        R,   0]
               [skew(p) R, R].

    Inputs:
        T : (4x4) homogeneous transformation

    Returns:
        Ad_T : (6x6) adjoint matrix
    """
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    p = T[:3, 3]
    return np.block([[R, np.zeros((3, 3))], [skew(p) @ R, R]])


def adjoint_inverse(T: np.ndarray) -> np.ndarray:
    """
    Computes:
        Ad_{T^{-1}} = [        R^T,    0]
                      [-R^T skew(p), R^T].

    Inputs:
        T : (4x4) homogeneous transformation

    Returns:
        Ad_T_inv : (6x6) inverse adjoint matrix
    """
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    p = T[:3, 3]
    Ad_T_inv = np.block([[R.T, np.zeros((3, 3))], [-R.T @ skew(p), R.T]])
    return Ad_T_inv


def adjoint_transform(
    T: np.ndarray, X: np.ndarray, to_space: bool = True
) -> np.ndarray:
    """
    Computes:
        X' = Ad_{T_sb} X        (to_space=True)
        X' = Ad_{T_sb^{-1}} X   (to_space=False)

    Inputs:
        T : (4x4) pose of the body frame in the space frame
        X : (6,) twist coordinates in the body frame (to_space=True) or space frame (to_space=False)
        to_space : if True, computes Ad_T X (body -> space) for the twist;
                   if False, computes Ad_{T^{-1}} X (space -> body) for the twist

    Returns:
        X_out : (6,) transformed twist coordinates
    """
    T = np.asarray(T, dtype=float).reshape(4, 4)
    X = np.asarray(X, dtype=float).reshape(
        6,
    )

    if to_space:
        return adjoint(T) @ X
    return adjoint_inverse(T) @ X


# --------------------------------------------------
# Forward Kinematics Functions
# --------------------------------------------------
def space_product_of_exponentials(
    M: np.ndarray, S_list: list, theta: np.ndarray
) -> np.ndarray:
    """
    Computes:
        T(θ) = exp([S1]θ1) ... exp([Sn]θn) M

    Inputs:
        M : (4x4) home configuration of the end-effector
        S_list : list of (6,) screw axes in the space frame
        theta: (n,) array of joint variables

    Returns:
        T_ee : (4x4) end-effector pose in the space frame
    """
    T = np.eye(4)

    for S, theta_i in zip(S_list, theta):
        T = T @ exp_screw_axis(S, theta_i)

    T_ee = T @ M
    return T_ee


def body_product_of_exponentials(
    M: np.ndarray, B_list: list, theta: np.ndarray
) -> np.ndarray:
    """
    Computes:
        T(θ) = M exp([B1]θ1) ... exp([Bn]θn)

    Inputs:
        M : (4x4) home configuration of the end-effector
        B_list : list of (6,) screw axes in the body frame
        theta: (n,) array of joint variables

    Returns:
        T_ee : (4x4) end-effector pose in the space frame
    """
    T = np.asarray(M, dtype=float).reshape(4, 4)

    for B, theta_i in zip(B_list, theta):
        T = T @ exp_screw_axis(B, theta_i)

    T_ee = T
    return T_ee

# --------------------------------------------------
# Jacobian Functions
# --------------------------------------------------
def space_jacobian(S_list: list, theta: np.ndarray) -> np.ndarray:
    """
    Computes:
        J_s = [S1,
               Ad_{exp([S1]θ1)} S2,
               Ad_{exp([S1]θ1) exp([S2]θ2)} S3,
               ...]

    Inputs:
        S_list : list of (6,) screw axes in the space frame
        theta: (n,) array of joint variables

    Returns:
        J_s : (6xn) space Jacobian
    """

    n = len(S_list)
    J = np.zeros((6, n))
    J[:, 0] = S_list[0]

    T = np.eye(4)
    for i in range(1, n):
        T = T @ exp_screw_axis(S_list[i - 1], theta[i - 1])
        J[:, i] = adjoint(T) @ S_list[i]

    return J


def body_jacobian(B_list: list, theta: np.ndarray) -> np.ndarray:
    """
    Computes:
        J_b = [Ad_{exp(-[Bn]θn) ... exp(-[B2]θ2)} B1,
               Ad_{exp(-[Bn]θn) ... exp(-[B3]θ3)} B2,
               ...
               Bn]

    Inputs:
        B_list : list of (6,) screw axes in the body frame
        theta: (n,) array of joint variables

    Returns:
        J_b : (6xn) body Jacobian
    """
    n = len(B_list)
    J_b = np.zeros((6, n), dtype=float)
    J_b[:, n - 1] = B_list[n - 1]

    T = np.eye(4)
    for i in range(n - 2, -1, -1):
        T = T @ exp_screw_axis(B_list[i + 1], -theta[i + 1])
        J_b[:, i] = adjoint(T) @ B_list[i]

    return J_b


def check_singularity(J: np.ndarray) -> bool:
    """
    Computes:
        True if J is singular (i.e. rank deficient), False otherwise.

    Inputs:
        J : (m,n) Jacobian matrix

    Returns:
        is_singular : boolean indicating if J is singular
    """
    J = np.asarray(J, dtype=float)
    is_singular = np.linalg.matrix_rank(J) < min(J.shape)
    return is_singular


def manipulability(J: np.ndarray) -> float:
    """
    Computes:
        Manipulability measure μ = sqrt(det(J J^T))

    Inputs:
        J : (m,n) Jacobian matrix

    Returns:
        mu : manipulability measure
    """
    J = np.asarray(J, dtype=float)
    A = J @ J.T
    mu = np.sqrt(np.linalg.det(A))
    return mu


def manipulability_ellipsoid(J: np.ndarray) -> list:
    """
    Computes:
        Manipulability ellipsoid data for the angular and linear velocity subspaces.

    Inputs:
        J : (6,n) Jacobian matrix

    Returns:
        data : [[A_w, eigvals_w, eigvecs_w, mu1_w, mu2_w, mu3_w],
                [A_v, eigvals_v, eigvecs_v, mu1_v, mu2_v, mu3_v]]

                * = w (angular) or v (linear)
                A_* : (3x3) manipulability matrix J_* J_*^T
                eigvals_* : (3,) eigenvalues of A_*
                eigvecs_* : (3x3) principal axes of the ellipsoid
                mu1_* : sqrt(λ_max / λ_min)  (Isotropy measure)
                mu2_* : λ_max / λ_min (Condition number measure)
                mu3_* : sqrt(det(A_*)) (Volume of the manipulability ellipsoid)
    """
    J = np.asarray(J, dtype=float)
    J_w = J[:3, :]
    J_v = J[3:, :]

    A_w = J_w @ J_w.T
    eigvals_w, eigvecs_w = np.linalg.eigh(A_w)
    if np.min(eigvals_w) < 1e-12:
        mu1_w = np.inf
        mu2_w = np.inf
    else:
        mu1_w = np.sqrt(np.max(eigvals_w) / np.min(eigvals_w))
        mu2_w = np.max(eigvals_w) / np.min(eigvals_w)
    mu3_w = np.sqrt(np.linalg.det(A_w))

    A_v = J_v @ J_v.T
    eigvals_v, eigvecs_v = np.linalg.eigh(A_v)
    if np.min(eigvals_v) < 1e-12:
        mu1_v = np.inf
        mu2_v = np.inf
    else:
        mu1_v = np.sqrt(np.max(eigvals_v) / np.min(eigvals_v))
        mu2_v = np.max(eigvals_v) / np.min(eigvals_v)
    mu3_v = np.sqrt(np.linalg.det(A_v))

    return [
        [A_w, eigvals_w, eigvecs_w, mu1_w, mu2_w, mu3_w],
        [A_v, eigvals_v, eigvecs_v, mu1_v, mu2_v, mu3_v],
    ]


def pseudoinverse_jacobian(J: np.ndarray) -> np.ndarray:
    """
    Computes:
        Pseudo-inverse of the Jacobian matrix J.
        J_dagger = J^T (J J^T)^{-1} if n > m and J has full row rank
        J_dagger = (J^T J)^{-1} J^T if n <= m and J has full column rank

    Inputs:
        J : (m,n) Jacobian matrix

    Returns:
        J_dagger : (nxm) pseudo-inverse of J
    """
    J = np.asarray(J, dtype=float)
    m, n = J.shape
    r = np.linalg.matrix_rank(J)

    if n > m:  # fat Jacobian, use right pseudo-inverse
        if r != m:
            raise ValueError("Jacobian does not have full row rank")
        J_dagger = J.T @ np.linalg.inv(J @ J.T)
    else:  # tall Jacobian, use left pseudo-inverse
        if r != n:
            raise ValueError("Jacobian does not have full column rank")
        J_dagger = np.linalg.inv(J.T @ J) @ J.T

    return J_dagger


def damped_least_square_inverse(J, k=0.01):
    """
    Computes:
        J^* = J^T (JJ^T + k^2 I)^(-1)

    Inputs:
        J : (m x n) Jacobian matrix
        k : damping factor

    Returns:
        J^* : (n x m) damped least squares pseudo-inverse of J
    """
    JJT = J @ J.T
    J_star = J.T @ np.linalg.inv((JJT + k**2 * np.eye(JJT.shape[0])))
    return J_star


# --------------------------------------------------
# Inverse Kinematics Functions
# --------------------------------------------------
def jacobian_transpose_position(
    M_ee,
    B_list,
    theta_init,
    p_des,
    max_iters=100,
    tol_converge=1e-3,
    q_min=None,
    q_max=None,
    alpha_epsilon=1e-12,
    print_iterations=False,
):
    """
    Numerical inverse kinematics for position only using the
    body Jacobian transpose method.

    The Jacobian-transpose step size follows Samuel R. Buss:

        dq = alpha * J_v.T @ error_b

        alpha =
            <error_b, J_v J_v.T error_b>
            --------------------------------
            <J_v J_v.T error_b,
             J_v J_v.T error_b>

    where:

        error_b = R_sb.T @ (p_des - p_ee)

    Joint limits are enforced using np.clip().

    Parameters
    ----------
    M_ee : np.ndarray, shape (4, 4)
        End-effector home configuration.

    B_list : array-like
        Body screw axes.

    theta_init : array-like
        Initial joint angles in radians.

    p_des : array-like, shape (3,)
        Desired end-effector position in the base frame.

    max_iters : int
        Maximum number of iterations.

    tol_converge : float
        Position-error convergence tolerance in meters.

    q_min, q_max : array-like or None
        Joint limits in radians.

    alpha_epsilon : float
        Small threshold used to avoid division by zero when
        calculating the adaptive step size.

    print_iterations : bool
        Print iteration information.

    Returns
    -------
    theta : np.ndarray
        Final joint angles in radians.

    theta_history : np.ndarray
        Joint-angle history.

    error_norm_history : np.ndarray
        Position-error norm history in meters.
    """

    theta = np.asarray(
        theta_init,
        dtype=float,
    ).reshape(-1).copy()

    p_des = np.asarray(
        p_des,
        dtype=float,
    ).reshape(3)

    n = len(theta)

    # --------------------------------------------------
    # Joint limits
    # --------------------------------------------------

    if q_min is None:
        q_min = np.deg2rad(
            np.array(
                [-105, -95, -90, -90, -90, -90],
                dtype=float,
            )
        )

    if q_max is None:
        q_max = np.deg2rad(
            np.array(
                [105, 105, 95, 90, 90, 90],
                dtype=float,
            )
        )

    q_min = np.asarray(
        q_min,
        dtype=float,
    ).reshape(-1)

    q_max = np.asarray(
        q_max,
        dtype=float,
    ).reshape(-1)

    if q_min.shape != (n,):
        raise ValueError(
            "q_min must contain one value for each joint."
        )

    if q_max.shape != (n,):
        raise ValueError(
            "q_max must contain one value for each joint."
        )

    if np.any(q_min >= q_max):
        raise ValueError(
            "Every lower joint limit must be smaller than "
            "its corresponding upper joint limit."
        )

    # Ensure that the initial configuration is valid.
    theta = np.clip(
        theta,
        q_min,
        q_max,
    )

    # --------------------------------------------------
    # History
    # --------------------------------------------------

    theta_history = []
    error_norm_history = []

    converged = False

    # --------------------------------------------------
    # IK iterations
    # --------------------------------------------------

    for i in range(max_iters + 1):

        # Current end-effector pose in the base frame.
        T_sb = body_product_of_exponentials(
            M_ee,
            B_list,
            theta,
        )

        R_sb = T_sb[:3, :3]
        p_ee = T_sb[:3, 3]

        # Position error expressed in the base frame.
        error_space = p_des - p_ee

        # Because we use the body Jacobian, express the
        # position error in the body/end-effector frame.
        error_body = R_sb.T @ error_space

        error_norm = float(
            np.linalg.norm(error_space)
        )

        theta_history.append(theta.copy())
        error_norm_history.append(error_norm)

        if print_iterations:
            theta_deg = np.rad2deg(theta)

            joint_text = ", ".join(
                f"theta{j + 1}={theta_deg[j]:.2f}deg"
                for j in range(n)
            )

            print(
                f"Iteration {i}: "
                f"({joint_text}), "
                f"(x,y,z)=("
                f"{p_ee[0]:.3f}, "
                f"{p_ee[1]:.3f}, "
                f"{p_ee[2]:.3f}), "
                f"||error||={error_norm:.3e}"
            )

        # Check convergence.
        if error_norm < tol_converge:
            converged = True
            break

        if i == max_iters:
            break

        # --------------------------------------------------
        # Position Jacobian
        # --------------------------------------------------

        J_b = body_jacobian(
            B_list,
            theta,
        )

        # Body Jacobian convention:
        #
        #   J_b[0:3, :] = angular component
        #   J_b[3:6, :] = linear component
        #
        J_v = J_b[3:6, :]

        # --------------------------------------------------
        # Samuel R. Buss adaptive step size
        # --------------------------------------------------

        # J_v J_v.T error
        JJT_error = J_v @ (
            J_v.T @ error_body
        )

        numerator = float(
            error_body @ JJT_error
        )

        denominator = float(
            JJT_error @ JJT_error
        )

        if denominator <= alpha_epsilon:
            if print_iterations:
                print(
                    "[IK] Stopping because "
                    "||J_v J_v.T error|| is approximately zero."
                )

            break

        alpha = numerator / denominator

        if not np.isfinite(alpha) or alpha <= 0.0:
            if print_iterations:
                print(
                    "[IK] Stopping because alpha is invalid: "
                    f"{alpha}"
                )

            break

        # Jacobian-transpose update.
        dq = alpha * (
            J_v.T @ error_body
        )

        # Update and clamp joint angles.
        theta = theta + dq

        theta = np.clip(
            theta,
            q_min,
            q_max,
        )

        if print_iterations:
            print(
                f"    alpha={alpha:.6e}, "
                f"||dq||={np.linalg.norm(dq):.6e}, "
                f"max|dq|="
                f"{np.max(np.abs(np.rad2deg(dq))):.3f}deg"
            )

    if print_iterations:
        if converged:
            print(
                "[IK] Position tolerance reached."
            )
        else:
            print(
                "[IK] Solver stopped without reaching "
                "the requested position tolerance."
            )

    return (
        theta,
        np.asarray(theta_history),
        np.asarray(error_norm_history),
    )

def jacobian_transpose_pose(
    M_ee,
    B_list,
    theta_init,
    T_sd,
    max_iters=100,
    tol_w=1e-6,
    tol_v=1e-6,
    q_min=None,
    q_max=None,
    alpha_epsilon=1e-12,
    print_iterations=False,
):
    """
    Numerical inverse kinematics for full pose using the
    body Jacobian transpose method.

    The Jacobian-transpose step size is calculated using the
    Samuel R. Buss formula:

        dq = alpha * J_b.T @ Vb

        alpha =
            <Vb, J_b J_b.T Vb>
            -------------------
            <J_b J_b.T Vb, J_b J_b.T Vb>

    Joint limits are enforced afterward using np.clip().
    """

    theta = np.asarray(
        theta_init,
        dtype=float,
    ).reshape(-1).copy()

    n = len(theta)

    if q_min is None:
        q_min = np.deg2rad(
            np.array(
                [-105, -95, -90, -90, -90, -90],
                dtype=float,
            )
        )

    if q_max is None:
        q_max = np.deg2rad(
            np.array(
                [105, 105, 95, 90, 90, 90],
                dtype=float,
            )
        )

    q_min = np.asarray(
        q_min,
        dtype=float,
    ).reshape(-1)

    q_max = np.asarray(
        q_max,
        dtype=float,
    ).reshape(-1)

    if q_min.shape != (n,):
        raise ValueError(
            "q_min must contain one limit for each joint."
        )

    if q_max.shape != (n,):
        raise ValueError(
            "q_max must contain one limit for each joint."
        )

    if np.any(q_min >= q_max):
        raise ValueError(
            "Each lower joint limit must be smaller than "
            "its corresponding upper joint limit."
        )

    # Ensure the initial guess starts inside the joint limits.
    theta = np.clip(
        theta,
        q_min,
        q_max,
    )

    theta_history = []
    norm_w_b_hist = []
    norm_v_b_hist = []

    for i in range(max_iters + 1):

        # Current end-effector pose in the base/space frame.
        T_sb = body_product_of_exponentials(
            M_ee,
            B_list,
            theta,
        )

        # Body-frame pose error:
        #
        #     T_bd = inv(T_sb) @ T_sd
        #
        T_bs = inv_SE3(T_sb)
        T_bd = T_bs @ T_sd

        # log_screw_axis() returns:
        #
        #     T_bd = exp([S_err] * theta_err)
        #
        # Therefore:
        #
        #     Vb = S_err * theta_err
        #
        S_err, theta_err = log_screw_axis(T_bd)

        Vb = (
            np.asarray(
                S_err,
                dtype=float,
            ).reshape(6)
            * float(theta_err)
        )

        w_b = Vb[0:3]
        v_b = Vb[3:6]

        norm_w = float(np.linalg.norm(w_b))
        norm_v = float(np.linalg.norm(v_b))

        theta_history.append(theta.copy())
        norm_w_b_hist.append(norm_w)
        norm_v_b_hist.append(norm_v)

        if print_iterations:
            theta_deg = np.rad2deg(theta)

            joint_text = ", ".join(
                f"theta{j + 1}={theta_deg[j]:.2f}deg"
                for j in range(n)
            )

            print(
                f"Iteration {i}: "
                f"({joint_text}), "
                f"(x,y,z)=("
                f"{T_sb[0, 3]:.3f}, "
                f"{T_sb[1, 3]:.3f}, "
                f"{T_sb[2, 3]:.3f}), "
                f"||w_b||={norm_w:.3e}, "
                f"||v_b||={norm_v:.3e}"
            )

        if norm_w < tol_w and norm_v < tol_v:
            break

        if i == max_iters:
            break

        J_b = body_jacobian(
            B_list,
            theta,
        )

        # --------------------------------------------------
        # Samuel R. Buss Jacobian-transpose step size
        # --------------------------------------------------
        #
        # predicted_error_change = J J^T e
        #
        JJT_Vb = J_b @ (J_b.T @ Vb)

        numerator = float(
            Vb @ JJT_Vb
        )

        denominator = float(
            JJT_Vb @ JJT_Vb
        )

        # If J J^T e is approximately zero, the Jacobian
        # transpose cannot produce useful motion for this error.
        if denominator <= alpha_epsilon:
            if print_iterations:
                print(
                    "[IK] Stopping because "
                    "||J J^T Vb|| is approximately zero."
                )

            break

        alpha = numerator / denominator

        if not np.isfinite(alpha) or alpha <= 0.0:
            if print_iterations:
                print(
                    "[IK] Stopping because alpha is invalid: "
                    f"{alpha}"
                )

            break

        dq = alpha * (J_b.T @ Vb)

        theta = theta + dq

        # Retain your chosen joint-limit enforcement.
        theta = np.clip(
            theta,
            q_min,
            q_max,
        )

        if print_iterations:
            print(
                f"    alpha={alpha:.6e}, "
                f"||dq||={np.linalg.norm(dq):.6e}, "
                f"max|dq|="
                f"{np.max(np.abs(np.rad2deg(dq))):.3f}deg"
            )

    return (
        theta,
        np.asarray(theta_history),
        np.asarray(norm_w_b_hist),
        np.asarray(norm_v_b_hist),
    )


# --------------------------------------------------
# Robot geometry
# Measurements are in meters and degrees
# --------------------------------------------------

w1 = np.array([0, 0, -1])
q1 = np.array([0.038, 0, 0.065])

w2 = np.array([0, 1, 0])
q2 = np.array([0.06874, 0, 0.105])

w3 = np.array([0, 1, 0])
q3 = np.array([0.097, 0, 0.228])

w4 = np.array([0, 1, 0])
q4 = np.array([0.225, 0, 0.228])

w5 = np.array([1, 0, 0])
q5 = np.array([0.289, 0, 0.228])

w6 = np.array([0, 1, 0])
q6 = np.array([0.326, 0, 0.228])


M = np.array([
    [1, 0, 0, 0.430],
    [0, 1, 0, 0.000],
    [0, 0, 1, 0.228],
    [0, 0, 0, 1.000]
])


S_list = [
    screw_axis_from_w_q(w1, q1),
    screw_axis_from_w_q(w2, q2),
    screw_axis_from_w_q(w3, q3),
    screw_axis_from_w_q(w4, q4),
    screw_axis_from_w_q(w5, q5),
    screw_axis_from_w_q(w6, q6),
]


# Convert screw axes to body frame
B_list = [adjoint(np.linalg.inv(M)) @ S for S in S_list]


# --------------------------------------------------
# Joint offset handling
# --------------------------------------------------
# Robot command angle and IK angle are not always the same.
#
# For joint 6:
#   robot command range: 0 to 100 deg
#   physical home:      50 deg
#   IK zero:            50 deg robot command
#
# Therefore:
#   theta_ik_deg = theta_robot_deg - 50
#   theta_robot_deg = theta_ik_deg + 50
# --------------------------------------------------

JOINT_OFFSETS_DEG = np.array([
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    50.0
])


# Robot command limits in degrees
theta_min_robot_deg = np.array([
    -120.0,
    -106.0,
    -97.0,
    -95.0,
    -180.0,
    0.0
])

theta_max_robot_deg = np.array([
    120.0,
    106.0,
    97.0,
    95.0,
    180.0,
    100.0
])


# IK limits in radians
# These are offset-corrected limits used by FK/Jacobian/IK.
theta_min = np.radians(theta_min_robot_deg - JOINT_OFFSETS_DEG)
theta_max = np.radians(theta_max_robot_deg - JOINT_OFFSETS_DEG)


# --------------------------------------------------
# Robot poses
# These are physical robot command angles in degrees.
# --------------------------------------------------

home = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 50.0,
}


rest = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -105.0,
    "elbow_flex.pos": 95.0,
    "wrist_flex.pos": -90.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 50.0,
}


class SOArm101:
    def __init__(self, port="/dev/ttyACM0", id="dbot"):
        self.M = M
        self.S_list = S_list
        self.B_list = B_list

        self.home = home
        self.rest = rest

        self.joint_offsets_deg = JOINT_OFFSETS_DEG

        # IK limits, radians
        self.theta_max = theta_max
        self.theta_min = theta_min

        # Physical robot command limits, degrees
        self.theta_max_robot_deg = theta_max_robot_deg
        self.theta_min_robot_deg = theta_min_robot_deg

        self.robot = SO101Follower(SO101FollowerConfig(port=port, id=id))

        # Assume scripts begin with robot in rest pose
        self.current_action = dict(self.rest)

    
    def connect(self, calibrate=False):
        self.robot.connect(calibrate=calibrate)


    def disconnect(self):
        self.robot.disconnect()

    # --------------------------------------------------
    # Joint conversion helpers
    # --------------------------------------------------

    def robot_deg_to_ik_deg(self, theta_robot_deg):
        """
        Convert physical robot command angles to IK/model angles.

        Robot joint 6:
            0 to 100 deg command

        IK joint 6:
            -50 to +50 deg

        Parameters
        ----------
        theta_robot_deg : array-like, shape (6,)
            Robot command angles in degrees.

        Returns
        -------
        theta_ik_deg : np.ndarray, shape (6,)
            Offset-corrected IK angles in degrees.
        """

        theta_robot_deg = np.asarray(theta_robot_deg, dtype=float).flatten()

        if len(theta_robot_deg) != 6:
            raise ValueError("theta_robot_deg must contain 6 joint values")

        return theta_robot_deg - self.joint_offsets_deg


    def ik_deg_to_robot_deg(self, theta_ik_deg):
        """
        Convert IK/model angles back to physical robot command angles.

        Parameters
        ----------
        theta_ik_deg : array-like, shape (6,)
            Offset-corrected IK angles in degrees.

        Returns
        -------
        theta_robot_deg : np.ndarray, shape (6,)
            Robot command angles in degrees.
        """

        theta_ik_deg = np.asarray(theta_ik_deg, dtype=float).flatten()

        if len(theta_ik_deg) != 6:
            raise ValueError("theta_ik_deg must contain 6 joint values")

        theta_robot_deg = theta_ik_deg + self.joint_offsets_deg

        theta_robot_deg = np.clip(
            theta_robot_deg,
            self.theta_min_robot_deg,
            self.theta_max_robot_deg
        )

        return theta_robot_deg
    
    def get_joint_angles_deg(self) -> np.ndarray:
        """
        Return the currently observed physical joint angles in degrees.

        These values come from motor feedback, not from the most recent
        commanded action.
        """

        joint_names = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

        observation = self.robot.get_observation()

        missing_joints = [
            name
            for name in joint_names
            if name not in observation
        ]

        if missing_joints:
            raise KeyError(
                "Observation is missing the following joints: "
                + ", ".join(missing_joints)
            )

        return np.array(
            [observation[name] for name in joint_names],
            dtype=float,
        )
    
    def get_theta_rad(self):
        """
        Get current commanded joint angles as IK/model angles in radians.
        """

        theta_robot_deg = self.get_joint_angles_deg()
        theta_ik_deg = self.robot_deg_to_ik_deg(theta_robot_deg)

        return np.radians(theta_ik_deg)


    def get_T_base_to_ee(self):
        """
        Compute FK using offset-corrected IK/model angles.
        """

        theta_robot_deg = self.get_joint_angles_deg()
        theta_ik_deg = self.robot_deg_to_ik_deg(theta_robot_deg)
        theta_ik_rad = np.radians(theta_ik_deg)

        T_base_to_ee = space_product_of_exponentials(
            self.M,
            self.S_list,
            theta_ik_rad
        )

        return T_base_to_ee


    def moveSO101(self, target_action, max_step_deg=2.0, step_delay=0.05):
        """
        Move the physical robot using robot command angles in degrees.
        """

        joint_names = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

        # Fill missing joints from current state
        final_action = dict(self.current_action)

        for name in joint_names:
            if name in target_action:
                final_action[name] = float(target_action[name])

        # Clip to physical robot command limits
        target_deg = np.array(
            [final_action[name] for name in joint_names],
            dtype=float
        )

        target_deg = np.clip(
            target_deg,
            self.theta_min_robot_deg,
            self.theta_max_robot_deg
        )

        final_action = {
            name: float(target_deg[i])
            for i, name in enumerate(joint_names)
        }

        current = np.array(
            [self.current_action[name] for name in joint_names],
            dtype=float
        )

        target = np.array(
            [final_action[name] for name in joint_names],
            dtype=float
        )

        diff = target - current
        max_diff = np.max(np.abs(diff))

        if max_diff < 1e-9:
            return

        n_steps = max(1, int(np.ceil(max_diff / max_step_deg)))

        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            intermediate = current + alpha * diff

            action = {
                name: float(intermediate[idx])
                for idx, name in enumerate(joint_names)
            }

            self.robot.send_action(action)
            time.sleep(step_delay)

        # Update state after motion completes
        self.current_action = final_action


    def move_to_home(self, max_step_deg=2.0, step_delay=0.05):
        self.moveSO101(
            self.home,
            max_step_deg=max_step_deg,
            step_delay=step_delay
        )


    def move_to_rest(self, max_step_deg=2.0, step_delay=0.05):
        self.moveSO101(
            self.rest,
            max_step_deg=max_step_deg,
            step_delay=step_delay
        )


    def V_to_pos(self, previous_pos, velocity, duration, timestep):
        """
        Convert velocity command into final position using Euler integration.

        Parameters
        ----------
        previous_pos : array-like, shape (3,)
            Starting position [x, y, z]

        velocity : array-like, shape (3,)
            Linear velocity [vx, vy, vz]

        duration : float
            How long to apply the velocity [s]

        timestep : float
            Integration timestep [s]

        Returns
        -------
        final_pos : np.ndarray, shape (3,)
            Final position after applying velocity

        pos_history : np.ndarray, shape (N, 3)
            Position history during integration
        """

        previous_pos = np.asarray(previous_pos, dtype=float).reshape(3)
        velocity = np.asarray(velocity, dtype=float).reshape(3)

        if duration <= 0:
            return previous_pos.copy(), np.array([previous_pos.copy()])

        if timestep <= 0:
            raise ValueError("timestep must be greater than zero")

        pos = previous_pos.copy()
        pos_history = [pos.copy()]

        elapsed = 0.0

        while elapsed < duration:
            dt = min(timestep, duration - elapsed)

            # Euler integration
            pos = pos + velocity * dt

            pos_history.append(pos.copy())

            elapsed += dt

        return pos, np.asarray(pos_history)
    
    def solve_position(
        self,
        p_des,
        theta_init=None,
        max_iters=100,
        tol_converge=2e-3
    ):
        """
        Solve IK for desired EE position.

        Input theta_init is expected in physical robot command degrees.

        Returns
        -------
        theta_sol_robot_deg : np.ndarray
            Joint angles in physical robot command degrees.
        """

        p_des = np.asarray(p_des, dtype=float).reshape(3)

        if theta_init is None:
            theta_robot_deg = self.get_joint_angles_deg()
        else:
            theta_robot_deg = np.asarray(theta_init, dtype=float).flatten()

        if len(theta_robot_deg) != 6:
            raise ValueError("theta_init must contain 6 joint values")

        # Convert physical robot command angles to IK/model angles
        theta_ik_deg = self.robot_deg_to_ik_deg(theta_robot_deg)

        # Convert IK/model angles to radians for the solver
        theta_ik_rad = np.radians(theta_ik_deg)

        theta_sol_ik_rad, _ = jacobian_transpose_position(
            M_ee=self.M,
            B_list=self.B_list,
            theta_init=theta_ik_rad,
            p_des=p_des,
            max_iters=max_iters,
            tol_converge=tol_converge,
            q_min=self.theta_min,
            q_max=self.theta_max
        )

        # Convert IK result to degrees
        theta_sol_ik_deg = np.degrees(theta_sol_ik_rad)

        # Convert IK/model angles back to physical robot command angles
        theta_sol_robot_deg = self.ik_deg_to_robot_deg(theta_sol_ik_deg)

        return theta_sol_robot_deg
    
    def solve_pose(
        self,
        T_des,
        theta_init=None,
        max_iters=100,
        tol_w=1e-6,
        tol_v=1e-6,
    ):
        """
        Solve IK for a desired end-effector pose.

        Parameters
        ----------
        T_des : np.ndarray, shape (4,4)
            Desired end-effector pose in the base frame.

        theta_init : array-like, optional
            Initial guess in physical robot command degrees.
            If None, the current robot configuration is used.

        max_iters : int
            Maximum IK iterations.

        tol_w : float
            Orientation convergence tolerance.

        tol_v : float
            Position convergence tolerance.

        K : np.ndarray, optional
            6x6 gain matrix.

        Returns
        -------
        theta_sol_robot_deg : np.ndarray
            Solution in robot command degrees.

        theta_history_robot_deg : np.ndarray
            Robot command angle history.

        norm_w_hist : np.ndarray
            Orientation error history.

        norm_v_hist : np.ndarray
            Position error history.
        """

        T_des = np.asarray(T_des, dtype=float).reshape(4, 4)

        # Initial guess
        if theta_init is None:
            theta_robot_deg = self.get_joint_angles_deg()
        else:
            theta_robot_deg = np.asarray(theta_init, dtype=float).flatten()

        if len(theta_robot_deg) != 6:
            raise ValueError("theta_init must contain 6 joint values")

        # Robot command -> IK model
        theta_ik_deg = self.robot_deg_to_ik_deg(theta_robot_deg)
        theta_ik_rad = np.radians(theta_ik_deg)

        # Solve pose IK
        (
            theta_sol_ik_rad,
            theta_history_ik_rad,
            norm_w_hist,
            norm_v_hist,
        ) = jacobian_transpose_pose(
            M_ee=self.M,
            B_list=self.B_list,
            theta_init=theta_ik_rad,
            T_sd=T_des,
            max_iters=max_iters,
            tol_w=tol_w,
            tol_v=tol_v,
            q_min=self.theta_min,
            q_max=self.theta_max,
        )

        # Final solution back to robot command angles
        theta_sol_ik_deg = np.degrees(theta_sol_ik_rad)
        theta_sol_robot_deg = self.ik_deg_to_robot_deg(theta_sol_ik_deg)

        # Convert history back to robot command angles
        theta_history_robot_deg = []

        for theta in theta_history_ik_rad:
            theta_deg = np.degrees(theta)
            theta_robot = self.ik_deg_to_robot_deg(theta_deg)
            theta_history_robot_deg.append(theta_robot)

        theta_history_robot_deg = np.asarray(theta_history_robot_deg)

        return (
            theta_sol_robot_deg,
            theta_history_robot_deg,
            norm_w_hist,
            norm_v_hist,
        )
    


