import numpy as np
from se3 import *


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