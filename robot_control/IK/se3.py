import numpy as np
from so3 import *


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


def adjoint_transform_list(T: np.ndarray, X_list: list, to_space: bool = True) -> list:
    """
    Computes:
        X'_i = Ad_{T_sb} X_i        (to_space=True)
        X'_i = Ad_{T_sb^{-1}} X_i   (to_space=False)

    Inputs:
        T : (4x4) pose of the body frame in the space frame
        X_list : list of (6,) twist coordinates in the body frame (to_space=True) or space frame (to_space=False)
        to_space : if True, computes Ad_T X_i (body -> space) for the twists;
                   if False, computes Ad_{T^{-1}} X_i (space -> body) for the twists

    Returns:
        X_out_list : list of (6,) transformed twist coordinates
    """
    return [adjoint_transform(T, X, to_space) for X in X_list]