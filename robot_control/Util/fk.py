import numpy as np
from se3 import *


def space_product_of_transforms(M: np.ndarray, T_list: list) -> np.ndarray:
    """
    Computes:
        T = T1 T2 ... Tn M

    Inputs:
        M : (4x4) home configuration of the end-effector
        T_list : list of (4x4) joint-generated homogeneous transforms

    Returns:
        T_ee : (4x4) end-effector pose in the space frame
    """
    T = np.eye(4)

    for T_i in T_list:
        T = T @ T_i

    T_ee = T @ M
    return T_ee


def body_product_of_transforms(M: np.ndarray, T_list: list) -> np.ndarray:
    """
    Computes:
        T = M T1 T2 ... Tn

    Inputs:
        M : (4x4) home configuration of the end-effector
        T_list : list of (4x4) joint-generated homogeneous transforms

    Returns:
        T_ee : (4x4) end-effector pose in the space frame
    """
    T = np.asarray(M, dtype=float).reshape(4, 4)

    for T_i in T_list:
        T = T @ T_i

    T_ee = T
    return T_ee


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


### followings are just for visualizing the system.
### Used ChatGPT to create this functions
def space_link_poses(M_list: list, S_list: list, theta: np.ndarray) -> list:
    """
    Computes:
        T_i = exp([S1]θ1) ... exp([Si]θi) M_i for i = 0, 1, ..., n

    Inputs:
        M_list : list of (4x4) home configurations for each link frame in the space frame
        S_list : list of (6,) screw axes in the space frame
        theta: (n,) array of joint variables

    Returns:
        T_links : list of (4x4) homogeneous transforms for each link frame in the space frame
    """
    T_list = []
    T_prefix = np.eye(4)

    T_list.append(T_prefix @ np.asarray(M_list[0], dtype=float).reshape(4, 4))

    for i in range(len(S_list)):
        T_prefix = T_prefix @ exp_screw_axis(S_list[i], theta[i])
        T_i = T_prefix @ np.asarray(M_list[i + 1], dtype=float).reshape(4, 4)
        T_list.append(T_i)

    return T_list