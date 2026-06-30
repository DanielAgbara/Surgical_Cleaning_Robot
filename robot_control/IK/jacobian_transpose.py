import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UTIL_PATH = ROOT / "Util"

from fk import body_product_of_exponentials
from jacobian import body_jacobian
from se3 import inv_SE3, log_screw_axis


def jacobian_transpose_position(
    M_ee,
    B_list,
    theta_init,
    p_des,
    max_iters=100,
    tol_converge= 1e-3,
    q_min=None,
    q_max=None,
    K=None,
    print_iterations=True,
):
    """
    Numerical inverse kinematics for position only using
    the body Jacobian transpose method.

    This solves for joint angles theta such that the end-effector
    position gets close to p_des.

    Parameters
    ----------
    M_ee : np.ndarray, shape (4, 4)
        Home configuration of the end-effector.

    B_list : list of np.ndarray
        Body screw axes.

    theta_init : array-like
        Initial joint angles in radians.

    p_des : array-like, shape (3,)
        Desired end-effector position [x, y, z].

    max_iters : int
        Maximum number of IK iterations.

    tol_converge : float
        Position error tolerance.

    q_min : array-like or None
        Minimum joint limits in radians.

    q_max : array-like or None
        Maximum joint limits in radians.

    K : np.ndarray, shape (3, 3)
        Gain matrix for position error.

    print_iterations : bool
        Print iteration information.

    Returns
    -------
    theta : np.ndarray
        Final joint angles in radians.

    theta_history : np.ndarray
        Joint history, shape (num_steps, num_joints).
    """

    theta = np.asarray(theta_init, dtype=float).flatten()
    p_des = np.asarray(p_des, dtype=float).reshape(3)

    n = len(theta)

    if q_min is None:
        q_min = np.deg2rad(
            np.array([-105, -95, -90, -90, -90, -90], dtype=float)
        )

    if q_max is None:
        q_max = np.deg2rad(
            np.array([105, 105, 95, 90, 90, 90], dtype=float)
        )

    q_min = np.asarray(q_min, dtype=float).flatten()
    q_max = np.asarray(q_max, dtype=float).flatten()

    if K is None:
        K = np.eye(3)

    K = np.asarray(K, dtype=float).reshape(3, 3)

    theta_history = [theta.copy()]

    for i in range(max_iters):

        # Forward kinematics using body POE
        T_ee = body_product_of_exponentials(
            M_ee,
            B_list,
            theta
        )

        # Current end-effector position
        p_ee = T_ee[:3, 3]

        # Position error in task space
        error_vec = p_des - p_ee

        error_norm = np.linalg.norm(error_vec)

        if print_iterations:
            theta_deg = np.rad2deg(theta)

            joint_text = ", ".join(
                [
                    f"theta{j + 1}={theta_deg[j]:.2f}deg"
                    for j in range(n)
                ]
            )

            print(
                f"Iteration {i}: "
                f"({joint_text}), "
                f"(x,y,z)=({p_ee[0]:.3f}, {p_ee[1]:.3f}, {p_ee[2]:.3f}), "
                f"||error||={error_norm:.3e}"
            )

        # Check convergence
        if error_norm < tol_converge:
            break

        # Body Jacobian
        J_b = body_jacobian(
            B_list,
            theta
        )

        # Linear velocity rows
        # Your convention is:
        #   J_b[0:3, :] = angular
        #   J_b[3:6, :] = linear
        J_v = J_b[3:6, :]

        # Jacobian transpose update
        dq = J_v.T @ K @ error_vec

        # Update joint angles
        theta = theta + dq

        # Apply joint limits
        if q_min is not None and q_max is not None:
            theta = np.clip(
                theta,
                q_min,
                q_max
            )

        theta_history.append(theta.copy())

    theta_history = np.asarray(theta_history)

    return theta, theta_history



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
    K=None,
    print_iterations=False,
):
    """
    Numerical inverse kinematics for full pose using
    the body Jacobian transpose method.
    """

    theta = np.asarray(theta_init, dtype=float).reshape(-1)
    n = len(theta)

    if q_min is None:
        q_min = np.deg2rad(
            np.array([-105, -95, -90, -90, -90, -90], dtype=float)
        )

    if q_max is None:
        q_max = np.deg2rad(
            np.array([105, 105, 95, 90, 90, 90], dtype=float)
        )

    q_min = np.asarray(q_min, dtype=float).reshape(-1)
    q_max = np.asarray(q_max, dtype=float).reshape(-1)

    if K is None:
        K = np.eye(6)

    K = np.asarray(K, dtype=float).reshape(6, 6)

    theta_history = []
    norm_w_b_hist = []
    norm_v_b_hist = []

    for i in range(max_iters + 1):

        # Current end-effector pose in space/base frame
        T_sb = body_product_of_exponentials(
            M_ee,
            B_list,
            theta
        )

        # Body-frame pose error
        # T_bd = inv(T_sb) @ T_sd
        T_bs = inv_SE3(T_sb)
        T_bd = T_bs @ T_sd

        # Your se3.py returns:
        #   T = exp([S] * theta_err)
        # So the body twist error vector is:
        #   Vb = S * theta_err
        S_err, theta_err = log_screw_axis(T_bd)
        Vb = S_err * theta_err

        w_b = Vb[0:3]
        v_b = Vb[3:6]

        norm_w = np.linalg.norm(w_b)
        norm_v = np.linalg.norm(v_b)

        theta_history.append(theta.copy())
        norm_w_b_hist.append(norm_w)
        norm_v_b_hist.append(norm_v)

        if print_iterations:
            theta_deg = np.rad2deg(theta)

            joint_text = ", ".join(
                [
                    f"theta{j + 1}={theta_deg[j]:.2f}deg"
                    for j in range(n)
                ]
            )

            print(
                f"Iteration {i}: "
                f"({joint_text}), "
                f"(x,y,z)=({T_sb[0, 3]:.3f}, {T_sb[1, 3]:.3f}, {T_sb[2, 3]:.3f}), "
                f"||w_b||={norm_w:.3e}, "
                f"||v_b||={norm_v:.3e}"
            )

        if norm_w < tol_w and norm_v < tol_v:
            break

        J_b = body_jacobian(
            B_list,
            theta
        )

        dq = J_b.T @ K @ Vb

        theta = theta + dq

        theta = np.clip(
            theta,
            q_min,
            q_max
        )

    theta_history = np.asarray(theta_history)
    norm_w_b_hist = np.asarray(norm_w_b_hist)
    norm_v_b_hist = np.asarray(norm_v_b_hist)

    return theta, theta_history, norm_w_b_hist, norm_v_b_hist