import numpy as np
import warnings


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