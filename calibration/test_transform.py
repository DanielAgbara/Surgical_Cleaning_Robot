import cv2
import numpy as np
import pyzed.sl as sl
from pathlib import Path


ROOT = Path("/home/agbara-admin/Documents/Cleaning_Robot")
T_PATH = ROOT / "data" / "eye_to_hand" / "T_base_to_camera.npy"

clicked_pixel = None


def mouse_callback(event, x, y, flags, param):
    global clicked_pixel

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pixel = (x, y)


def transform_point(T, p):
    p_h = np.array([p[0], p[1], p[2], 1.0], dtype=float)
    p_out = T @ p_h
    return p_out[:3]


def is_reachable(p_base):
    x, y, z = p_base

    r = np.sqrt(x**2 + y**2)

    if z < 0.02:
        return False

    if z > 0.45:
        return False

    if r < 0.05:
        return False

    if r > 0.40:
        return False

    return True


def get_average_camera_point(point_cloud, u, v, window=5):
    """
    Average valid ZED 3D points around clicked pixel.

    This is more stable than using one pixel because stereo depth
    can be noisy or invalid at edges.
    """

    points = []

    half = window // 2

    for yy in range(v - half, v + half + 1):
        for xx in range(u - half, u + half + 1):
            err, point = point_cloud.get_value(xx, yy)

            if err != sl.ERROR_CODE.SUCCESS:
                continue

            X, Y, Z = point[:3]

            if np.isfinite(X) and np.isfinite(Y) and np.isfinite(Z):
                points.append([X, Y, Z])

    if len(points) == 0:
        return None

    return np.mean(np.array(points), axis=0)


def main():
    global clicked_pixel

    T_base_to_camera = np.load(T_PATH)

    print("\nT_base_to_camera:")
    print(T_base_to_camera)

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL

    # Must match the coordinate system used during calibration.
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")

    runtime = sl.RuntimeParameters()

    image_zed = sl.Mat()
    point_cloud = sl.Mat()

    cv2.namedWindow("Click Depth Test")
    cv2.setMouseCallback("Click Depth Test", mouse_callback)

    last_text = "Click a pixel"

    try:
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image_zed, sl.VIEW.LEFT)

            # Use XYZRGBA because it is commonly supported in ZED examples.
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            frame = image_zed.get_data()

            if frame.shape[2] == 4:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                frame_bgr = frame.copy()

            if clicked_pixel is not None:
                u, v = clicked_pixel

                p_camera = get_average_camera_point(
                    point_cloud,
                    u,
                    v,
                    window=7
                )

                if p_camera is None:
                    last_text = "Invalid depth around clicked pixel"
                    print(last_text)

                else:
                    p_base = transform_point(T_base_to_camera, p_camera)
                    reachable = is_reachable(p_base)

                    last_text = (
                        f"cam=({p_camera[0]:.3f},{p_camera[1]:.3f},{p_camera[2]:.3f}) m | "
                        f"base=({p_base[0]:.3f},{p_base[1]:.3f},{p_base[2]:.3f}) m | "
                        f"reachable={reachable}"
                    )

                    print("\nClicked pixel:", clicked_pixel)
                    print("p_camera:", p_camera)
                    print("p_base:", p_base)
                    print("reachable:", reachable)

                clicked_pixel = None

            cv2.putText(
                frame_bgr,
                last_text,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Click Depth Test", frame_bgr)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()