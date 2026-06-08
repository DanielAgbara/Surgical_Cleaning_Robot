import pyzed.sl as sl
import cv2 as cv
import numpy as np


# =====================================================
# ZED BODY_18 ARM TRACKING UTILITIES
# =====================================================
#
# This file provides helper functions for:
#
#   1. Setting up ZED body tracking
#   2. Selecting only one detected human
#   3. Extracting left or right arm points
#   4. Drawing shoulder, elbow, and wrist
#   5. Estimating arm length from one frame
#   6. Averaging arm length over many frames
#   7. Testing everything live
#
# The main goal is to track:
#
#   shoulder -> elbow -> wrist
#
# and eventually use the wrist displacement to control
# the SO-ARM101 robot in its own robot frame.
#
# =====================================================


# =====================================================
# BODY_18 Configuration
# =====================================================

BODY_FORMAT = sl.BODY_FORMAT.BODY_18

# Right arm BODY_18 indices
RIGHT_SHOULDER = 2
RIGHT_ELBOW = 3
RIGHT_WRIST = 4

# Left arm BODY_18 indices
LEFT_SHOULDER = 5
LEFT_ELBOW = 6
LEFT_WRIST = 7


def setup_body_tracking(zed):
    """
    Enable BODY_18 body tracking on an already-opened ZED camera.

    Parameters
    ----------
    zed : sl.Camera
        Already opened ZED camera object.

    Returns
    -------
    body_runtime : sl.BodyTrackingRuntimeParameters
        Runtime parameters used by zed.retrieve_bodies(...).
    """

    # -------------------------------------------------
    # Create body tracking parameter object
    # -------------------------------------------------

    body_params = sl.BodyTrackingParameters()

    # -------------------------------------------------
    # Use the accurate human body tracking model
    # -------------------------------------------------

    body_params.detection_model = (
        sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    )

    # -------------------------------------------------
    # Enable tracking so the ZED can keep person IDs
    # consistent between frames
    # -------------------------------------------------

    body_params.enable_tracking = True

    # -------------------------------------------------
    # Enable body fitting for smoother skeleton points
    # -------------------------------------------------

    body_params.enable_body_fitting = True

    # -------------------------------------------------
    # Use the BODY_18 skeleton format
    # -------------------------------------------------

    body_params.body_format = BODY_FORMAT

    # -------------------------------------------------
    # Positional tracking is required for ZED body tracking
    # -------------------------------------------------

    positional_params = sl.PositionalTrackingParameters()

    err = zed.enable_positional_tracking(positional_params)

    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(
            f"Failed to enable positional tracking: {err}"
        )

    # -------------------------------------------------
    # Enable body tracking
    # -------------------------------------------------

    err = zed.enable_body_tracking(body_params)

    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(
            f"Failed to enable body tracking: {err}"
        )

    # -------------------------------------------------
    # Create runtime body tracking parameters
    # -------------------------------------------------

    body_runtime = sl.BodyTrackingRuntimeParameters()

    # -------------------------------------------------
    # Ignore low-confidence body detections
    # -------------------------------------------------

    body_runtime.detection_confidence_threshold = 40

    print("Body tracking enabled.")

    return body_runtime


def get_single_body(bodies, mode="closest"):
    """
    Select one person from all detected people.

    Parameters
    ----------
    bodies : sl.Bodies
        Body container returned by zed.retrieve_bodies(...).

    mode : str
        Selection method.

        "closest":
            Select the detected person closest to the camera.

        "first":
            Select the first detected person.

    Returns
    -------
    body : sl.BodyData or None
        One selected body.

        Returns None if no bodies are detected.
    """

    # -------------------------------------------------
    # If no human is detected, return None
    # -------------------------------------------------

    if len(bodies.body_list) == 0:
        print("No detected bodies!")
        return None

    # -------------------------------------------------
    # Option 1: use the first detected person
    # -------------------------------------------------

    if mode == "first":
        return bodies.body_list[0]

    # -------------------------------------------------
    # Option 2: use the closest detected person
    # -------------------------------------------------
    #
    # body.position[2] is the depth value.
    #
    # Smaller positive z value usually means the person
    # is closer to the camera.
    #
    # -------------------------------------------------

    if mode == "closest":
        return min(
            bodies.body_list,
            key=lambda b: float(b.position[2])
            if b.position[2] > 0
            else float("inf")
        )

    # -------------------------------------------------
    # Reject invalid selection modes
    # -------------------------------------------------

    raise ValueError("mode must be 'closest' or 'first'")


def get_arm_indices(arm="right"):
    """
    Return BODY_18 indices for the selected arm.

    Parameters
    ----------
    arm : str
        "right" or "left".

    Returns
    -------
    shoulder_idx, elbow_idx, wrist_idx : tuple
        BODY_18 joint indices for the selected arm.
    """

    # -------------------------------------------------
    # Normalize user input
    # -------------------------------------------------

    arm = arm.lower()

    # -------------------------------------------------
    # Return right arm indices
    # -------------------------------------------------

    if arm == "right":
        return RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST

    # -------------------------------------------------
    # Return left arm indices
    # -------------------------------------------------

    if arm == "left":
        return LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST

    # -------------------------------------------------
    # Reject invalid arm names
    # -------------------------------------------------

    raise ValueError("arm must be 'right' or 'left'")


def get_arm_points(body, arm="right"):
    """
    Extract shoulder, elbow, and wrist points for one arm.

    Parameters
    ----------
    body : sl.BodyData
        One detected person from bodies.body_list.

    arm : str
        Arm to extract.
        Options:
            "right"
            "left"

    Returns
    -------
    arm_data : dict or None
        Dictionary containing 2D and 3D points.

        Returns None if the selected arm points are invalid.
    """

    # -------------------------------------------------
    # Get correct BODY_18 indices
    # -------------------------------------------------

    shoulder_idx, elbow_idx, wrist_idx = get_arm_indices(arm)

    # -------------------------------------------------
    # Get 2D pixel keypoints
    # -------------------------------------------------
    #
    # These are used for drawing on the OpenCV image.
    #
    # -------------------------------------------------

    keypoints_2d = body.keypoint_2d

    # -------------------------------------------------
    # Get 3D keypoints
    # -------------------------------------------------
    #
    # These are used for distance calculations and
    # later robot control.
    #
    # -------------------------------------------------

    keypoints_3d = body.keypoint

    # -------------------------------------------------
    # Check that the selected indices exist
    # -------------------------------------------------

    max_idx = max(shoulder_idx, elbow_idx, wrist_idx)

    if len(keypoints_2d) <= max_idx:
        return None

    if len(keypoints_3d) <= max_idx:
        return None

    # -------------------------------------------------
    # Extract 2D points
    # -------------------------------------------------

    shoulder_2d = np.array(keypoints_2d[shoulder_idx], dtype=float)
    elbow_2d = np.array(keypoints_2d[elbow_idx], dtype=float)
    wrist_2d = np.array(keypoints_2d[wrist_idx], dtype=float)

    # -------------------------------------------------
    # Extract 3D points
    # -------------------------------------------------

    shoulder_3d = np.array(keypoints_3d[shoulder_idx], dtype=float)
    elbow_3d = np.array(keypoints_3d[elbow_idx], dtype=float)
    wrist_3d = np.array(keypoints_3d[wrist_idx], dtype=float)

    # -------------------------------------------------
    # Check 2D points
    # -------------------------------------------------
    #
    # Invalid 2D points are often [0, 0] or negative.
    #
    # -------------------------------------------------

    for point in [shoulder_2d, elbow_2d, wrist_2d]:
        if point[0] <= 0 or point[1] <= 0:
            return None

    # -------------------------------------------------
    # Check 3D points
    # -------------------------------------------------
    #
    # Invalid 3D points can contain nan or inf.
    #
    # -------------------------------------------------

    for point in [shoulder_3d, elbow_3d, wrist_3d]:
        if not np.isfinite(point).all():
            return None

    # -------------------------------------------------
    # Package arm data
    # -------------------------------------------------

    arm_data = {
        "arm": arm,

        "shoulder_2d": shoulder_2d,
        "elbow_2d": elbow_2d,
        "wrist_2d": wrist_2d,

        "shoulder_3d": shoulder_3d,
        "elbow_3d": elbow_3d,
        "wrist_3d": wrist_3d,
    }

    return arm_data


def draw_arm_points_and_lines(image, arm_data):
    """
    Draw shoulder, elbow, and wrist on an OpenCV image.

    Parameters
    ----------
    image : np.ndarray
        OpenCV image.

    arm_data : dict or None
        Output from get_arm_points(...).

    Returns
    -------
    image : np.ndarray
        Image with arm overlay.
    """

    # -------------------------------------------------
    # If arm data is invalid, return image unchanged
    # -------------------------------------------------

    if arm_data is None:
        return image

    # -------------------------------------------------
    # Extract 2D points
    # -------------------------------------------------

    shoulder = arm_data["shoulder_2d"]
    elbow = arm_data["elbow_2d"]
    wrist = arm_data["wrist_2d"]

    # -------------------------------------------------
    # Convert float pixel coordinates to integer pixels
    # -------------------------------------------------

    shoulder = (int(shoulder[0]), int(shoulder[1]))
    elbow = (int(elbow[0]), int(elbow[1]))
    wrist = (int(wrist[0]), int(wrist[1]))

    # -------------------------------------------------
    # Draw arm links first
    # -------------------------------------------------

    cv.line(image, shoulder, elbow, (0, 255, 255), 3)
    cv.line(image, elbow, wrist, (0, 255, 255), 3)

    # -------------------------------------------------
    # Draw arm joints
    # -------------------------------------------------

    cv.circle(image, shoulder, 7, (0, 255, 0), -1)
    cv.circle(image, elbow, 7, (0, 255, 0), -1)
    cv.circle(image, wrist, 7, (0, 255, 0), -1)

    # -------------------------------------------------
    # Label joints
    # -------------------------------------------------

    cv.putText(
        image,
        "Shoulder",
        (shoulder[0] + 10, shoulder[1] - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

    cv.putText(
        image,
        "Elbow",
        (elbow[0] + 10, elbow[1] - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

    cv.putText(
        image,
        "Wrist",
        (wrist[0] + 10, wrist[1] - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

    return image


def estimate_arm_length(arm_data):
    """
    Estimate arm segment lengths from one valid arm frame.

    Parameters
    ----------
    arm_data : dict
        Output from get_arm_points(...).

    Returns
    -------
    lengths : dict
        Arm lengths in the same units as the ZED camera.

        If your ZED is set to sl.UNIT.MILLIMETER,
        the output is in millimeters.
    """

    # -------------------------------------------------
    # Extract 3D points
    # -------------------------------------------------

    shoulder = arm_data["shoulder_3d"]
    elbow = arm_data["elbow_3d"]
    wrist = arm_data["wrist_3d"]

    # -------------------------------------------------
    # Compute shoulder -> elbow distance
    # -------------------------------------------------

    shoulder_to_elbow = np.linalg.norm(elbow - shoulder)

    # -------------------------------------------------
    # Compute elbow -> wrist distance
    # -------------------------------------------------

    elbow_to_wrist = np.linalg.norm(wrist - elbow)

    # -------------------------------------------------
    # Compute direct shoulder -> wrist distance
    # -------------------------------------------------
    #
    # This is not always equal to shoulder_to_elbow
    # plus elbow_to_wrist unless the arm is straight.
    #
    # -------------------------------------------------

    shoulder_to_wrist = np.linalg.norm(wrist - shoulder)

    # -------------------------------------------------
    # Store results
    # -------------------------------------------------

    lengths = {
        "shoulder_to_elbow": float(shoulder_to_elbow),
        "elbow_to_wrist": float(elbow_to_wrist),
        "shoulder_to_wrist": float(shoulder_to_wrist),
    }

    return lengths


def average_arm_data_over_frames(
    zed,
    body_runtime,
    arm="right",
    num_frames=30,
    body_selection_mode="closest"
):
    """
    Collect and average arm data over multiple valid frames.

    This function performs NO visualization.

    Returns
    -------
    avg_arm_data : dict or None
    """

    # -------------------------------------------------
    # ZED containers
    # -------------------------------------------------

    bodies = sl.Bodies()
    runtime = sl.RuntimeParameters()

    # -------------------------------------------------
    # Storage
    # -------------------------------------------------

    shoulder_2d_samples = []
    elbow_2d_samples = []
    wrist_2d_samples = []

    shoulder_3d_samples = []
    elbow_3d_samples = []
    wrist_3d_samples = []

    # -------------------------------------------------
    # Collect valid samples
    # -------------------------------------------------

    while len(wrist_3d_samples) < num_frames:

        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_bodies(
            bodies,
            body_runtime
        )

        body = get_single_body(
            bodies,
            mode=body_selection_mode
        )

        if body is None:
            continue

        arm_data = get_arm_points(
            body,
            arm=arm
        )

        if arm_data is None:
            continue

        shoulder_2d_samples.append(
            arm_data["shoulder_2d"]
        )

        elbow_2d_samples.append(
            arm_data["elbow_2d"]
        )

        wrist_2d_samples.append(
            arm_data["wrist_2d"]
        )

        shoulder_3d_samples.append(
            arm_data["shoulder_3d"]
        )

        elbow_3d_samples.append(
            arm_data["elbow_3d"]
        )

        wrist_3d_samples.append(
            arm_data["wrist_3d"]
        )

    # -------------------------------------------------
    # Compute averages
    # -------------------------------------------------

    avg_arm_data = {
        "arm": arm,

        "shoulder_2d": np.mean(
            shoulder_2d_samples,
            axis=0
        ),

        "elbow_2d": np.mean(
            elbow_2d_samples,
            axis=0
        ),

        "wrist_2d": np.mean(
            wrist_2d_samples,
            axis=0
        ),

        "shoulder_3d": np.mean(
            shoulder_3d_samples,
            axis=0
        ),

        "elbow_3d": np.mean(
            elbow_3d_samples,
            axis=0
        ),

        "wrist_3d": np.mean(
            wrist_3d_samples,
            axis=0
        ),
    }

    return avg_arm_data

def get_reference_position(
    zed,
    body_runtime,
    arm="right",
    num_frames=30,
    body_selection_mode="closest"
):
    """
    Get the reference arm position.

    The reference position is collected while the human
    holds their arm in the starting pose.

    The function averages 30 valid frames by default.

    It also computes the center point between the
    shoulder and wrist.

    Parameters
    ----------
    zed : sl.Camera
        Open ZED camera.

    body_runtime : sl.BodyTrackingRuntimeParameters
        Body tracking runtime settings.

    arm : str
        Arm to use: "right" or "left".

    num_frames : int
        Number of valid frames to average.

    body_selection_mode : str
        How to select one human.

    Returns
    -------
    reference_data : dict or None
        Reference data containing:

            shoulder_reference_3d
            elbow_reference_3d
            wrist_reference_3d
            center_reference_3d
            arm_length_data

        Returns None if calibration is canceled.
    """

    # -------------------------------------------------
    # Average arm points over multiple frames
    # -------------------------------------------------

    avg_arm_data = average_arm_data_over_frames(
        zed=zed,
        body_runtime=body_runtime,
        arm=arm,
        num_frames=num_frames,
        body_selection_mode=body_selection_mode
    )

    # -------------------------------------------------
    # Stop if no valid average was produced
    # -------------------------------------------------

    if avg_arm_data is None:
        return None

    # -------------------------------------------------
    # Extract averaged 3D points
    # -------------------------------------------------

    shoulder_3d = avg_arm_data["shoulder_3d"]
    elbow_3d = avg_arm_data["elbow_3d"]
    wrist_3d = avg_arm_data["wrist_3d"]

    # -------------------------------------------------
    # Compute arm length data from averaged pose
    # -------------------------------------------------

    arm_length_data = estimate_arm_length(avg_arm_data)

    # -------------------------------------------------
    # Package reference data
    # -------------------------------------------------

    reference_data = {
        "arm": arm,

        "shoulder_reference_3d": shoulder_3d,
        "elbow_reference_3d": elbow_3d,
        "wrist_reference_3d": wrist_3d,

        "shoulder_reference_2d": avg_arm_data["shoulder_2d"],
        "elbow_reference_2d": avg_arm_data["elbow_2d"],
        "wrist_reference_2d": avg_arm_data["wrist_2d"],

        "arm_length_data": arm_length_data,

        "num_frames": num_frames,
    }

    return reference_data


def test_reference_position(
    zed,
    body_runtime,
    arm="right",
    num_frames=30,
    body_selection_mode="closest"
):
    """
    One-window reference position test.

    Behavior:
        - Shows live arm tracking.
        - Displays instructions in the same window.
        - Press ENTER to collect reference position.
        - Collects num_frames valid samples in the background.
        - Prints reference data on the same window.
        - Window only closes when q is pressed or window is closed.
    """

    image = sl.Mat()
    bodies = sl.Bodies()
    runtime = sl.RuntimeParameters()

    window_name = "Reference Position Setup"

    reference_data = None
    collecting = False
    collection_done = False

    # Storage for reference samples
    shoulder_2d_samples = []
    elbow_2d_samples = []
    wrist_2d_samples = []

    shoulder_3d_samples = []
    elbow_3d_samples = []
    wrist_3d_samples = []

    print("Hold arm straight.")
    print("Press ENTER to collect reference.")
    print("Press q to quit.")

    while True:

        # -------------------------------------------------
        # Grab frame
        # -------------------------------------------------

        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        # -------------------------------------------------
        # Retrieve image
        # -------------------------------------------------

        zed.retrieve_image(image, sl.VIEW.LEFT)

        frame = image.get_data()

        if frame.shape[2] == 4:
            frame = cv.cvtColor(
                frame,
                cv.COLOR_BGRA2BGR
            )

        # -------------------------------------------------
        # Retrieve body tracking result
        # -------------------------------------------------

        zed.retrieve_bodies(
            bodies,
            body_runtime
        )

        body = get_single_body(
            bodies,
            mode=body_selection_mode
        )

        arm_data = None

        if body is not None:
            arm_data = get_arm_points(
                body,
                arm=arm
            )

        # -------------------------------------------------
        # Draw live arm tracking
        # -------------------------------------------------

        if arm_data is not None:
            frame = draw_arm_points_and_lines(
                frame,
                arm_data
            )

        # -------------------------------------------------
        # If collecting, save valid samples
        # -------------------------------------------------

        if collecting and arm_data is not None:

            shoulder_2d_samples.append(
                arm_data["shoulder_2d"]
            )

            elbow_2d_samples.append(
                arm_data["elbow_2d"]
            )

            wrist_2d_samples.append(
                arm_data["wrist_2d"]
            )

            shoulder_3d_samples.append(
                arm_data["shoulder_3d"]
            )

            elbow_3d_samples.append(
                arm_data["elbow_3d"]
            )

            wrist_3d_samples.append(
                arm_data["wrist_3d"]
            )

            # Stop collecting when enough valid frames are saved
            if len(wrist_3d_samples) >= num_frames:

                collecting = False
                collection_done = True

                avg_arm_data = {
                    "arm": arm,

                    "shoulder_2d": np.mean(
                        shoulder_2d_samples,
                        axis=0
                    ),

                    "elbow_2d": np.mean(
                        elbow_2d_samples,
                        axis=0
                    ),

                    "wrist_2d": np.mean(
                        wrist_2d_samples,
                        axis=0
                    ),

                    "shoulder_3d": np.mean(
                        shoulder_3d_samples,
                        axis=0
                    ),

                    "elbow_3d": np.mean(
                        elbow_3d_samples,
                        axis=0
                    ),

                    "wrist_3d": np.mean(
                        wrist_3d_samples,
                        axis=0
                    ),
                }

                arm_length_data = estimate_arm_length(
                    avg_arm_data
                )

                reference_data = {
                    "arm": arm,

                    "shoulder_reference_3d": avg_arm_data["shoulder_3d"],
                    "elbow_reference_3d": avg_arm_data["elbow_3d"],
                    "wrist_reference_3d": avg_arm_data["wrist_3d"],

                    "shoulder_reference_2d": avg_arm_data["shoulder_2d"],
                    "elbow_reference_2d": avg_arm_data["elbow_2d"],
                    "wrist_reference_2d": avg_arm_data["wrist_2d"],

                    "arm_length_data": arm_length_data,

                    "num_frames": num_frames,
                }

                print("\nReference calibration complete.")
                print(
                    "Shoulder reference:",
                    reference_data["shoulder_reference_3d"]
                )
                print(
                    "Elbow reference:",
                    reference_data["elbow_reference_3d"]
                )
                print(
                    "Wrist reference:",
                    reference_data["wrist_reference_3d"]
                )
                print(
                    "Arm length data:",
                    reference_data["arm_length_data"]
                )

        # -------------------------------------------------
        # Draw instructions/status text
        # -------------------------------------------------

        if not collecting and not collection_done:

            cv.putText(
                frame,
                "Hold arm straight",
                (20, 40),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv.putText(
                frame,
                "Press ENTER to collect reference",
                (20, 75),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv.putText(
                frame,
                "Press q to quit",
                (20, 110),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        elif collecting:

            cv.putText(
                frame,
                f"Collecting reference: {len(wrist_3d_samples)}/{num_frames}",
                (20, 40),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv.putText(
                frame,
                "Keep arm still and straight",
                (20, 75),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        elif collection_done and reference_data is not None:

            wrist_ref = reference_data["wrist_reference_3d"]
            shoulder_ref = reference_data["shoulder_reference_3d"]
            lengths = reference_data["arm_length_data"]

            cv.putText(
                frame,
                "Reference collected",
                (20, 40),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv.putText(
                frame,
                f"Shoulder: x={shoulder_ref[0]:.3f}, y={shoulder_ref[1]:.3f}, z={shoulder_ref[2]:.3f}",
                (20, 75),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv.putText(
                frame,
                f"Wrist: x={wrist_ref[0]:.3f}, y={wrist_ref[1]:.3f}, z={wrist_ref[2]:.3f}",
                (20, 105),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv.putText(
                frame,
                f"Upper arm: {lengths['shoulder_to_elbow']:.3f}",
                (20, 135),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv.putText(
                frame,
                f"Forearm: {lengths['elbow_to_wrist']:.3f}",
                (20, 165),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv.putText(
                frame,
                "Press q to quit",
                (20, 200),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # -------------------------------------------------
        # Show one window only
        # -------------------------------------------------

        cv.imshow(
            window_name,
            frame
        )

        key = cv.waitKey(1) & 0xFF

        # ENTER starts collection
        if key in [10, 13] and not collecting:
            collecting = True
            collection_done = False
            reference_data = None

            shoulder_2d_samples.clear()
            elbow_2d_samples.clear()
            wrist_2d_samples.clear()

            shoulder_3d_samples.clear()
            elbow_3d_samples.clear()
            wrist_3d_samples.clear()

            print(f"Collecting {num_frames} valid reference frames...")

        # q exits
        if key == ord("q"):
            break

        # Closing the window exits
        if cv.getWindowProperty(
            window_name,
            cv.WND_PROP_VISIBLE
        ) < 1:
            break

    cv.destroyWindow(window_name)

    return reference_data

if __name__ == "__main__":

    # -------------------------------------------------
    # Create ZED camera object
    # -------------------------------------------------

    zed = sl.Camera()

    # -------------------------------------------------
    # Set camera initialization parameters
    # -------------------------------------------------

    init_params = sl.InitParameters()

    # Use HD1080 resolution
    init_params.camera_resolution = sl.RESOLUTION.HD1080

    # Use 30 FPS
    init_params.camera_fps = 30

    # Return 3D positions in meters
    init_params.coordinate_units = sl.UNIT.METER

    # Use image coordinate system
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

    # -------------------------------------------------
    # Open camera
    # -------------------------------------------------

    err = zed.open(init_params)

    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera:", err)
        exit(1)

    print("Camera opened.")

    try:
        # -------------------------------------------------
        # Enable body tracking
        # -------------------------------------------------

        body_runtime = setup_body_tracking(zed)

        # -------------------------------------------------
        # Run reference position test
        # -------------------------------------------------

        reference_data = test_reference_position(
            zed=zed,
            body_runtime=body_runtime,
            arm="right",
            num_frames=30
        )

        # -------------------------------------------------
        # Print final reference data
        # -------------------------------------------------

        if reference_data is not None:

            print("\nFinal Reference Data")

            print(
                "Wrist reference:",
                reference_data["wrist_reference_3d"]
            )

            print(
                "Center reference:",
                reference_data["center_reference_3d"]
            )

            print(
                "Arm length data:",
                reference_data["arm_length_data"]
            )

    finally:
        # -------------------------------------------------
        # Always close the ZED camera
        # -------------------------------------------------

        zed.close()

        # -------------------------------------------------
        # Always close OpenCV windows
        # -------------------------------------------------

        cv.destroyAllWindows()

        print("Program closed.")