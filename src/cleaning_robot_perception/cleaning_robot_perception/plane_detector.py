#!/usr/bin/env python3

"""
ROS 2 tray plane detector.

Subscribes
----------
/perception/tray/mask
    sensor_msgs/msg/Image
    Encoding: mono8

/camera/depth
    sensor_msgs/msg/Image
    Encoding: 16UC1
    Units: millimeters

/camera/left/camera_info
    sensor_msgs/msg/CameraInfo

Publishes
---------
/perception/tray/centroid
    geometry_msgs/msg/PointStamped
    Position in millimeters.

/perception/tray/normal
    geometry_msgs/msg/Vector3Stamped
    Unit surface normal.

/perception/tray/plane
    std_msgs/msg/Float64MultiArray

    [a, b, c, d]

    Plane equation:

        a*x + b*y + c*z + d = 0

    x, y, z and d use millimeters.

The normal is oriented from the tray toward the camera.

No cv_bridge is used.
"""

import sys

import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Vector3Stamped

from std_msgs.msg import Float64MultiArray

from message_filters import (
    Subscriber,
    TimeSynchronizer,
)


# ============================================================
# Plane Detector Node
# ============================================================

class PlaneDetectorNode(Node):

    def __init__(self):

        super().__init__("plane_detector")

        # ====================================================
        # Parameters
        # ====================================================

        self.declare_parameter(
            "min_depth_mm",
            100.0,
        )

        self.declare_parameter(
            "max_depth_mm",
            3000.0,
        )

        self.declare_parameter(
            "plane_distance_threshold_mm",
            15.0,
        )

        self.declare_parameter(
            "ransac_iterations",
            1000,
        )

        self.declare_parameter(
            "min_3d_points",
            100,
        )

        self.declare_parameter(
            "max_ransac_points",
            60000,
        )

        self.min_depth_mm = float(
            self.get_parameter(
                "min_depth_mm"
            ).value
        )

        self.max_depth_mm = float(
            self.get_parameter(
                "max_depth_mm"
            ).value
        )

        self.distance_threshold_mm = float(
            self.get_parameter(
                "plane_distance_threshold_mm"
            ).value
        )

        self.ransac_iterations = int(
            self.get_parameter(
                "ransac_iterations"
            ).value
        )

        self.min_3d_points = int(
            self.get_parameter(
                "min_3d_points"
            ).value
        )

        self.max_ransac_points = int(
            self.get_parameter(
                "max_ransac_points"
            ).value
        )

        # ====================================================
        # Camera Intrinsics
        # ====================================================

        self.camera_matrix = None

        self.camera_info_subscription = (
            self.create_subscription(
                CameraInfo,
                "/camera/left/camera_info",
                self.camera_info_callback,
                qos_profile_sensor_data,
            )
        )

        # ====================================================
        # Synchronize Mask + Depth
        # ====================================================
        #
        # zed_camera gives RGB and depth the same timestamp.
        #
        # object_detector preserves the RGB timestamp when
        # publishing the tray mask.
        #
        # Therefore mask and depth can be paired exactly.
        # ====================================================

        self.mask_subscription = Subscriber(
            self,
            Image,
            "/perception/tray/mask",
            qos_profile=qos_profile_sensor_data,
        )

        self.depth_subscription = Subscriber(
            self,
            Image,
            "/camera/depth",
            qos_profile=qos_profile_sensor_data,
        )

        self.synchronizer = TimeSynchronizer(
            [
                self.mask_subscription,
                self.depth_subscription,
            ],
            queue_size=30,
        )

        self.synchronizer.registerCallback(
            self.synced_callback
        )

        # ====================================================
        # Publishers
        # ====================================================

        self.centroid_publisher = (
            self.create_publisher(
                PointStamped,
                "/perception/tray/centroid",
                10,
            )
        )

        self.normal_publisher = (
            self.create_publisher(
                Vector3Stamped,
                "/perception/tray/normal",
                10,
            )
        )

        self.plane_publisher = (
            self.create_publisher(
                Float64MultiArray,
                "/perception/tray/plane",
                10,
            )
        )

        # ====================================================
        # Startup Information
        # ====================================================

        self.get_logger().info(
            "Plane detector started"
        )

        self.get_logger().info(
            "Depth units: millimeters"
        )

        self.get_logger().info(
            "Waiting for tray mask, depth, "
            "and CameraInfo..."
        )

        self.get_logger().info(
            f"RANSAC threshold: "
            f"{self.distance_threshold_mm:.1f} mm"
        )

    # ========================================================
    # CameraInfo Callback
    # ========================================================

    def camera_info_callback(
        self,
        msg: CameraInfo,
    ):
        """
        Store the rectified camera intrinsic matrix:

            [ fx  0 cx ]
        K = [  0 fy cy ]
            [  0  0  1 ]
        """

        self.camera_matrix = np.array(
            msg.k,
            dtype=np.float64,
        ).reshape(
            3,
            3,
        )

    # ========================================================
    # ROS mono8 -> NumPy
    # ========================================================

    def ros_mask_to_numpy(
        self,
        msg: Image,
    ):
        """
        Convert a mono8 ROS image directly into NumPy.
        """

        if msg.encoding != "mono8":

            raise ValueError(
                "Expected tray mask encoding "
                f"'mono8', received '{msg.encoding}'"
            )

        height = int(
            msg.height
        )

        width = int(
            msg.width
        )

        row_bytes = width

        if msg.step < row_bytes:

            raise ValueError(
                "Mask step is smaller than expected"
            )

        raw = np.frombuffer(
            msg.data,
            dtype=np.uint8,
        )

        required_bytes = (
            height * msg.step
        )

        if raw.size < required_bytes:

            raise ValueError(
                "Mask data buffer is smaller "
                "than expected"
            )

        # Account for possible row padding.
        rows = raw[
            :required_bytes
        ].reshape(
            height,
            msg.step,
        )

        mask = rows[
            :,
            :row_bytes,
        ].reshape(
            height,
            width,
        )

        return np.ascontiguousarray(
            mask
        )

    # ========================================================
    # ROS 16UC1 -> NumPy
    # ========================================================

    def ros_depth_to_numpy(
        self,
        msg: Image,
    ):
        """
        Convert a 16UC1 ROS depth image directly into NumPy.

        Returned values are uint16 millimeters.
        """

        if msg.encoding != "16UC1":

            raise ValueError(
                "Expected depth encoding "
                f"'16UC1', received '{msg.encoding}'"
            )

        height = int(
            msg.height
        )

        width = int(
            msg.width
        )

        bytes_per_pixel = 2

        expected_row_bytes = (
            width * bytes_per_pixel
        )

        if msg.step < expected_row_bytes:

            raise ValueError(
                "Depth image step is smaller "
                "than expected"
            )

        # ----------------------------------------------------
        # Determine uint16 byte order
        # ----------------------------------------------------

        if msg.is_bigendian:

            dtype = np.dtype(
                ">u2"
            )

        else:

            dtype = np.dtype(
                "<u2"
            )

        raw_bytes = np.frombuffer(
            msg.data,
            dtype=np.uint8,
        )

        required_bytes = (
            height * msg.step
        )

        if raw_bytes.size < required_bytes:

            raise ValueError(
                "Depth data buffer is smaller "
                "than expected"
            )

        # ----------------------------------------------------
        # Account for possible row padding
        # ----------------------------------------------------

        rows = raw_bytes[
            :required_bytes
        ].reshape(
            height,
            msg.step,
        )

        pixel_bytes = rows[
            :,
            :expected_row_bytes,
        ]

        depth = np.frombuffer(
            np.ascontiguousarray(
                pixel_bytes
            ).tobytes(),
            dtype=dtype,
        ).reshape(
            height,
            width,
        )

        # Convert to native byte order if necessary.
        if (
            msg.is_bigendian
            != (sys.byteorder == "big")
        ):

            depth = depth.byteswap().newbyteorder()

        return np.ascontiguousarray(
            depth,
            dtype=np.uint16,
        )

    # ========================================================
    # Synchronized Callback
    # ========================================================

    def synced_callback(
        self,
        mask_msg: Image,
        depth_msg: Image,
    ):
        """
        Process a tray mask and the depth frame corresponding
        to the exact RGB frame used for detection.
        """

        if self.camera_matrix is None:

            self.get_logger().warning(
                "CameraInfo has not been received yet"
            )

            return

        # ====================================================
        # ROS Images -> NumPy
        # ====================================================

        try:

            mask = self.ros_mask_to_numpy(
                mask_msg
            )

            depth_mm = self.ros_depth_to_numpy(
                depth_msg
            )

        except Exception as error:

            self.get_logger().error(
                "Could not convert ROS image: "
                f"{error}"
            )

            return

        # ----------------------------------------------------
        # Verify dimensions
        # ----------------------------------------------------

        if mask.shape != depth_mm.shape:

            self.get_logger().error(
                "Tray mask and depth image "
                "dimensions do not match: "
                f"mask={mask.shape}, "
                f"depth={depth_mm.shape}"
            )

            return

        tray_mask = (
            mask > 0
        )

        if not np.any(
            tray_mask
        ):

            # No tray detected in this frame.
            return

        # ====================================================
        # Tray Pixels -> XYZ
        # ====================================================

        tray_points = self.extract_tray_points(
            tray_mask,
            depth_mm,
        )

        if tray_points is None:

            return

        # ====================================================
        # Fit Plane
        # ====================================================

        plane = self.fit_plane(
            tray_points
        )

        if plane is None:

            return

        coefficients = (
            plane["coefficients"]
            .copy()
        )

        # ====================================================
        # Calculate Tray Centroid
        # ====================================================

        centroid = self.calculate_tray_centroid(
            tray_mask,
            coefficients,
        )

        if centroid is None:

            return

        # ====================================================
        # Orient Normal Toward Camera
        # ====================================================
        #
        # Camera origin:
        #
        #     [0, 0, 0]
        #
        # centroid points:
        #
        #     camera -> tray
        #
        # If:
        #
        #     normal dot centroid > 0
        #
        # then normal also points camera -> tray.
        #
        # We want:
        #
        #     tray -> camera
        #
        # so flip it.
        # ====================================================

        normal = (
            coefficients[:3]
            .copy()
        )

        if np.dot(
            normal,
            centroid,
        ) > 0.0:

            coefficients *= -1.0

            normal = (
                coefficients[:3]
                .copy()
            )

        # ====================================================
        # Publish
        # ====================================================

        self.publish_results(
            depth_msg,
            centroid,
            normal,
            coefficients,
        )

    # ========================================================
    # Mask + Depth -> XYZ
    # ========================================================

    def extract_tray_points(
        self,
        tray_mask,
        depth_mm,
    ):
        """
        Back-project valid tray pixels into camera-frame XYZ.

        For each tray pixel:

            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth

        Because Z is in millimeters:

            X, Y, Z are all millimeters.
        """

        K = self.camera_matrix

        fx = K[
            0,
            0,
        ]

        fy = K[
            1,
            1,
        ]

        cx = K[
            0,
            2,
        ]

        cy = K[
            1,
            2,
        ]

        # ----------------------------------------------------
        # Valid tray depth pixels
        # ----------------------------------------------------
        #
        # 0 means invalid depth in our zed_camera node.
        # ----------------------------------------------------

        valid = (
            tray_mask
            & (depth_mm > self.min_depth_mm)
            & (depth_mm < self.max_depth_mm)
        )

        v, u = np.nonzero(
            valid
        )

        number_of_points = len(
            u
        )

        if (
            number_of_points
            < self.min_3d_points
        ):

            self.get_logger().warning(
                "Not enough valid tray depth "
                f"points: {number_of_points}"
            )

            return None

        # ----------------------------------------------------
        # Get depth
        # ----------------------------------------------------

        z = depth_mm[
            v,
            u,
        ].astype(
            np.float64
        )

        # ----------------------------------------------------
        # Back-project into camera coordinates
        # ----------------------------------------------------

        x = (
            (
                u.astype(np.float64)
                - cx
            )
            * z
            / fx
        )

        y = (
            (
                v.astype(np.float64)
                - cy
            )
            * z
            / fy
        )

        points = np.column_stack(
            (
                x,
                y,
                z,
            )
        )

        return points

    # ========================================================
    # Plane Fitting
    # ========================================================

    def fit_plane(
        self,
        tray_points,
    ):
        """
        Fit the dominant tray plane with Open3D RANSAC.
        """

        fit_points = tray_points

        # ----------------------------------------------------
        # Limit number of RANSAC points
        # ----------------------------------------------------

        if (
            len(fit_points)
            > self.max_ransac_points
        ):

            # Uniformly select points across the array.
            indices = np.linspace(
                0,
                len(fit_points) - 1,
                self.max_ransac_points,
                dtype=np.int64,
            )

            fit_points = (
                fit_points[
                    indices
                ]
            )

        # ----------------------------------------------------
        # NumPy -> Open3D
        # ----------------------------------------------------

        point_cloud = (
            o3d.geometry.PointCloud()
        )

        point_cloud.points = (
            o3d.utility.Vector3dVector(
                fit_points
            )
        )

        # ----------------------------------------------------
        # RANSAC
        # ----------------------------------------------------

        try:

            coefficients, inliers = (
                point_cloud.segment_plane(
                    distance_threshold=(
                        self.distance_threshold_mm
                    ),
                    ransac_n=3,
                    num_iterations=(
                        self.ransac_iterations
                    ),
                )
            )

        except Exception as error:

            self.get_logger().error(
                f"Plane RANSAC failed: {error}"
            )

            return None

        if len(inliers) == 0:

            self.get_logger().warning(
                "RANSAC found no plane inliers"
            )

            return None

        coefficients = np.asarray(
            coefficients,
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # Normalize plane equation
        # ----------------------------------------------------

        normal_length = np.linalg.norm(
            coefficients[:3]
        )

        if (
            normal_length
            <= np.finfo(
                np.float64
            ).eps
        ):

            self.get_logger().warning(
                "Invalid plane normal"
            )

            return None

        coefficients /= (
            normal_length
        )

        return {
            "coefficients": coefficients,
            "number_of_mask_points": int(
                len(tray_points)
            ),
            "number_of_ransac_points": int(
                len(fit_points)
            ),
            "number_of_inliers": int(
                len(inliers)
            ),
        }

    # ========================================================
    # Tray Centroid
    # ========================================================

    def calculate_tray_centroid(
        self,
        tray_mask,
        coefficients,
    ):
        """
        Find the 2-D centroid of the segmentation mask.

        A ray is projected from the camera through that pixel
        and intersected with the fitted tray plane.

        This ensures the resulting centroid lies exactly on
        the fitted plane.
        """

        v_pixels, u_pixels = np.nonzero(
            tray_mask
        )

        if len(
            u_pixels
        ) == 0:

            return None

        # ----------------------------------------------------
        # 2-D mask centroid
        # ----------------------------------------------------

        u = float(
            np.mean(
                u_pixels
            )
        )

        v = float(
            np.mean(
                v_pixels
            )
        )

        # ----------------------------------------------------
        # Pixel -> camera ray
        # ----------------------------------------------------

        pixel = np.array(
            [
                u,
                v,
                1.0,
            ],
            dtype=np.float64,
        )

        try:

            ray = np.linalg.solve(
                self.camera_matrix,
                pixel,
            )

        except np.linalg.LinAlgError:

            self.get_logger().warning(
                "Camera matrix could not be inverted"
            )

            return None

        # ----------------------------------------------------
        # Ray-plane intersection
        #
        # Plane:
        #
        #     n · X + d = 0
        #
        # Ray:
        #
        #     X = lambda * ray
        #
        # Therefore:
        #
        #     lambda = -d / (n · ray)
        # ----------------------------------------------------

        normal = (
            coefficients[:3]
        )

        denominator = float(
            normal @ ray
        )

        if abs(
            denominator
        ) <= 1e-8:

            self.get_logger().warning(
                "Centroid ray is parallel "
                "to tray plane"
            )

            return None

        distance_along_ray = (
            -coefficients[3]
            / denominator
        )

        if (
            not np.isfinite(
                distance_along_ray
            )
            or distance_along_ray <= 0.0
        ):

            self.get_logger().warning(
                "Invalid centroid intersection"
            )

            return None

        centroid = (
            distance_along_ray
            * ray
        )

        return centroid

    # ========================================================
    # Publish Results
    # ========================================================

    def publish_results(
        self,
        source_msg,
        centroid,
        normal,
        coefficients,
    ):

        # ====================================================
        # Centroid
        # ====================================================

        centroid_msg = (
            PointStamped()
        )

        centroid_msg.header = (
            source_msg.header
        )

        centroid_msg.point.x = float(
            centroid[0]
        )

        centroid_msg.point.y = float(
            centroid[1]
        )

        centroid_msg.point.z = float(
            centroid[2]
        )

        self.centroid_publisher.publish(
            centroid_msg
        )

        # ====================================================
        # Normal
        # ====================================================

        normal_msg = (
            Vector3Stamped()
        )

        normal_msg.header = (
            source_msg.header
        )

        normal_msg.vector.x = float(
            normal[0]
        )

        normal_msg.vector.y = float(
            normal[1]
        )

        normal_msg.vector.z = float(
            normal[2]
        )

        self.normal_publisher.publish(
            normal_msg
        )

        # ====================================================
        # Plane Equation
        # ====================================================

        plane_msg = (
            Float64MultiArray()
        )

        plane_msg.data = [
            float(
                coefficients[0]
            ),
            float(
                coefficients[1]
            ),
            float(
                coefficients[2]
            ),
            float(
                coefficients[3]
            ),
        ]

        self.plane_publisher.publish(
            plane_msg
        )


# ============================================================
# Main
# ============================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = None

    try:

        node = PlaneDetectorNode()

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    except Exception as error:

        if node is not None:

            node.get_logger().error(
                "Plane detector failed: "
                f"{error}"
            )

        else:

            print(
                "[ERROR] Plane detector failed: "
                f"{error}"
            )

        raise

    finally:

        if node is not None:

            node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == "__main__":

    main()