#!/usr/bin/env python3

"""
ROS 2 ZED Camera Node

Publishes
---------
/camera/left/image_raw
    sensor_msgs/msg/Image
    Encoding: bgr8

/camera/left/camera_info
    sensor_msgs/msg/CameraInfo

/camera/depth
    sensor_msgs/msg/Image
    Encoding: 16UC1
    Units: millimeters

All outputs are retrieved from the same ZED frame.

Camera frame
------------
zed_left_camera

Depth convention
----------------
depth[v, u] = distance/depth in millimeters

Invalid depth pixels are published as 0.
"""

import sys

import cv2 as cv
import numpy as np
import pyzed.sl as sl

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo


# ============================================================
# ZED Configuration
# ============================================================

ZED_RESOLUTION = sl.RESOLUTION.HD1080
ZED_FPS = 30

# All ZED depth measurements will be returned in millimeters.
ZED_UNITS = sl.UNIT.MILLIMETER

ZED_DEPTH_MODE = sl.DEPTH_MODE.NEURAL_PLUS

CAMERA_FRAME = "zed_left_camera"


# ============================================================
# ZED Camera Node
# ============================================================

class ZedCameraNode(Node):

    def __init__(self):

        super().__init__("zed_camera")

        # ====================================================
        # ROS Parameters
        # ====================================================

        self.declare_parameter(
            "frame_id",
            CAMERA_FRAME,
        )

        self.declare_parameter(
            "fps",
            ZED_FPS,
        )

        self.frame_id = str(
            self.get_parameter(
                "frame_id"
            ).value
        )

        self.fps = int(
            self.get_parameter(
                "fps"
            ).value
        )

        # ====================================================
        # ROS Publishers
        # ====================================================

        self.image_publisher = self.create_publisher(
            Image,
            "/camera/left/image_raw",
            qos_profile_sensor_data,
        )

        self.camera_info_publisher = self.create_publisher(
            CameraInfo,
            "/camera/left/camera_info",
            qos_profile_sensor_data,
        )

        self.depth_publisher = self.create_publisher(
            Image,
            "/camera/depth",
            qos_profile_sensor_data,
        )

        # ====================================================
        # ZED Camera Initialization
        # ====================================================

        self.zed = sl.Camera()

        init_params = sl.InitParameters()

        init_params.camera_resolution = (
            ZED_RESOLUTION
        )

        init_params.camera_fps = (
            self.fps
        )

        init_params.depth_mode = (
            ZED_DEPTH_MODE
        )

        # Depth values returned by the SDK will be in mm.
        init_params.coordinate_units = (
            ZED_UNITS
        )

        # Preserve the camera coordinate convention used by
        # the existing project.
        init_params.coordinate_system = (
            sl.COORDINATE_SYSTEM.IMAGE
        )

        status = self.zed.open(
            init_params
        )

        if status != sl.ERROR_CODE.SUCCESS:

            raise RuntimeError(
                f"Could not open ZED camera: {status}"
            )

        # ====================================================
        # Runtime Parameters
        # ====================================================

        self.runtime_params = (
            sl.RuntimeParameters()
        )

        self.runtime_params.confidence_threshold = 50

        self.runtime_params.measure3D_reference_frame = (
            sl.REFERENCE_FRAME.CAMERA
        )

        # ====================================================
        # Reusable ZED Buffers
        # ====================================================

        self.image_zed = sl.Mat()

        self.depth_zed = sl.Mat()

        # ====================================================
        # Camera Calibration
        # ====================================================

        self.camera_info_msg = (
            self.create_camera_info()
        )

        # ====================================================
        # ROS Timer
        # ====================================================

        timer_period = (
            1.0 / float(self.fps)
        )

        self.timer = self.create_timer(
            timer_period,
            self.camera_callback,
        )

        # ====================================================
        # Startup Logging
        # ====================================================

        self.get_logger().info(
            f"ZED camera opened at {self.fps} FPS"
        )

        self.get_logger().info(
            f"Camera frame: {self.frame_id}"
        )

        self.get_logger().info(
            "Depth units: millimeters"
        )

        self.get_logger().info(
            "Publishing:"
        )

        self.get_logger().info(
            "  /camera/left/image_raw"
        )

        self.get_logger().info(
            "  /camera/left/camera_info"
        )

        self.get_logger().info(
            "  /camera/depth"
        )

    # ========================================================
    # CameraInfo
    # ========================================================

    def create_camera_info(self):
        """
        Create a ROS CameraInfo message for the rectified
        ZED left camera.

        sl.VIEW.LEFT is rectified, therefore zero distortion
        coefficients are published.
        """

        camera_information = (
            self.zed.get_camera_information()
        )

        camera_configuration = (
            camera_information
            .camera_configuration
        )

        left_camera = (
            camera_configuration
            .calibration_parameters
            .left_cam
        )

        resolution = (
            camera_configuration
            .resolution
        )

        fx = float(
            left_camera.fx
        )

        fy = float(
            left_camera.fy
        )

        cx = float(
            left_camera.cx
        )

        cy = float(
            left_camera.cy
        )

        msg = CameraInfo()

        msg.header.frame_id = (
            self.frame_id
        )

        msg.width = int(
            resolution.width
        )

        msg.height = int(
            resolution.height
        )

        # ----------------------------------------------------
        # Distortion
        # ----------------------------------------------------

        msg.distortion_model = (
            "plumb_bob"
        )

        msg.d = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        # ----------------------------------------------------
        # Intrinsic matrix K
        #
        # [ fx  0 cx ]
        # [  0 fy cy ]
        # [  0  0  1 ]
        # ----------------------------------------------------

        msg.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0,
        ]

        # ----------------------------------------------------
        # Rectification matrix
        # ----------------------------------------------------

        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]

        # ----------------------------------------------------
        # Projection matrix
        # ----------------------------------------------------

        msg.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

        return msg

    # ========================================================
    # NumPy -> ROS Image
    # ========================================================

    def numpy_to_image_msg(
        self,
        image,
        encoding,
        stamp,
    ):
        """
        Convert a NumPy image directly into sensor_msgs/Image.

        This intentionally avoids cv_bridge so the ROS node
        does not mix cv_bridge's system C++ libraries with
        Conda's native libraries.
        """

        image = np.ascontiguousarray(
            image
        )

        msg = Image()

        msg.header.stamp = stamp
        msg.header.frame_id = (
            self.frame_id
        )

        msg.height = int(
            image.shape[0]
        )

        msg.width = int(
            image.shape[1]
        )

        msg.encoding = encoding

        msg.is_bigendian = (
            sys.byteorder == "big"
        )

        # ----------------------------------------------------
        # Validate image according to ROS encoding
        # ----------------------------------------------------

        if encoding == "bgr8":

            if image.dtype != np.uint8:

                raise ValueError(
                    "bgr8 image must use uint8"
                )

            if (
                image.ndim != 3
                or image.shape[2] != 3
            ):

                raise ValueError(
                    "bgr8 image must have shape "
                    "(height, width, 3)"
                )

        elif encoding == "16UC1":

            if image.dtype != np.uint16:

                raise ValueError(
                    "16UC1 image must use uint16"
                )

            if image.ndim != 2:

                raise ValueError(
                    "16UC1 image must have shape "
                    "(height, width)"
                )

        else:

            raise ValueError(
                f"Unsupported image encoding: "
                f"{encoding}"
            )

        # Number of bytes in one image row.
        msg.step = int(
            image.strides[0]
        )

        msg.data = (
            image.tobytes()
        )

        return msg

    # ========================================================
    # Main Camera Callback
    # ========================================================

    def camera_callback(self):
        """
        Grab exactly ONE ZED frame.

        RGB and depth are retrieved from that same frame and
        assigned the same ROS timestamp.
        """

        # ----------------------------------------------------
        # Grab synchronized ZED frame
        # ----------------------------------------------------

        grab_status = self.zed.grab(
            self.runtime_params
        )

        if grab_status != sl.ERROR_CODE.SUCCESS:

            self.get_logger().warning(
                f"Could not grab ZED frame: "
                f"{grab_status}"
            )

            return

        # One timestamp for:
        #
        # RGB
        # depth
        # CameraInfo
        #
        stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        # ====================================================
        # Retrieve Rectified Left Image
        # ====================================================

        image_status = self.zed.retrieve_image(
            self.image_zed,
            sl.VIEW.LEFT,
        )

        if image_status != sl.ERROR_CODE.SUCCESS:

            self.get_logger().warning(
                f"Could not retrieve ZED image: "
                f"{image_status}"
            )

            return

        image_bgra = (
            self.image_zed.get_data()
        )

        if (
            image_bgra is None
            or image_bgra.size == 0
        ):

            self.get_logger().warning(
                "Retrieved empty ZED image"
            )

            return

        # ZED image is BGRA.
        # OpenCV / Detectron2 expect BGR.
        image_bgr = cv.cvtColor(
            image_bgra,
            cv.COLOR_BGRA2BGR,
        )

        # ====================================================
        # Retrieve Depth
        # ====================================================

        depth_status = self.zed.retrieve_measure(
            self.depth_zed,
            sl.MEASURE.DEPTH,
        )

        if depth_status != sl.ERROR_CODE.SUCCESS:

            self.get_logger().warning(
                f"Could not retrieve ZED depth: "
                f"{depth_status}"
            )

            return

        depth_data = (
            self.depth_zed.get_data()
        )

        if (
            depth_data is None
            or depth_data.size == 0
        ):

            self.get_logger().warning(
                "Retrieved empty ZED depth image"
            )

            return

        # ----------------------------------------------------
        # ZED depth is float32.
        #
        # Because coordinate_units = MILLIMETER:
        #
        # depth_mm_float[v, u]
        #
        # is already in millimeters.
        # ----------------------------------------------------

        depth_mm_float = np.asarray(
            depth_data,
            dtype=np.float32,
        )

        # ====================================================
        # Convert depth to 16UC1 millimeters
        # ====================================================
        #
        # ROS convention:
        #
        # 16UC1:
        #     unsigned 16-bit integer
        #     depth in millimeters
        #
        # 0:
        #     invalid depth
        # ====================================================

        depth_mm = np.zeros(
            depth_mm_float.shape,
            dtype=np.uint16,
        )

        valid_depth = (
            np.isfinite(
                depth_mm_float
            )
            & (depth_mm_float > 0.0)
            & (depth_mm_float <= 65535.0)
        )

        depth_mm[valid_depth] = np.rint(
            depth_mm_float[valid_depth]
        ).astype(
            np.uint16
        )

        # ====================================================
        # Publish RGB Image
        # ====================================================

        image_msg = self.numpy_to_image_msg(
            image_bgr,
            "bgr8",
            stamp,
        )

        self.image_publisher.publish(
            image_msg
        )

        # ====================================================
        # Publish Depth Image
        # ====================================================

        depth_msg = self.numpy_to_image_msg(
            depth_mm,
            "16UC1",
            stamp,
        )

        self.depth_publisher.publish(
            depth_msg
        )

        # ====================================================
        # Publish CameraInfo
        # ====================================================

        self.camera_info_msg.header.stamp = (
            stamp
        )

        self.camera_info_msg.header.frame_id = (
            self.frame_id
        )

        self.camera_info_publisher.publish(
            self.camera_info_msg
        )

    # ========================================================
    # Shutdown
    # ========================================================

    def destroy_node(self):
        """
        Close the ZED cleanly when the node shuts down.
        """

        self.get_logger().info(
            "Closing ZED camera..."
        )

        if self.zed is not None:

            self.zed.close()

        super().destroy_node()


# ============================================================
# Main
# ============================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = None

    try:

        node = ZedCameraNode()

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    except Exception as error:

        if node is not None:

            node.get_logger().error(
                f"ZED camera node failed: "
                f"{error}"
            )

        else:

            print(
                "[ERROR] ZED camera node failed: "
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