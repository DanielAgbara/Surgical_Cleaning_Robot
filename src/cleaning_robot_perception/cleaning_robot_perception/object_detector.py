#!/usr/bin/env python3

"""
ROS 2 object detection node using Detectron2.

Currently detects the surgical tray using the trained
Mask R-CNN model.

Subscribes
----------
/camera/left/image_raw
    sensor_msgs/msg/Image
    Encoding: bgr8

Publishes
---------
/perception/tray/mask
    sensor_msgs/msg/Image
    Encoding: mono8
    255 = tray
    0   = background

/perception/tray/score
    std_msgs/msg/Float32
    Detection confidence.

No cv_bridge is used.
"""

import os

import cv2 as cv
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from ament_index_python.packages import (
    get_package_share_directory,
)

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


# ============================================================
# Package
# ============================================================

PACKAGE_NAME = "cleaning_robot_perception"


# ============================================================
# Object Detector Node
# ============================================================

class ObjectDetectorNode(Node):

    def __init__(self):

        super().__init__("object_detector")

        # ====================================================
        # Parameters
        # ====================================================

        self.declare_parameter(
            "score_threshold",
            0.95,
        )

        self.declare_parameter(
            "device",
            "auto",
        )

        self.declare_parameter(
            "mask_kernel_size",
            5,
        )

        self.declare_parameter(
            "model_name",
            "maskrcnn_tray",
        )

        self.score_threshold = float(
            self.get_parameter(
                "score_threshold"
            ).value
        )

        requested_device = str(
            self.get_parameter(
                "device"
            ).value
        )

        self.mask_kernel_size = int(
            self.get_parameter(
                "mask_kernel_size"
            ).value
        )

        self.model_name = str(
            self.get_parameter(
                "model_name"
            ).value
        )

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        if requested_device == "auto":

            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

        else:

            self.device = requested_device

        # ====================================================
        # Load Detectron2 Model
        # ====================================================

        self.predictor = (
            self.build_predictor()
        )

        # ====================================================
        # Subscriber
        # ====================================================

        self.image_subscription = (
            self.create_subscription(
                Image,
                "/camera/left/image_raw",
                self.image_callback,
                qos_profile_sensor_data,
            )
        )

        # ====================================================
        # Publishers
        # ====================================================

        self.mask_publisher = (
            self.create_publisher(
                Image,
                "/perception/tray/mask",
                qos_profile_sensor_data,
            )
        )

        self.score_publisher = (
            self.create_publisher(
                Float32,
                "/perception/tray/score",
                qos_profile_sensor_data,
            )
        )

        # ====================================================
        # Startup Information
        # ====================================================

        self.get_logger().info(
            "Object detector started"
        )

        self.get_logger().info(
            f"Model: {self.model_name}"
        )

        self.get_logger().info(
            f"Threshold: "
            f"{self.score_threshold:.2f}"
        )

    # ========================================================
    # Build Predictor
    # ========================================================

    def build_predictor(self):
        """
        Load Detectron2 configuration and trained weights.
        """

        package_share = (
            get_package_share_directory(
                PACKAGE_NAME
            )
        )

        model_directory = os.path.join(
            package_share,
            "models",
            self.model_name,
        )

        config_path = os.path.join(
            model_directory,
            "config.yaml",
        )

        weights_path = os.path.join(
            model_directory,
            "model_final.pth",
        )

        # ----------------------------------------------------
        # Verify files
        # ----------------------------------------------------

        if not os.path.isfile(config_path):

            raise FileNotFoundError(
                "Detectron2 configuration not found: "
                f"{config_path}"
            )

        if not os.path.isfile(weights_path):

            raise FileNotFoundError(
                "Detectron2 weights not found: "
                f"{weights_path}"
            )

        # ----------------------------------------------------
        # Detectron2 configuration
        # ----------------------------------------------------

        cfg = get_cfg()

        cfg.merge_from_file(
            config_path
        )

        cfg.MODEL.WEIGHTS = (
            weights_path
        )

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
            self.score_threshold
        )

        cfg.MODEL.DEVICE = (
            self.device
        )

        self.get_logger().info(
            f"Loading Detectron2 model on "
            f"{self.device}"
        )

        predictor = DefaultPredictor(
            cfg
        )

        return predictor

    # ========================================================
    # ROS Image -> NumPy
    # ========================================================

    def ros_image_to_bgr(
        self,
        msg: Image,
    ):
        """
        Convert sensor_msgs/Image to a BGR NumPy array.

        This node expects the ZED camera node to publish bgr8.

        No cv_bridge is used.
        """

        if msg.encoding != "bgr8":

            raise ValueError(
                "Expected image encoding 'bgr8', "
                f"received '{msg.encoding}'"
            )

        height = int(
            msg.height
        )

        width = int(
            msg.width
        )

        expected_row_bytes = (
            width * 3
        )

        if msg.step < expected_row_bytes:

            raise ValueError(
                "ROS image step is smaller than expected "
                "for bgr8 image"
            )

        # ----------------------------------------------------
        # Read raw ROS byte data
        # ----------------------------------------------------

        raw = np.frombuffer(
            msg.data,
            dtype=np.uint8,
        )

        expected_total_bytes = (
            height * msg.step
        )

        if raw.size < expected_total_bytes:

            raise ValueError(
                "ROS image data buffer is smaller "
                "than expected"
            )

        # ----------------------------------------------------
        # Account for possible row padding
        # ----------------------------------------------------

        rows = raw[
            :expected_total_bytes
        ].reshape(
            height,
            msg.step,
        )

        image = rows[
            :,
            :expected_row_bytes,
        ].reshape(
            height,
            width,
            3,
        )

        # Detectron2/OpenCV work best with a contiguous,
        # writable array.
        image = np.ascontiguousarray(
            image
        ).copy()

        return image

    # ========================================================
    # NumPy Mask -> ROS Image
    # ========================================================

    def mask_to_ros_image(
        self,
        mask,
        source_msg,
    ):
        """
        Convert uint8 H x W mask to a mono8 ROS Image.
        """

        mask = np.ascontiguousarray(
            mask,
            dtype=np.uint8,
        )

        if mask.ndim != 2:

            raise ValueError(
                "Mask must have shape H x W"
            )

        msg = Image()

        # Preserve the RGB image timestamp and frame.
        #
        # This is important because plane_detector will
        # synchronize this mask with the corresponding depth
        # frame.
        msg.header = source_msg.header

        msg.height = int(
            mask.shape[0]
        )

        msg.width = int(
            mask.shape[1]
        )

        msg.encoding = "mono8"

        msg.is_bigendian = False

        msg.step = int(
            mask.shape[1]
        )

        msg.data = (
            mask.tobytes()
        )

        return msg

    # ========================================================
    # Image Callback
    # ========================================================

    def image_callback(
        self,
        msg: Image,
    ):
        """
        Run Detectron2 on the incoming image and publish
        the highest-confidence tray mask.
        """

        # ----------------------------------------------------
        # ROS image -> NumPy
        # ----------------------------------------------------

        try:

            image_bgr = (
                self.ros_image_to_bgr(
                    msg
                )
            )

        except Exception as error:

            self.get_logger().error(
                f"Could not convert ROS image: "
                f"{error}"
            )

            return

        # ----------------------------------------------------
        # Run Detectron2
        # ----------------------------------------------------

        try:

            outputs = self.predictor(
                image_bgr
            )

        except Exception as error:

            self.get_logger().error(
                f"Detectron2 inference failed: "
                f"{error}"
            )

            return

        # ----------------------------------------------------
        # Get predictions
        # ----------------------------------------------------

        instances = outputs[
            "instances"
        ].to(
            "cpu"
        )

        # ----------------------------------------------------
        # No detection
        # ----------------------------------------------------

        if len(instances) == 0:

            empty_mask = np.zeros(
                (
                    msg.height,
                    msg.width,
                ),
                dtype=np.uint8,
            )

            mask_msg = (
                self.mask_to_ros_image(
                    empty_mask,
                    msg,
                )
            )

            score_msg = Float32()
            score_msg.data = 0.0

            self.mask_publisher.publish(
                mask_msg
            )

            self.score_publisher.publish(
                score_msg
            )

            return

        # ----------------------------------------------------
        # Select highest-confidence detection
        # ----------------------------------------------------

        scores = (
            instances.scores
            .numpy()
        )

        best_index = int(
            np.argmax(scores)
        )

        best_score = float(
            scores[best_index]
        )

        # ----------------------------------------------------
        # Extract segmentation mask
        # ----------------------------------------------------

        mask = (
            instances.pred_masks[
                best_index
            ]
            .numpy()
            .astype(np.uint8)
            * 255
        )

        # ====================================================
        # Clean Mask
        # ====================================================

        kernel_size = max(
            1,
            self.mask_kernel_size,
        )

        kernel = np.ones(
            (
                kernel_size,
                kernel_size,
            ),
            dtype=np.uint8,
        )

        # Remove small isolated noise.
        mask = cv.morphologyEx(
            mask,
            cv.MORPH_OPEN,
            kernel,
        )

        # Fill small gaps in the detected tray.
        mask = cv.morphologyEx(
            mask,
            cv.MORPH_CLOSE,
            kernel,
        )

        # ====================================================
        # Publish Mask
        # ====================================================

        mask_msg = (
            self.mask_to_ros_image(
                mask,
                msg,
            )
        )

        self.mask_publisher.publish(
            mask_msg
        )

        # ====================================================
        # Publish Detection Score
        # ====================================================

        score_msg = Float32()

        score_msg.data = (
            best_score
        )

        self.score_publisher.publish(
            score_msg
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

        node = ObjectDetectorNode()

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    except Exception as error:

        if node is not None:

            node.get_logger().error(
                "Object detector failed: "
                f"{error}"
            )

        else:

            print(
                "[ERROR] Object detector failed: "
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