import cv2
import pyzed.sl as sl
import torch

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


# -------------------------------------------------
# Detectron2 Setup
# -------------------------------------------------

cfg = get_cfg()

cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
)

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

print(f"Detectron2 model loaded on {cfg.MODEL.DEVICE}")


# -------------------------------------------------
# COCO Class Metadata
# -------------------------------------------------

metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

class_names = metadata.thing_classes

print("\nCOCO Classes Loaded:")
print(class_names)


# -------------------------------------------------
# ZED Camera Setup
# -------------------------------------------------

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30

status = zed.open(init_params)

if status != sl.ERROR_CODE.SUCCESS:
    raise RuntimeError(f"Could not open ZED camera: {status}")

image_zed = sl.Mat()

print("\nZED camera opened successfully")
print("Press 'q' to quit")


# -------------------------------------------------
# Main Loop
# -------------------------------------------------

try:

    while True:

        if zed.grab() == sl.ERROR_CODE.SUCCESS:

            # Get left camera image
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)

            frame_rgba = image_zed.get_data()

            # Convert RGBA -> BGR
            frame = cv2.cvtColor(frame_rgba, cv2.COLOR_BGRA2BGR)

            # -------------------------------------------------
            # Run Detectron2 inference
            # -------------------------------------------------

            outputs = predictor(frame)

            instances = outputs["instances"].to("cpu")

            # -------------------------------------------------
            # Extract predictions
            # -------------------------------------------------

            pred_classes = instances.pred_classes
            scores = instances.scores

            print("\nDetected objects:")

            for class_id, score in zip(pred_classes, scores):

                object_name = class_names[class_id]

                confidence = score.item()

                print(f"{object_name}: {confidence:.2f}")

            # -------------------------------------------------
            # Draw detections
            # -------------------------------------------------

            visualizer = Visualizer(
                frame[:, :, ::-1],
                metadata=metadata,
                scale=1.0
            )

            vis_output = visualizer.draw_instance_predictions(
                instances
            )

            result_frame = vis_output.get_image()[:, :, ::-1]

            # -------------------------------------------------
            # Show frame
            # -------------------------------------------------

            cv2.imshow(
                "ZED 2 + Detectron2 Mask R-CNN",
                result_frame
            )

        # Quit with q key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    zed.close()

    cv2.destroyAllWindows()

    print("\nCamera closed")