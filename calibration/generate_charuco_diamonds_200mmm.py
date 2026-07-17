#!/usr/bin/env python3

"""
Generate a 200 mm x 200 mm ChArUco diamond-style board.

This version does NOT use cv2.aruco.drawCharucoDiamond,
because that function is missing in some OpenCV Python builds.

Board/page:
    Total printed size: 200 mm x 200 mm

Diamond pattern:
    3 x 3 squares
    Square length: 60 mm
    Marker length: 42 mm
    Pattern size: 180 mm x 180 mm
    Margin around pattern: 10 mm

Marker IDs:
    [0, 1, 2, 3]

Dictionary:
    DICT_4X4_50

Outputs:
    generated_boards/charuco_diamond_200mm.png
    generated_boards/charuco_diamond_200mm.pdf

Print the PDF at 100% scale.
Do NOT use "fit to page".
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# --------------------------------------------------
# Output settings
# --------------------------------------------------

OUTPUT_DIR = Path("generated_boards")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_NAME = "charuco_diamond_200mm"

DPI = 300


# --------------------------------------------------
# Physical dimensions
# --------------------------------------------------

BOARD_SIZE_MM = 200.0

SQUARE_LENGTH_MM = 60.0
MARKER_LENGTH_MM = 42.0

# 3 squares * 60 mm = 180 mm
PATTERN_SIZE_MM = 3.0 * SQUARE_LENGTH_MM

# 200 mm board - 180 mm pattern = 20 mm total margin
MARGIN_MM = (BOARD_SIZE_MM - PATTERN_SIZE_MM) / 2.0


# --------------------------------------------------
# ArUco settings
# --------------------------------------------------

ARUCO_DICT_NAME = cv2.aruco.DICT_4X4_50
MARKER_IDS = [0, 1, 2, 3]


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def mm_to_px(mm: float, dpi: int) -> int:
    return int(round((mm / 25.4) * dpi))


def generate_marker(dictionary, marker_id: int, size_px: int) -> np.ndarray:
    """
    Generate one ArUco marker image.

    Uses the newer OpenCV API when available.
    """

    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(
            dictionary,
            marker_id,
            size_px,
            borderBits=1,
        )
        return marker

    if hasattr(cv2.aruco, "drawMarker"):
        marker = np.zeros((size_px, size_px), dtype=np.uint8)
        cv2.aruco.drawMarker(
            dictionary,
            marker_id,
            size_px,
            marker,
            borderBits=1,
        )
        return marker

    raise RuntimeError(
        "No marker generation function found. "
        "Install opencv-contrib-python."
    )


def paste_centered_marker(canvas, marker, square_x0, square_y0, square_size_px):
    """
    Paste marker centered inside a chessboard square.
    """

    marker_h, marker_w = marker.shape[:2]

    x0 = square_x0 + (square_size_px - marker_w) // 2
    y0 = square_y0 + (square_size_px - marker_h) // 2

    x1 = x0 + marker_w
    y1 = y0 + marker_h

    canvas[y0:y1, x0:x1] = marker


def save_png_and_pdf(img_gray: np.ndarray, output_name: str, dpi: int):
    png_path = OUTPUT_DIR / f"{output_name}.png"
    pdf_path = OUTPUT_DIR / f"{output_name}.pdf"

    pil_img = Image.fromarray(img_gray)

    pil_img.save(png_path, dpi=(dpi, dpi))
    pil_img.save(pdf_path, "PDF", resolution=dpi)

    print("")
    print("Saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    print("")
    print("Print instructions:")
    print("  - Print the PDF at 100% scale.")
    print("  - Disable 'fit to page'.")
    print("  - Measure one square after printing.")
    print("  - One square should be 60 mm.")
    print("")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("OpenCV version:", cv2.__version__)
    print("cv2 path:", cv2.__file__)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)

    board_size_px = mm_to_px(BOARD_SIZE_MM, DPI)
    square_px = mm_to_px(SQUARE_LENGTH_MM, DPI)
    marker_px = mm_to_px(MARKER_LENGTH_MM, DPI)
    margin_px = mm_to_px(MARGIN_MM, DPI)

    # Full white 200 mm x 200 mm canvas
    canvas = np.ones(
        (board_size_px, board_size_px),
        dtype=np.uint8,
    ) * 255

    # Draw 3x3 chessboard pattern.
    # White background already exists.
    # We draw black squares manually.
    #
    # Layout:
    #   black white black
    #   white black white
    #   black white black
    #
    pattern_x0 = margin_px
    pattern_y0 = margin_px

    for row in range(3):
        for col in range(3):
            if (row + col) % 2 == 0:
                x0 = pattern_x0 + col * square_px
                y0 = pattern_y0 + row * square_px
                x1 = x0 + square_px
                y1 = y0 + square_px
                canvas[y0:y1, x0:x1] = 0

    # Marker placement on the four white squares around the center.
    #
    # Positions:
    #   top marker:    row 0, col 1
    #   left marker:   row 1, col 0
    #   right marker:  row 1, col 2
    #   bottom marker: row 2, col 1
    #
    marker_positions = [
        (0, 1),  # top
        (1, 0),  # left
        (1, 2),  # right
        (2, 1),  # bottom
    ]

    for marker_id, (row, col) in zip(MARKER_IDS, marker_positions):
        marker = generate_marker(dictionary, marker_id, marker_px)

        square_x0 = pattern_x0 + col * square_px
        square_y0 = pattern_y0 + row * square_px

        paste_centered_marker(
            canvas=canvas,
            marker=marker,
            square_x0=square_x0,
            square_y0=square_y0,
            square_size_px=square_px,
        )

    save_png_and_pdf(canvas, OUTPUT_NAME, DPI)

    print("Diamond-style board parameters:")
    print(f"  BOARD_SIZE_M = {BOARD_SIZE_MM / 1000.0}")
    print(f"  SQUARE_LENGTH_M = {SQUARE_LENGTH_MM / 1000.0}")
    print(f"  MARKER_LENGTH_M = {MARKER_LENGTH_MM / 1000.0}")
    print(f"  MARGIN_M = {MARGIN_MM / 1000.0}")
    print(f"  MARKER_IDS = {MARKER_IDS}")
    print("  ARUCO_DICT = cv2.aruco.DICT_4X4_50")
    print("")
    print("Important:")
    print("  This is generated manually to avoid OpenCV API issues.")
    print("  For eye-to-hand calibration, I still recommend the full 5x5 ChArUco board.")


if __name__ == "__main__":
    main()