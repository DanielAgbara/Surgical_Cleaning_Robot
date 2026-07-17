#!/usr/bin/env python3

"""
Generate a 4x4 ChArUco board for eye-to-hand calibration.

This board is designed to have larger markers than the 5x5 board.

Intended physical board:
    - Board size:      200 mm x 200 mm
    - Squares:         4 x 4
    - Square length:   50 mm
    - Marker length:   37.5 mm
    - Dictionary:      DICT_4X4_50

Why 4x4?
    - Larger markers are easier to detect from farther away.
    - 4x4 still gives 3x3 = 9 ChArUco corners.
    - This is less accurate than 5x5, but better for wider distance range.

Outputs:
    generated_boards/charuco_4x4_200mm.png
    generated_boards/charuco_4x4_200mm.pdf

Print instructions:
    - Print the PDF at 100% scale.
    - Disable "fit to page".
    - After printing, measure:
        1. one square side length
        2. one marker side length
    - Use the measured values in your detection/calibration script.
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

OUTPUT_NAME = "charuco_4x4_200mm"

# Print resolution.
# 300 DPI is enough for accurate printing.
DPI = 300


# --------------------------------------------------
# Physical board settings
# --------------------------------------------------

BOARD_WIDTH_MM = 200.0
BOARD_HEIGHT_MM = 200.0

SQUARES_X = 4
SQUARES_Y = 4

# 4 squares * 50 mm = 200 mm
SQUARE_LENGTH_MM = 50.0

# Marker is 75% of square size.
# 37.5 / 50 = 0.75
MARKER_LENGTH_MM = 37.5

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def mm_to_pixels(mm: float, dpi: int) -> int:
    """
    Convert millimeters to pixels for a given DPI.

    1 inch = 25.4 mm
    pixels = inches * DPI
    """
    return int(round((mm / 25.4) * dpi))


def create_charuco_board(dictionary):
    """
    Create a ChArUco board.

    OpenCV's Python API changed across versions, so this function supports both:
        - newer API: cv2.aruco.CharucoBoard(...)
        - older API: cv2.aruco.CharucoBoard_create(...)
    """

    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y),
            SQUARE_LENGTH_MM,
            MARKER_LENGTH_MM,
            dictionary,
        )
        return board

    if hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(
            SQUARES_X,
            SQUARES_Y,
            SQUARE_LENGTH_MM,
            MARKER_LENGTH_MM,
            dictionary,
        )
        return board

    raise RuntimeError(
        "Your OpenCV install does not support ChArUco boards. "
        "Install opencv-contrib-python."
    )


def draw_charuco_board(board, image_size_px):
    """
    Draw the ChArUco board image.

    OpenCV version compatibility:
        - newer API: board.generateImage(...)
        - older API: board.draw(...)
    """

    if hasattr(board, "generateImage"):
        img = board.generateImage(
            image_size_px,
            marginSize=0,
            borderBits=1,
        )
        return img

    img = board.draw(
        image_size_px,
        marginSize=0,
        borderBits=1,
    )
    return img


def save_png_and_pdf(img_gray: np.ndarray, output_base: str, dpi: int):
    """
    Save the generated board as both PNG and PDF.

    The PDF is what you should print.
    The PNG is useful for previewing or documentation.
    """

    png_path = OUTPUT_DIR / f"{output_base}.png"
    pdf_path = OUTPUT_DIR / f"{output_base}.pdf"

    pil_img = Image.fromarray(img_gray)

    # Save PNG with DPI metadata.
    pil_img.save(png_path, dpi=(dpi, dpi))

    # Save PDF with the desired print resolution.
    pil_img.save(pdf_path, "PDF", resolution=dpi)

    print("")
    print("Saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    print("")
    print("Print instructions:")
    print("  - Print the PDF at 100% scale.")
    print("  - Disable 'fit to page'.")
    print("  - Measure the printed square length.")
    print("  - Measure the printed marker length.")
    print("")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("[INFO] OpenCV version:", cv2.__version__)
    print("[INFO] OpenCV path:", cv2.__file__)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)

    board_width_px = mm_to_pixels(BOARD_WIDTH_MM, DPI)
    board_height_px = mm_to_pixels(BOARD_HEIGHT_MM, DPI)

    image_size_px = (board_width_px, board_height_px)

    board = create_charuco_board(dictionary)
    img = draw_charuco_board(board, image_size_px)

    save_png_and_pdf(img, OUTPUT_NAME, DPI)

    print("Board parameters for detection/calibration code:")
    print(f"  SQUARES_X = {SQUARES_X}")
    print(f"  SQUARES_Y = {SQUARES_Y}")
    print(f"  SQUARE_LENGTH_M = {SQUARE_LENGTH_MM / 1000.0}")
    print(f"  MARKER_LENGTH_M = {MARKER_LENGTH_MM / 1000.0}")
    print("  ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50")
    print("")
    print("Expected printed board:")
    print(f"  Board width:    {BOARD_WIDTH_MM:.1f} mm")
    print(f"  Board height:   {BOARD_HEIGHT_MM:.1f} mm")
    print(f"  Square length:  {SQUARE_LENGTH_MM:.1f} mm")
    print(f"  Marker length:  {MARKER_LENGTH_MM:.1f} mm")
    print("")
    print("After printing, update the test script with the measured values.")


if __name__ == "__main__":
    main()