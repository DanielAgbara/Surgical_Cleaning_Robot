#!/usr/bin/env python3

"""
Generate a 5x5 ChArUco board for eye-to-hand calibration.

Board:
    - Total pattern size: 200 mm x 200 mm
    - Squares: 5 x 5
    - Square length: 40 mm
    - Marker length: 28 mm
    - Dictionary: DICT_4X4_50

Outputs:
    - charuco_5x5_200mm.png
    - charuco_5x5_200mm.pdf

Print the PDF at 100% scale.
Do NOT use "fit to page".
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path


# --------------------------------------------------
# Board settings
# --------------------------------------------------

OUTPUT_DIR = Path("generated_boards")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_NAME = "charuco_5x5_200mm"

# Printable resolution
DPI = 300

# Physical board size
BOARD_WIDTH_MM = 200.0
BOARD_HEIGHT_MM = 200.0

# ChArUco board parameters
SQUARES_X = 5
SQUARES_Y = 5

SQUARE_LENGTH_MM = 40.0
MARKER_LENGTH_MM = 28.0

# OpenCV ArUco dictionary
ARUCO_DICT_NAME = cv2.aruco.DICT_4X4_50


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def mm_to_pixels(mm, dpi):
    """
    Convert millimeters to pixels for a given print DPI.
    """
    return int(round((mm / 25.4) * dpi))


def create_charuco_board(dictionary):
    """
    Create ChArUco board with compatibility for newer and older OpenCV versions.
    """

    # Newer OpenCV API
    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y),
            SQUARE_LENGTH_MM,
            MARKER_LENGTH_MM,
            dictionary,
        )
        return board

    # Older OpenCV API
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
        "Your OpenCV version does not support ChArUco boards. "
        "Install opencv-contrib-python."
    )


def draw_charuco_board(board, image_size_px):
    """
    Draw ChArUco board with compatibility for newer and older OpenCV versions.
    """

    # Newer OpenCV API
    if hasattr(board, "generateImage"):
        img = board.generateImage(
            image_size_px,
            marginSize=0,
            borderBits=1,
        )
        return img

    # Older OpenCV API
    img = board.draw(
        image_size_px,
        marginSize=0,
        borderBits=1,
    )
    return img


def save_png_and_pdf(img_gray, output_base, dpi):
    """
    Save board as PNG and PDF with DPI information.
    """

    png_path = OUTPUT_DIR / f"{output_base}.png"
    pdf_path = OUTPUT_DIR / f"{output_base}.pdf"

    # Convert OpenCV grayscale image to PIL image
    pil_img = Image.fromarray(img_gray)

    # Save PNG with DPI metadata
    pil_img.save(png_path, dpi=(dpi, dpi))

    # Save PDF at correct physical scale
    pil_img.save(pdf_path, "PDF", resolution=dpi)

    print("")
    print("Saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    print("")
    print("Print instructions:")
    print("  - Print the PDF at 100% scale.")
    print("  - Disable 'fit to page'.")
    print("  - After printing, measure one square.")
    print("  - It should be 40 mm.")
    print("")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)

    board_width_px = mm_to_pixels(BOARD_WIDTH_MM, DPI)
    board_height_px = mm_to_pixels(BOARD_HEIGHT_MM, DPI)

    image_size_px = (board_width_px, board_height_px)

    board = create_charuco_board(dictionary)
    img = draw_charuco_board(board, image_size_px)

    save_png_and_pdf(img, OUTPUT_NAME, DPI)

    print("Board parameters to use in calibration code:")
    print(f"  SQUARES_X = {SQUARES_X}")
    print(f"  SQUARES_Y = {SQUARES_Y}")
    print(f"  SQUARE_LENGTH_M = {SQUARE_LENGTH_MM / 1000.0}")
    print(f"  MARKER_LENGTH_M = {MARKER_LENGTH_MM / 1000.0}")
    print("  ARUCO_DICT = cv2.aruco.DICT_4X4_50")


if __name__ == "__main__":
    main()