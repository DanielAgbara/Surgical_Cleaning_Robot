#!/usr/bin/env python3

import cv2
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from PIL import Image
import io


OUT_PDF = Path("aruco_grid_3x3_9markers_66mm_each.pdf")

GRID_ROWS = 3
GRID_COLS = 3

MARKER_SIZE_MM = 61.5
MARKER_GAP_MM = 3.72

BOARD_WIDTH_MM = GRID_COLS * MARKER_SIZE_MM + (GRID_COLS - 1) * MARKER_GAP_MM
BOARD_HEIGHT_MM = GRID_ROWS * MARKER_SIZE_MM + (GRID_ROWS - 1) * MARKER_GAP_MM

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)


def marker_to_png_bytes(marker_id, marker_px=2000):
    marker_img = cv2.aruco.generateImageMarker(
        DICTIONARY,
        marker_id,
        marker_px,
    )

    pil_img = Image.fromarray(marker_img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer


def generate_pdf():
    c = canvas.Canvas(
        str(OUT_PDF),
        pagesize=(BOARD_WIDTH_MM * mm, BOARD_HEIGHT_MM * mm),
    )

    marker_id = 0

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x = col * (MARKER_SIZE_MM + MARKER_GAP_MM)

            # ReportLab PDF origin is bottom-left, so flip y.
            y_top = row * (MARKER_SIZE_MM + MARKER_GAP_MM)
            y = BOARD_HEIGHT_MM - y_top - MARKER_SIZE_MM

            marker_buffer = marker_to_png_bytes(marker_id)
            marker_reader = ImageReader(marker_buffer)

            c.drawImage(
                marker_reader,
                x * mm,
                y * mm,
                MARKER_SIZE_MM * mm,
                MARKER_SIZE_MM * mm,
                mask="auto",
            )

            marker_id += 1

    c.save()


def main():
    generate_pdf()

    print(f"Saved PDF: {OUT_PDF.resolve()}")
    print(f"Grid: {GRID_ROWS} rows x {GRID_COLS} cols")
    print(f"Total markers: {GRID_ROWS * GRID_COLS}")
    print(f"Marker size: {MARKER_SIZE_MM:.1f} mm")
    print(f"Marker gap: {MARKER_GAP_MM:.1f} mm")
    print(f"Pattern size: {BOARD_WIDTH_MM:.1f} mm x {BOARD_HEIGHT_MM:.1f} mm")


if __name__ == "__main__":
    main()