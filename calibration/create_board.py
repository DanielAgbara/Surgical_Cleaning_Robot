#!/usr/bin/env python3

import argparse
from pathlib import Path
import cv2


def mm_to_pixels(mm, dpi):
    return int(round((mm / 25.4) * dpi))


def get_dictionary(name):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown dictionary: {name}")

    return cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, name)
    )


def generate_charuco(args):
    dictionary = get_dictionary(args.dictionary)

    square_length_m = args.square_length_mm / 1000.0
    marker_length_m = args.marker_length_mm / 1000.0

    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        square_length_m,
        marker_length_m,
        dictionary,
    )

    board_width_mm = args.squares_x * args.square_length_mm
    board_height_mm = args.squares_y * args.square_length_mm

    return board, board_width_mm, board_height_mm


def generate_aruco(args):
    dictionary = get_dictionary(args.dictionary)

    marker_length_m = args.marker_length_mm / 1000.0
    marker_separation_m = args.marker_separation_mm / 1000.0

    board = cv2.aruco.GridBoard(
        (args.markers_x, args.markers_y),
        marker_length_m,
        marker_separation_m,
        dictionary,
    )

    board_width_mm = (
        args.markers_x * args.marker_length_mm
        + (args.markers_x - 1) * args.marker_separation_mm
    )

    board_height_mm = (
        args.markers_y * args.marker_length_mm
        + (args.markers_y - 1) * args.marker_separation_mm
    )

    return board, board_width_mm, board_height_mm


def main():
    parser = argparse.ArgumentParser(
        description="Generate ArUco or ChArUco calibration boards."
    )

    parser.add_argument(
        "--type",
        choices=["charuco", "aruco"],
        default="charuco",
        help="Board type to generate.",
    )

    parser.add_argument(
        "--dictionary",
        default="DICT_5X5_100",
        help="OpenCV ArUco dictionary name.",
    )

    parser.add_argument(
        "--squares-x",
        type=int,
        default=10,
        help="ChArUco squares in X direction.",
    )

    parser.add_argument(
        "--squares-y",
        type=int,
        default=10,
        help="ChArUco squares in Y direction.",
    )

    parser.add_argument(
        "--markers-x",
        type=int,
        default=2,
        help="ArUco markers in X direction.",
    )

    parser.add_argument(
        "--markers-y",
        type=int,
        default=2,
        help="ArUco markers in Y direction.",
    )

    parser.add_argument(
        "--square-length-mm",
        type=float,
        default=20.0,
        help="ChArUco square size in millimeters.",
    )

    parser.add_argument(
        "--marker-length-mm",
        type=float,
        default=14.0,
        help="Marker size in millimeters.",
    )

    parser.add_argument(
        "--marker-separation-mm",
        type=float,
        default=40.0,
        help="ArUco marker separation in millimeters.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output print DPI.",
    )

    parser.add_argument(
        "--margin-mm",
        type=float,
        default=10.0,
        help="White margin around board in millimeters.",
    )

    parser.add_argument(
        "--border-bits",
        type=int,
        default=1,
        help="ArUco marker border bits.",
    )

    parser.add_argument(
        "--output",
        default="calibration_board.png",
        help="Output image file.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Preview generated board.",
    )

    args = parser.parse_args()

    if args.type == "charuco":
        board, board_width_mm, board_height_mm = generate_charuco(args)
    else:
        board, board_width_mm, board_height_mm = generate_aruco(args)

    image_width_px = mm_to_pixels(
        board_width_mm + 2 * args.margin_mm,
        args.dpi,
    )

    image_height_px = mm_to_pixels(
        board_height_mm + 2 * args.margin_mm,
        args.dpi,
    )

    margin_px = mm_to_pixels(args.margin_mm, args.dpi)

    img = board.generateImage(
        (image_width_px, image_height_px),
        marginSize=margin_px,
        borderBits=args.border_bits,
    )

    out_path = Path(args.output)
    cv2.imwrite(str(out_path), img)

    print("\nSaved:", out_path.resolve())
    print("Board type:", args.type)
    print(f"Board pattern size: {board_width_mm:.1f} mm x {board_height_mm:.1f} mm")
    print(f"Margin: {args.margin_mm:.1f} mm")
    print(f"Image size: {image_width_px} px x {image_height_px} px")
    print(f"DPI: {args.dpi}")

    if args.show:
        cv2.imshow("Calibration Board", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()