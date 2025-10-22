#!/usr/bin/env python3
"""
Crop Tool for SEM/Optical Images with Scale Bar
-----------------------------------------------
Usage:
    python crop_tool.py \
        --image_path "./Ta5Al4/as_cast/10um/S1_09.tif" \
        --actual_length_um 10.0 \
        --target_crop_um 50.0 \
        --target_pixel_size 128 \
        --output_dir "./patches"

Click anywhere on the image to extract patches of target_crop_um × target_crop_um.
Press 'q' in the window or close it when finished.
"""

import argparse
import os
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from datetime import datetime


def detect_scale_bar(image_bgr):
    """Detect green scale bar in the image. Returns pixel length or 0."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return w


def interactive_crop(image_rgb, crop_size_px, target_pixel_size, output_dir):
    """Interactive Tkinter GUI for selecting patches and saving them."""

    patches_saved = 0

    # Convert to PIL image for display
    img_pil = Image.fromarray(image_rgb)

    root = tk.Tk()
    root.title("Crop Tool — Click to Save Patches. Press Q to Quit")

    img_tk = ImageTk.PhotoImage(img_pil, master=root)
    canvas = tk.Canvas(root, width=img_pil.width, height=img_pil.height)
    canvas.pack()
    canvas.create_image(0, 0, anchor="nw", image=img_tk)

    rect = canvas.create_rectangle(0, 0, crop_size_px, crop_size_px, outline="red", width=2)

    def on_motion(event):
        x, y = event.x, event.y
        canvas.coords(rect, x, y, x + crop_size_px, y + crop_size_px)

    def on_click(event):
        nonlocal patches_saved
        x, y = int(event.x), int(event.y)

        # Compute crop boundaries safely
        y_end = min(y + crop_size_px, image_rgb.shape[0])
        x_end = min(x + crop_size_px, image_rgb.shape[1])

        crop = image_rgb[y:y_end, x:x_end, :]
        crop_resized = cv2.resize(crop, (target_pixel_size, target_pixel_size), interpolation=cv2.INTER_AREA)

        filename = f"patch_{patches_saved+1:03d}_{datetime.now().strftime('%H%M%S')}.png"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR))

        patches_saved += 1
        print(f"Saved: {save_path}")

    def on_key(event):
        key = event.keysym.lower()
        if key == 'q':
            print(f"Total patches saved: {patches_saved}")
            root.destroy()

    canvas.bind("<Motion>", on_motion)
    canvas.bind("<Button-1>", on_click)
    root.bind("<Key>", on_key)

    print("Instructions:")
    print(" • Move mouse to position the red box.")
    print(" • Left-click to save a patch.")
    print(" • Press 'q' or close window to exit.")
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Crop Tool with GUI patch selection")
    parser.add_argument("--image_path", required=True, help="Path to input image file")
    parser.add_argument("--actual_length_um", type=float, required=True, help="Actual length of scale bar in microns")
    parser.add_argument("--target_crop_um", type=float, required=True, help="Target crop size in microns")
    parser.add_argument("--target_pixel_size", type=int, default=128, help="Final crop size in pixels (default 128)")
    parser.add_argument("--output_dir", required=True, help="Folder to save cropped patches")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {args.image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect scale bar
    pixel_length_of_bar = detect_scale_bar(image)
    if pixel_length_of_bar <= 0:
        print("Warning: No green scale bar detected. Using default 1 px/µm.")
        pixels_per_um = 1.0
    else:
        pixels_per_um = pixel_length_of_bar / args.actual_length_um
        print(f"Detected scale bar: {pixel_length_of_bar:.2f} px  →  {pixels_per_um:.2f} px/µm")

    crop_size_px = int(args.target_crop_um * pixels_per_um)
    print(f"Crop box size: {crop_size_px}×{crop_size_px} pixels")

    interactive_crop(image_rgb, crop_size_px, args.target_pixel_size, args.output_dir)


if __name__ == "__main__":
    main()
