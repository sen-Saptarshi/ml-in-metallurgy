#!/usr/bin/env python3

"""
A script to normalize a folder of images by matching their histograms
to a single reference image.

This version interactively prompts the user for the required paths
when the script is run.
"""

import cv2
import numpy as np
import os
import glob
import sys
from skimage.exposure import match_histograms

def process_images(reference_path, input_dir, output_dir):
    """
    Matches the histogram of all images in input_dir to the
    histogram of the reference_path image and saves them in output_dir.
    """
    
    # --- 1. Load Reference Image ---
    print(f"\nLoading reference image from: {reference_path}")
    try:
        img_reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
        if img_reference is None:
            # This handles the case where the file exists but is not a valid image
            raise IOError(f"File {reference_path} is not a valid image or is corrupt.")
    except Exception as e:
        print(f"--- FATAL ERROR ---")
        print(f"Could not load reference image. Check path and file integrity.")
        print(f"Details: {e}")
        sys.exit(1) # Exit the script if the reference can't be loaded

    print("Reference image loaded successfully.")

    # --- 2. Create Output Directory ---
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"--- FATAL ERROR ---")
        print(f"Could not create output directory: {output_dir}")
        print(f"Details: {e}")
        sys.exit(1)

    # --- 3. Find Images to Process ---
    # Find all common image types, not just .png
    supported_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    image_paths = []
    for ext in supported_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_paths:
        print(f"--- WARNING ---")
        print(f"No images found in {input_dir}. Exiting.")
        sys.exit(0)

    print(f"Found {len(image_paths)} images. Processing...")

    # --- 4. Process and Save Images ---
    for img_path in image_paths:
        try:
            # Read the image to be processed
            img_to_process = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img_to_process is None:
                print(f"Warning: Could not read {img_path}, skipping.")
                continue
            
            # Match the histogram to the reference
            img_matched = match_histograms(img_to_process, img_reference)
            
            # Get the original filename
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, filename)
            
            # Save the processed image
            # Note: matched image is float, convert back to uint8
            cv2.imwrite(output_path, img_matched.astype(np.uint8))
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    print("\nDone! All images have been matched and saved.")
    print(f"Your processed dataset is in: {output_dir}")


if __name__ == "__main__":
    # --- 5. Prompt user for paths ---
    print("--- Image Histogram Normalizer ---")
    print("Please provide the following paths. You can drag and drop folders/files into the terminal.")
    
    # Get the reference image path
    reference_path = input("Enter the full path to your REFERENCE (template) image: ").strip()
    
    # Get the input directory path
    input_dir = input("Enter the path to the INPUT folder (images to process): ").strip()
    
    # Get the output directory path
    output_dir = input("Enter the path for the OUTPUT folder (to save results): ").strip()
    
    # Run the main processing function
    process_images(reference_path, input_dir, output_dir)