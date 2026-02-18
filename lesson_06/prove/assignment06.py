"""
Course: CSE 351
Assignment: 06
Author: [Kevin Rogers]

Instructions:

- see instructions in the assignment description in Canvas

""" 

import multiprocessing as mp
import os
import cv2
import numpy as np
from collections import deque
from cse351 import *

# Folders
INPUT_FOLDER = "faces"
STEP1_OUTPUT_FOLDER = "step1_smoothed"
STEP2_OUTPUT_FOLDER = "step2_grayscale"
STEP3_OUTPUT_FOLDER = "step3_edges"

# Parameters for image processing
GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5)
CANNY_THRESHOLD1 = 75
CANNY_THRESHOLD2 = 155

SENTINEL = None

# Allowed image extensions
ALLOWED_EXTENSIONS = ['.jpg']

# ---------------------------------------------------------------------------
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")

# --------------------------------------------------------------------------- New stuff
def worker_smooth(q_in_files, q_out_imgs, kernel_size):
    create_folder_if_not_exists(STEP1_OUTPUT_FOLDER)
    """
    Reads filenames, loads images, smooths them, saves step1 output,
    and passes (filename, smoothed_image) to the next stage.
    """

    while True:
        filename = q_in_files.get()
        if filename is SENTINEL:
            # Tell next stage to stop too
            q_out_imgs.put(SENTINEL)
            break

        input_path = os.path.join(INPUT_FOLDER, filename)
        img = cv2.imread(input_path)
        if img is None:
            print(f"Warning: Could not read image '{input_path}'. Skipping.")
            continue

        smoothed = task_smooth_image(img, kernel_size)
        cv2.imwrite(os.path.join(STEP1_OUTPUT_FOLDER, filename), smoothed)

        # pass forward in-memory to reduce re-reads
        q_out_imgs.put((filename, smoothed))


def worker_grayscale(q_in_imgs, q_out_gray):
    """
    Receives (filename, image), converts to grayscale, saves step2 output,
    and passes (filename, grayscale_image) to the next stage.
    """
    create_folder_if_not_exists(STEP2_OUTPUT_FOLDER)

    while True:
        item = q_in_imgs.get()
        if item is SENTINEL:
            q_out_gray.put(SENTINEL)
            break

        filename, img = item
        gray = task_convert_to_grayscale(img)
        cv2.imwrite(os.path.join(STEP2_OUTPUT_FOLDER, filename), gray)

        q_out_gray.put((filename, gray))


def worker_edges(q_in_gray, threshold1, threshold2):
    """
    Receives (filename, grayscale_image), runs Canny, saves step3 output.
    """
    create_folder_if_not_exists(STEP3_OUTPUT_FOLDER)

    while True:
        item = q_in_gray.get()
        if item is SENTINEL:
            break

        filename, gray = item
        edges = task_detect_edges(gray, threshold1, threshold2)
        cv2.imwrite(os.path.join(STEP3_OUTPUT_FOLDER, filename), edges)

# ---------------------------------------------------------------------------
def task_convert_to_grayscale(image):
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        return image # Already grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------------------------------
def task_smooth_image(image, kernel_size):
    return cv2.GaussianBlur(image, kernel_size, 0)

# ---------------------------------------------------------------------------
def task_detect_edges(image, threshold1, threshold2):
    if len(image.shape) == 3 and image.shape[2] == 3:
        print("Warning: Applying Canny to a 3-channel image. Converting to grayscale first for Canny.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] != 1 : # Should not happen with typical images
        print(f"Warning: Input image for Canny has an unexpected number of channels: {image.shape[2]}")
        return image # Or raise error
    return cv2.Canny(image, threshold1, threshold2)

# ---------------------------------------------------------------------------
def process_images_in_folder(input_folder,              # input folder with images
                             output_folder,             # output folder for processed images
                             processing_function,       # function to process the image (ie., task_...())
                             load_args=None,            # Optional args for cv2.imread
                             processing_args=None):     # Optional args for processing function

    create_folder_if_not_exists(output_folder)
    print(f"\nProcessing images from '{input_folder}' to '{output_folder}'...")

    processed_count = 0
    for filename in os.listdir(input_folder):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            continue

        input_image_path = os.path.join(input_folder, filename)
        output_image_path = os.path.join(output_folder, filename) # Keep original filename

        try:
            # Read the image
            if load_args is not None:
                img = cv2.imread(input_image_path, load_args)
            else:
                img = cv2.imread(input_image_path)

            if img is None:
                print(f"Warning: Could not read image '{input_image_path}'. Skipping.")
                continue

            # Apply the processing function
            if processing_args:
                processed_img = processing_function(img, *processing_args)
            else:
                processed_img = processing_function(img)

            # Save the processed image
            cv2.imwrite(output_image_path, processed_img)

            processed_count += 1
        except Exception as e:
            print(f"Error processing file '{input_image_path}': {e}")

    print(f"Finished processing. {processed_count} images processed into '{output_folder}'.")

# ---------------------------------------------------------------------------
def run_image_processing_pipeline():
    print("Starting image processing pipeline...")

    # TODO
    # - create queues
    # - create barriers
    # - create the three processes groups
    # - you are free to change anything in the program as long as you
    #   do all requirements.

    # 1) Build list of input .jpg files
    files = []
    for filename in os.listdir(INPUT_FOLDER):
        ext = os.path.splitext(filename)[1].lower()
        if ext in ALLOWED_EXTENSIONS:
            files.append(filename)

    if not files:
        print(f"No {ALLOWED_EXTENSIONS} files found in '{INPUT_FOLDER}'.")
        return

    # 2) Create queues between stages
    q_files = mp.Queue(maxsize=10)   # filenames -> smooth
    q_imgs  = mp.Queue(maxsize=10)   # smoothed images -> grayscale
    q_gray  = mp.Queue(maxsize=10)   # grayscale images -> edges

    # 3) Start the 3 processes (one per stage)
    p1 = mp.Process(target=worker_smooth,   args=(q_files, q_imgs, GAUSSIAN_BLUR_KERNEL_SIZE))
    p2 = mp.Process(target=worker_grayscale,args=(q_imgs, q_gray))
    p3 = mp.Process(target=worker_edges,    args=(q_gray, CANNY_THRESHOLD1, CANNY_THRESHOLD2))

    p1.start()
    p2.start()
    p3.start()

    # 4) Feed filenames into stage 1
    for f in files:
        q_files.put(f)

    # 5) Stop the pipeline: send ONE sentinel to stage 1
    q_files.put(SENTINEL)

    # 6) Wait for all processes to finish
    p1.join()
    p2.join()
    p3.join()

    # --- Step 1: Smooth Images ---
    process_images_in_folder(INPUT_FOLDER, STEP1_OUTPUT_FOLDER, task_smooth_image,
                             processing_args=(GAUSSIAN_BLUR_KERNEL_SIZE,))

    # --- Step 2: Convert to Grayscale ---
    process_images_in_folder(STEP1_OUTPUT_FOLDER, STEP2_OUTPUT_FOLDER, task_convert_to_grayscale)

    # --- Step 3: Detect Edges ---
    process_images_in_folder(STEP2_OUTPUT_FOLDER, STEP3_OUTPUT_FOLDER, task_detect_edges,
                             load_args=cv2.IMREAD_GRAYSCALE,        
                             processing_args=(CANNY_THRESHOLD1, CANNY_THRESHOLD2))

    print("\nImage processing pipeline finished!")
    print(f"Original images are in: '{INPUT_FOLDER}'")
    print(f"Grayscale images are in: '{STEP1_OUTPUT_FOLDER}'")
    print(f"Smoothed images are in: '{STEP2_OUTPUT_FOLDER}'")
    print(f"Edge images are in: '{STEP3_OUTPUT_FOLDER}'")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log = Log(show_terminal=True)
    log.start_timer('Processing Images')

    # check for input folder
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Error: The input folder '{INPUT_FOLDER}' was not found.")
        print(f"Create it and place your face images inside it.")
        print('Link to faces.zip:')
        print('   https://drive.google.com/file/d/1eebhLE51axpLZoU6s_Shtw1QNcXqtyHM/view?usp=sharing')
    else:
        run_image_processing_pipeline()

    log.write()
    log.stop_timer('Total Time To complete')
