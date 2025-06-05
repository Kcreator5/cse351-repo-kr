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
from cse351 import *
 # print("Current working directory:", os.getcwd())  # trebleshooting 
import threading
import queue
import os
import cv2
import numpy as np

# Folders
INPUT_FOLDER = "faces"
STEP1_OUTPUT_FOLDER = "step1_smoothed"
STEP2_OUTPUT_FOLDER = "step2_grayscale"
STEP3_OUTPUT_FOLDER = "step3_edges"

# Parameters for image processing
GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5)
CANNY_THRESHOLD1 = 75
CANNY_THRESHOLD2 = 155

# Allowed image extensions
ALLOWED_EXTENSIONS = ['.jpg']



# workers. 


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
    queue_smooth = queue.Queue()
    queue_gray = queue.Queue()
    queue_edges = queue.Queue()
    # - create barriers
    # - create the three processes groups
    def smooth_worker():
        while True:
            filename = queue_smooth.get()
            try:
                if filename is None:
                    break
                path = os.path.join(INPUT_FOLDER, filename)
                image = cv2.imread(path)
                if image is None:
                    print(f"[Smooth] Couldn't load: {filename}")
                    continue
                result = cv2.GaussianBlur(image, (5, 5), 0)
                queue_gray.put((filename, result))
                print(f"[Smooth] Processed: {filename}")
            except Exception as e:
                print(f"[Smooth] Error with {filename}: {e}")
            finally:
                queue_smooth.task_done()


    def grayscale_worker():
        while True:
            item = queue_gray.get()
            try:
                if item is None:
                    break
                filename, image = item
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                queue_edges.put((filename, gray))
                print(f"[Gray] Converted: {filename}")
            except Exception as e:
                print(f"[Gray] Error with {filename}: {e}")
            finally:
                queue_gray.task_done()

    def edge_worker():
        while True:
            item = queue_edges.get()
            try:
                if item is None:
                    break
                filename, image = item
                edges = cv2.Canny(image, 75, 155)
                out_path = os.path.join(STEP3_OUTPUT_FOLDER, filename)
                cv2.imwrite(out_path, edges)
                print(f"[Edge] Saved: {filename}")
            except Exception as e:
                print(f"[Edge] Error with {filename}: {e}")
            finally:
                queue_edges.task_done()

    
    # Create output folders
    for folder in [STEP1_OUTPUT_FOLDER, STEP2_OUTPUT_FOLDER, STEP3_OUTPUT_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Start threads
    for _ in range(4): threading.Thread(target=smooth_worker, daemon=True).start()
    for _ in range(4): threading.Thread(target=grayscale_worker, daemon=True).start()
    for _ in range(4): threading.Thread(target=edge_worker, daemon=True).start()

    # Enqueue all filenames to the first queue
    for file in os.listdir(INPUT_FOLDER):
        if file.lower().endswith(".jpg"):
            queue_smooth.put(file)

    # Wait for smoothers to finish
    queue_smooth.join()
    for _ in range(4): queue_gray.put(None)

    # Wait for grayscale to finish
    queue_gray.join()
    for _ in range(4): queue_edges.put(None)

    # Wait for edge detection to finish
    queue_edges.join()

    print("Pipeline completed.")

# ---------------------------------------------------------------------------
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    # - you are free to change anything in the program as long as you
    #   do all requirements.

   # Create output folders
    for folder in [STEP1_OUTPUT_FOLDER, STEP2_OUTPUT_FOLDER, STEP3_OUTPUT_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    
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
