import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pandas as pd
def increase_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Apply adaptive histogram equalization for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    contrast_image = clahe.apply(image)
    return contrast_image

def binarize_image(image, threshold_type=cv2.THRESH_BINARY + cv2.THRESH_OTSU):
    # Apply binary thresholding
    _, binary_image = cv2.threshold(image, 0, 255, threshold_type)
    return binary_image

def find_and_draw_dividers(image, row_threshold_factor=0.04):
    # Sum pixels along the horizontal axis (rows)
    horizontal_sum = np.sum(image == 0, axis=1)
    
    # Set a threshold to detect empty (or near-empty) rows
    row_threshold = image.shape[1] * row_threshold_factor  # Adjust threshold based on image width
    dividers = [i for i, count in enumerate(horizontal_sum) if count < row_threshold]

    # Create an image to display dividers
    display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for row in dividers:
        cv2.line(display_image, (0, row), (image.shape[1], row), (0, 0, 255), 1)  # Red line for dividers

    return display_image, dividers

def extract_main_line(image, dividers_in, width_weight=0.6, center_weight=0.4):
    # Identify the widest line or the middle line based on weighted criteria
    if not dividers_in:
        return image  # No dividers found, return the whole image

    dividers = [0]
    dividers.extend(dividers_in)
    dividers.append(image.shape[0])

    # Remove loner dividers
    dividers_nonloner = [dividers[i] for i in range(1, len(dividers) - 1)
                         if dividers[i] - dividers[i - 1] == 1 or dividers[i + 1] - dividers[i] == 1]

    dividers = [0]
    dividers.extend(dividers_nonloner)
    dividers.append(image.shape[0])

    line_boundaries = [(dividers[i], dividers[i + 1]) for i in range(len(dividers) - 1)]
    line_widths = [(end - start, start, end) for start, end in line_boundaries]

    # Find the widest line
    widest_line = max(line_widths, key=lambda x: x[0])

    # Find the line closest to the center
    image_center = (image.shape[0]+1) // 2
    center_line = min(line_widths, key=lambda x: abs((x[1] + x[2]) // 2 - image_center))

    # Weighted combination (adjust weights as needed)
    chosen_line = max(
        [widest_line, center_line],
        key=lambda x: x[0] * width_weight - abs((x[1] + x[2]) // 2 - image_center) * center_weight
    )

    return image[chosen_line[1]:chosen_line[2], :]

def crop_white_space(image):
    # Find the first and last columns that are not entirely white (non-background pixels)
    non_white_cols = np.where(image != 255)[1]  # Non-white pixel columns

    if non_white_cols.size == 0:
        # If no non-white pixels found, return the original image
        return image

    # Get the leftmost and rightmost non-white columns
    left = non_white_cols.min()
    right = non_white_cols.max()

    # Crop the image to remove the extra white space
    cropped_image = image[:, left:right+1]

    return cropped_image

def detect_colon(image, width_thresh=0.5, left_offset=4, right_offset=4,
                 left_white_threshold=2, right_white_threshold=2,
                 black_distance_threshold=5, top_black_ratio=1/3,
                 bottom_black_ratio=2/3, middle_offset=1):
    # Detect a colon in the image
    height, width = image.shape
    start_col = 0
    end_col = int(width_thresh * width)
    for col in range(start_col, end_col):
        if col - left_offset < 0 or col + right_offset + 1 >= width:
            continue  # Skip if offsets go out of bounds

        # Check if there is an all white column within left_offset pixels to the left AND right_offset pixels to the right
        left_white = np.all(image[:, col-left_offset:col] == 255, axis=0)
        right_white = np.all(image[:, col+1:col+right_offset+1] == 255, axis=0)

        if np.sum(left_white) > left_white_threshold and np.sum(right_white) > right_white_threshold:
            if np.sum(image[:, col] == 0) > 0:  # Check for black pixels in the column
                # Check for black pixels on top and bottom
                first_black_row = np.argmax(image[:, col] == 0)
                last_black_row = height - np.argmax(image[::-1, col] == 0) - 1

                if last_black_row - first_black_row > black_distance_threshold:
                    # Check if top and bottom parts are black with white in the middle (typical of a colon)
                    top_region = image[first_black_row:first_black_row+int(top_black_ratio*(last_black_row-first_black_row)), col]
                    bottom_region = image[first_black_row+int(bottom_black_ratio*(last_black_row-first_black_row)):last_black_row, col]
                    middle_region = image[first_black_row+int(top_black_ratio*(last_black_row-first_black_row))+middle_offset:
                                          first_black_row+int(bottom_black_ratio*(last_black_row-first_black_row))-middle_offset, col]

                    top_black = np.sum(top_region == 0) > 0
                    bottom_black = np.sum(bottom_region == 0) > 0
                    middle_white = np.sum(middle_region == 0) == 0

                    if top_black and bottom_black and middle_white:
                        return col  # Return the column index of the detected colon
    return None  # No colon detected

def segment_letters_contours_with_boxes(image):
    # Invert image: letters should be white, background black
    inv_image = cv2.bitwise_not(image)

    # Find contours
    contours, hierarchy = cv2.findContours(inv_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract bounding boxes
    letter_bboxes = []
    letters = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5:  # Filter small contours
            letter_bboxes.append((x, y, w, h))

    # Sort bounding boxes by x coordinate
    letter_bboxes = sorted(letter_bboxes, key=lambda b: b[0])

    # Extract letter images
    for bbox in letter_bboxes:
        x, y, w, h = bbox
        letter_image = image[y:y+h, x:x+w]
        letters.append(letter_image)

    return letters, letter_bboxes

def preview_directory(directory_path, output_base_path, label_csv_path, clip_limit=2.0, tile_grid_size=(8, 8),
                      row_threshold_factor=0.02, width_weight=0.6, center_weight=0.4):
    # Load the CSV file with labels
    labels_df = pd.read_csv(label_csv_path)

    # Create a dictionary for quick lookup of word identities
    filename_to_identity = dict(zip(labels_df['FILENAME'], labels_df['IDENTITY']))

    # List all image files in the directory
    image_files = [f for f in os.listdir(directory_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Create the base directory for saving outputs if it doesn't exist
    os.makedirs(output_base_path, exist_ok=True)
    count=0
    # Process images in the directory
    for image_name in image_files:
        image_path = os.path.join(directory_path, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Error loading image: {image_name}")
            continue  # Skip if the image is not loaded properly

        # Extract the image number from the filename
        image_number = image_name
        word_identity = filename_to_identity[image_name]
        # Get the word identity for this image
        if image_name not in filename_to_identity:
            print(f"No label found for {image_name}. Skipping...")
            continue
        if not isinstance(word_identity, str) or pd.isna(word_identity) or word_identity.upper() == "UNREADABLE":
            print(f"Invalid or unreadable label for {image_name}. Skipping...")
            continue
        word_identity = filename_to_identity[image_name]
        word_length = len(word_identity)

        # Increase contrast
        contrast_image = increase_contrast(image, clip_limit, tile_grid_size)

        # Binarize the image
        binary_image = binarize_image(contrast_image)

        # Find and draw horizontal dividers
        display_image_h, dividers_h = find_and_draw_dividers(binary_image, row_threshold_factor)

        # Extract the main horizontal line
        main_line_image = extract_main_line(binary_image, dividers_h, width_weight, center_weight)
        main_line_image = crop_white_space(main_line_image)

        # Detect colon
        colon_index = detect_colon(main_line_image)

        if colon_index is not None:
            # Crop the image to include only the part after the colon
            main_line_image = main_line_image[:, colon_index+1:]
            main_line_image = crop_white_space(main_line_image)  # Crop white space again after cropping
            # print(f"Colon detected in {image_name} at column {colon_index}. Processing text after colon.")
        # else:
            # No colon detected, proceed with the entire main line
            # print(f"No colon detected in {image_name}. Processing entire main line.")

        # Segment the main_line_image into letters
        letters, bounding_boxes = segment_letters_contours_with_boxes(main_line_image)

        # Check if the number of segmented letters matches the word length
        if len(letters) != word_length:
            count+=1
            # print(f"Number of segmented letters ({len(letters)}) does not match word length ({word_length}) for {image_name}. Skipping...")
            continue  # Skip saving if lengths do not match

        # Create a folder for the current word
        word_output_path = os.path.join(output_base_path, image_number)
        os.makedirs(word_output_path, exist_ok=True)

        # Save each letter as an image in the word's folder
        for idx, letter_image in enumerate(letters):
            letter_filename = f"letter_{idx + 1}.png"
            letter_path = os.path.join(word_output_path, letter_filename)
            cv2.imwrite(letter_path, letter_image)

        # print(f"Processed and saved letters for {image_name} in folder {word_output_path}.")
    print(count)
# Example usage
directory_path = 'C:/ELL409_Assignments/Project/validation_v2'  # Replace with your image directory path
output_base_path = 'C:\ELL409_Assignments\Project\Test_dataset'
label_csv_path = 'C:\ELL409_Assignments\Project\written_name_validation_v2.csv'
preview_directory(directory_path, output_base_path, label_csv_path)