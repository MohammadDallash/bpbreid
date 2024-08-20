import os
import cv2
import numpy as np
from Util import *

def merge_images_vertically(image_paths):
    # Read all images
    images = [cv2.imread(img_path) for img_path in image_paths]
    
    # Check if all images have the same width
    widths = [img.shape[1] for img in images]
    max_width = max(widths)
    
    # Resize images to the same width
    resized_images = [cv2.resize(img, (max_width, img.shape[0])) for img in images]
    
    # Stack images vertically
    merged_image = np.vstack(resized_images)
    
    return merged_image

def merge_images_from_folders(folders, output_folder):
    # Get list of image names (assuming all folders have the same images)
    image_names = os.listdir(folders[0])
    
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for image_name in image_names:
        image_paths = [os.path.join(folder, image_name) for folder in folders]
        
        # Merge images
        merged_image = merge_images_vertically(image_paths)
        
        # Save the merged image
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, merged_image)
        print(f"Saved merged image: {output_path}")

# Example usage
folders = ["outputs/occluded_duke", "outputs/market1501", "outputs/p_dukemtmc", "outputs/dukemtmcreid"]  # Replace with your folder paths
output_folder = "outputs/merged_output"
clear_or_create_folder(output_folder)
merge_images_from_folders(folders, output_folder)
