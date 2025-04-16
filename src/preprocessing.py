# # GAMMAR CORRECTION

# import cv2
# import numpy as np
# import os

# def gamma_correction(img, gamma):
#     # Build a lookup table for gamma correction
#     gamma_table = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
#     # Apply the gamma correction
#     return cv2.LUT(img, gamma_table)

# def adjust_exposure(img, gamma_values):
#     exposure_images = []
#     for gamma in gamma_values:
#         exposure_images.append(gamma_correction(img, gamma))
#     return exposure_images

# # Load your image
# image = cv2.imread(r'C:\DatasetAWB\CWCC_outside\Exposure\DSC03572.png')

# # Define gamma values for different exposure levels
# gamma_values = [2.5]  # Underexposed, correct, and overexposed

# # Get the exposure adjusted images
# exposure_images = adjust_exposure(image, gamma_values)

# # Define the directory to save images
# output_dir = r'C:\DatasetAWB\CWCC_outside\Exposure\Final'

# # Ensure the output directory exists
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Save the images to the specified directory
# for i, img in enumerate(exposure_images):
#     filename = f'DSC03572_1.png'
#     save_path = os.path.join(output_dir, filename)
#     cv2.imwrite(save_path, img)

# cv2.destroyAllWindows()

# # Linear Exposure Adjustment

# import cv2
# import numpy as np
# import os

# def linear_exposure(img, factor):
#     # Scale the image brightness
#     img = np.clip(img * factor, 0, 255)  # Ensure pixel values are between 0 and 255
#     return img.astype(np.uint8)

# def adjust_exposure_linear(img, factors):
#     exposure_images = []
#     for factor in factors:
#         exposure_images.append(linear_exposure(img, factor))
#     return exposure_images

# # Load your image
# image = cv2.imread(r'C:\DatasetAWB\CWCC_outside\Exposure\DSC03572.png')

# # Define factors for different exposure levels
# factors = [2.0]  # Underexposed, correct, and overexposed

# # Get the exposure adjusted images
# exposure_images = adjust_exposure_linear(image, factors)

# # Define the directory to save images
# output_dir = r'C:\DatasetAWB\CWCC_outside\Exposure\Final'

# # Ensure the output directory exists
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Save the images to the specified directory
# for i, img in enumerate(exposure_images):
#     filename = f'DSC03572_3.png'
#     save_path = os.path.join(output_dir, filename)
#     cv2.imwrite(save_path, img)

# cv2.destroyAllWindows()

import cv2
import numpy as np
import os

def apply_exposure_adjustment(img, gamma_values=None, linear_factors=None):
    exposure_images = []
    filenames = []

    # Gamma correction if gamma values are provided
    if gamma_values is not None:
        for gamma in gamma_values:
            gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
            exposure_images.append(cv2.LUT(img, gamma_table))
            filenames.append(f"{base_filename}_1.png")  # Gamma-corrected image at _1

    # Original image as _2
    exposure_images.append(img)
    filenames.append(f"{base_filename}_2.png")

    # Linear exposure adjustment if linear factors are provided
    if linear_factors is not None:
        for factor in linear_factors:
            exposure_images.append(np.clip(img * factor, 0, 255).astype(np.uint8))
            filenames.append(f"{base_filename}_3.png")  # Linear adjusted image at _3

    return exposure_images, filenames

def save_images(images, filenames, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for img, filename in zip(images, filenames):
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, img)

# Parameters
input_image_path = 'fusion_mertens.jpg'
output_dir = 'E:/refactor/refactor/src'
base_filename = '707'

# Load image
image = cv2.imread(input_image_path)

# Define gamma values and linear factors for different exposure adjustments
gamma_values = [2.5]      # Example gamma value for gamma correction
linear_factors = [2.0]    # Example factor for linear exposure adjustment

# Adjust exposure and get filenames
exposure_images, filenames = apply_exposure_adjustment(image, gamma_values=gamma_values, linear_factors=linear_factors)

# Save adjusted images with specific filenames
save_images(exposure_images, filenames, output_dir)


