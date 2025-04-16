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

# import cv2
# import numpy as np
# import os

# def apply_exposure_adjustment(img, gamma_values=None, linear_factors=None):
#     exposure_images = []
#     filenames = []

#     # Gamma correction if gamma values are provided
#     if gamma_values is not None:
#         for gamma in gamma_values:
#             gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
#             exposure_images.append(cv2.LUT(img, gamma_table))
#             filenames.append(f"{base_filename}_1.png")  # Gamma-corrected image at _1

#     # Original image as _2
#     exposure_images.append(img)
#     filenames.append(f"{base_filename}_2.png")

#     # Linear exposure adjustment if linear factors are provided
#     if linear_factors is not None:
#         for factor in linear_factors:
#             exposure_images.append(np.clip(img * factor, 0, 255).astype(np.uint8))
#             filenames.append(f"{base_filename}_3.png")  # Linear adjusted image at _3

#     return exposure_images, filenames

# def save_images(images, filenames, output_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     for img, filename in zip(images, filenames):
#         save_path = os.path.join(output_dir, filename)
#         cv2.imwrite(save_path, img)

# # Parameters
# input_image_path = r'C:\DatasetAWB\CWCC_outside\Exposure\DSC03572.png'
# output_dir = r'C:\DatasetAWB\CWCC_outside\Exposure\Final'
# base_filename = 'DSC03572'

# # Load image
# image = cv2.imread(input_image_path)

# # Define gamma values and linear factors for different exposure adjustments
# gamma_values = [2.5]      # Example gamma value for gamma correction
# linear_factors = [2.0]    # Example factor for linear exposure adjustment

# # Adjust exposure and get filenames
# exposure_images, filenames = apply_exposure_adjustment(image, gamma_values=gamma_values, linear_factors=linear_factors)

# # Save adjusted images with specific filenames
# save_images(exposure_images, filenames, output_dir)

# cv2.destroyAllWindows()

import os
import cv2
import numpy as np


def apply_exposure_adjustment(img, gamma_values=[2.2], linear_factors=[1.5]):
    """
    Áp dụng các điều chỉnh gamma và tuyến tính cho ảnh đầu vào.
    Args:
        img (np.ndarray): Ảnh gốc từ người dùng.
        gamma_values (list): Danh sách các giá trị gamma để điều chỉnh ảnh tối.
        linear_factors (list): Danh sách các hệ số để điều chỉnh ảnh sáng.
    Returns:
        exposure_images (list): Danh sách chứa các ảnh đã điều chỉnh.
        filenames (list): Danh sách tên các ảnh.
    """
    if img is None:
        raise ValueError("Input image is None. Please check the image path or format.")

    exposure_images = []
    filenames = []

    # Điều chỉnh gamma (ảnh tối hơn)
    for gamma in gamma_values:
        gamma_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
        adjusted_img = cv2.LUT(img, gamma_table)
        exposure_images.append(adjusted_img)
        filenames.append(f"gamma_{gamma:.2f}.png")

    # Điều chỉnh tuyến tính (ảnh sáng hơn)
    for factor in linear_factors:
        adjusted_img = cv2.convertScaleAbs(img, alpha=factor)
        exposure_images.append(adjusted_img)
        filenames.append(f"linear_{factor:.2f}.png")

    return exposure_images, filenames


def save_images(images, filenames, output_dir):
    """
    Lưu các ảnh đã điều chỉnh vào thư mục.
    Args:
        images (list): Danh sách các ảnh cần lưu.
        filenames (list): Danh sách các tên file tương ứng.
        output_dir (str): Đường dẫn thư mục lưu ảnh.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for img, fname in zip(images, filenames):
        path = os.path.join(output_dir, fname)
        cv2.imwrite(path, img)
        saved_paths.append(path)

    return saved_paths


def process_image_pipeline(input_image_path, output_dir, gamma_values=[2.2], linear_factors=[1.5]):
    """
    Pipeline xử lý ảnh: nhận ảnh đầu vào, tạo ảnh điều chỉnh và lưu.
    Args:
        input_image_path (str): Đường dẫn ảnh gốc.
        output_dir (str): Thư mục lưu ảnh.
        gamma_values (list): Giá trị gamma cho filter tối.
        linear_factors (list): Giá trị tuyến tính cho filter sáng.
    Returns:
        all_images (list): Danh sách đường dẫn đến 3 ảnh (gốc, tối, sáng).
    """
    # Đọc ảnh gốc
    img = cv2.imread(input_image_path)
    if img is None:
        raise ValueError(f"Image at {input_image_path} could not be read.")

    # Tạo các ảnh điều chỉnh (tối và sáng)
    exposure_images, filenames = apply_exposure_adjustment(img, gamma_values=gamma_values, linear_factors=linear_factors)

    # Lưu các ảnh điều chỉnh
    saved_paths = save_images(exposure_images, filenames, output_dir)

    # Lưu ảnh gốc vào danh sách
    base_name = os.path.basename(input_image_path)
    original_path = os.path.join(output_dir, base_name)
    cv2.imwrite(original_path, img)

    # Danh sách tất cả các ảnh (gốc, tối, sáng)
    all_images = [original_path] + saved_paths
    return all_images
