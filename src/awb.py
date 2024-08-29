import cv2
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time
import os

# Đọc ảnh
# image = cv2.imread('/content/86_output.png')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Chuyển đổi không gian màu sang không gian sắc độ
def convert_to_chromaticity_space(image):
    R = cp.asarray(image[:, :, 0], dtype=cp.float32)
    G = cp.asarray(image[:, :, 1], dtype=cp.float32)
    B = cp.asarray(image[:, :, 2], dtype=cp.float32)

    # Tính toán các giá trị r, g, b
    sum_RGB = R + G + B
    sum_RGB[sum_RGB == 0] = 1e-6  # Tránh chia cho 0

    r = R / sum_RGB
    g = G / sum_RGB
    b = B / sum_RGB

    return r, g, b

# start_time = time.time()
# r, g, b = convert_to_chromaticity_space(image)
# end_time = time.time()
# print(f"Time for converting to chromaticity space: {end_time - start_time:.4f} seconds")

# Tạo histogram sắc độ
def compute_chromaticity_histogram(r, g, b, bins=256):
    hist_r, _ = cp.histogram(r, bins=bins, range=(0, 1))
    hist_g, _ = cp.histogram(g, bins=bins, range=(0, 1))
    hist_b, _ = cp.histogram(b, bins=bins, range=(0, 1))

    return hist_r, hist_g, hist_b

# start_time = time.time()
# hist_r, hist_g, hist_b = compute_chromaticity_histogram(r, g, b)
# end_time = time.time()
# print(f"Time for computing chromaticity histogram: {end_time - start_time:.4f} seconds")

# Tính diện tích giao nhau của histogram sắc độ
def compute_overlap_area(hist_r, hist_g, hist_b):
    overlap_area = cp.sum(cp.minimum(cp.minimum(hist_r, hist_g), hist_b))
    return overlap_area

# start_time = time.time()
# overlap_area = compute_overlap_area(hist_r, hist_g, hist_b)
# end_time = time.time()
# print(f"Time for computing overlap area: {end_time - start_time:.4f} seconds")

# print("Overlap Area: ", overlap_area.get())

# Tìm các hệ số tối ưu cho kênh R, G, B
def find_optimal_gains(r, g, b, hist_r, hist_g, hist_b):
    max_overlap = 0
    optimal_kr, optimal_kg, optimal_kb = 1, 1, 1

    kr_values = cp.linspace(0.5, 1.5, 20)
    kg_values = cp.linspace(0.5, 1.5, 20)
    kb_values = cp.linspace(0.5, 1.5, 20)

    for kr in tqdm(kr_values, position=0, leave=False):
        for kg in kg_values:
            for kb in kb_values:
                adjusted_r = cp.clip(r * kr, 0, 1)
                adjusted_g = cp.clip(g * kg, 0, 1)
                adjusted_b = cp.clip(b * kb, 0, 1)

                hist_r_adj, hist_g_adj, hist_b_adj = compute_chromaticity_histogram(adjusted_r, adjusted_g, adjusted_b)
                overlap = compute_overlap_area(hist_r_adj, hist_g_adj, hist_b_adj)

                if overlap > max_overlap:
                    max_overlap = overlap
                    optimal_kr, optimal_kg, optimal_kb = kr, kg, kb

    return optimal_kr, optimal_kg, optimal_kb

# start_time = time.time()
# optimal_kr, optimal_kg, optimal_kb = find_optimal_gains(r, g, b, hist_r, hist_g, hist_b)
# end_time = time.time()
# print(f"Time for finding optimal gains: {end_time - start_time:.4f} seconds")

# print("Optimal gains: ", optimal_kr, optimal_kg, optimal_kb)

# Điều chỉnh ảnh theo các hệ số tối ưu và hiển thị ảnh
def apply_awb(image, kr, kg, kb):
    adjusted_image = image.copy()
    adjusted_image[:, :, 0] = np.clip(image[:, :, 0] * kr.get(), 0, 255)
    adjusted_image[:, :, 1] = np.clip(image[:, :, 1] * kg.get(), 0, 255)
    adjusted_image[:, :, 2] = np.clip(image[:, :, 2] * kb.get(), 0, 255)

    return adjusted_image

# start_time = time.time()
# awb_image = apply_awb(image, optimal_kr, optimal_kg, optimal_kb)
# awb_image = awb_image.astype(np.uint8)
# end_time = time.time()
# print(f"Time for applying AWB: {end_time - start_time:.4f} seconds")

def main(image):
    r, g, b = convert_to_chromaticity_space(image)
    hist_r, hist_g, hist_b = compute_chromaticity_histogram(r, g, b)
    # overlap_area = compute_overlap_area(hist_r, hist_g, hist_b)
    optimal_kr, optimal_kg, optimal_kb = find_optimal_gains(r, g, b, hist_r, hist_g, hist_b)
    awb_image = apply_awb(image, optimal_kr, optimal_kg, optimal_kb)
    awb_image = awb_image.astype(np.uint8)
    return awb_image

# Hiển thị ảnh trước và sau khi áp dụng AWB
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(image)

# plt.subplot(1, 2, 2)
# plt.title('AWB Image')
# plt.imshow(awb_image)

# plt.show()

# Lưu ảnh sau khi áp dụng AWB
# cv2.imwrite('/content/new_86_output.png', cv2.cvtColor(awb_image, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    img_folder = "datahub/results/sample-epoch=94_new_v4"
    for img in os.listdir(img_folder):
        # Read image
        image = cv2.imread(f"{img_folder}/{img}/{img}_output.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Apply AWB
        awb_img = main(image)
        # Save image
        cv2.imwrite(f"{img_folder}/{img}/{img}_output_awb.png", cv2.cvtColor(awb_img, cv2.COLOR_RGB2BGR))