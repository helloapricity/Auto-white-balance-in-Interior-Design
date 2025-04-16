# import cv2
# import matplotlib.pyplot as plt


# image_path = "1509_1.png"
# image_color = cv2.imread(image_path)

# # Chuyển sang không gian màu YUV
# yuv_image = cv2.cvtColor(image_color, cv2.COLOR_BGR2YUV)

# # Cân bằng histogram trên kênh Y (độ sáng)
# yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])

# # Chuyển về không gian màu BGR
# equalized_color_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
# cv2.imwrite("1509_equal.jpg", equalized_color_image)



import cv2
import matplotlib.pyplot as plt

# Đọc ảnh từ file
image_path = "1509_equal.jpg"
image = cv2.imread(image_path)

# Chuyển sang không gian màu LAB
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Áp dụng CLAHE trên kênh L (Lightness)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])

# Chuyển lại về không gian màu BGR
enhanced_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

# Hiển thị ảnh gốc và ảnh sau CLAHE
plt.figure(figsize=(10, 5))

# Ảnh gốc
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

# Ảnh sau CLAHE
plt.subplot(1, 2, 2)
plt.title("Enhanced Image (CLAHE)")
plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()

# Lưu ảnh sau xử lý
cv2.imwrite("1509_enhanced_clahe_image.jpg", enhanced_image)
