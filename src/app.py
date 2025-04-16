from flask import Flask, request, render_template, send_from_directory
import os
import cv2
from preprocessing import process_image_pipeline
from onnx_inference import AWBInference

# Khởi tạo Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Đường dẫn đến mô hình ONNX
ONNX_MODEL_PATH = "wbnet_model.onnx"
t_size = 320  # Kích thước ảnh đầu vào cho mô hình ONNX

# Tạo đối tượng inference
inference = AWBInference(net=None, t_size=t_size, post_process=True, onnx_model_path=ONNX_MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file uploaded.", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file.", 400

    # Lưu ảnh gốc vào thư mục uploads
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Bước 1: Xử lý ảnh đầu vào và tạo 2 ảnh mới (tối và sáng)
    try:
        # Gọi pipeline xử lý từ preprocessing.py
        processed_images = process_image_pipeline(filepath, UPLOAD_FOLDER)
    except Exception as e:
        return f"Error during image processing: {str(e)}", 500

    # Bước 2: Truyền 3 ảnh (gốc, tối, sáng) vào pipeline ONNX
    output_path = os.path.join(OUTPUT_FOLDER, 'output.jpg')
    try:
        inference.run(processed_images, output_path)  # Gọi pipeline ONNX để xử lý
    except Exception as e:
        return f"Error during ONNX inference: {str(e)}", 500

    # Bước 3: Trả về giao diện hiển thị ảnh gốc và ảnh đầu ra
    return render_template('result.html', input_image=file.filename, output_image='output.jpg')


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)
