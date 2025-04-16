import gradio as gr
from onnx_inference_1 import AWBInference  # Import class từ file chứa class

def create_gradio_interface():
    onnx_model_path = "wbnet_model.onnx"
    t_size = 320
    inference = AWBInference(net=None, t_size=t_size, post_process=True, onnx_model_path=onnx_model_path)

    inputs = [
        gr.Image(type="pil", label="Ảnh 1"),
    ]
    outputs = gr.Image(type="pil", label="Ảnh Output")

    interface = gr.Interface(
        fn=inference.gradio_process,
        inputs=inputs,
        outputs=outputs,
        title="AWB Inference",
        description="Tải lên 1 ảnh đầu vào để xử lý cân bằng trắng và kết hợp ảnh.",
    )
    return interface

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch()
