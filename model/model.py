import os
from ultralytics import YOLO

def export_yolov11_to_tensorrt(model_name: str, output_dir: str = '.', img_size: int = 640, half_precision: bool = True, dynamic_batch: bool = False, workspace_size: int = 4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, model_name.replace('.pt', f'_{img_size}_fp16' if half_precision else f'_{img_size}_fp32') + (f'_dynamic' if dynamic_batch else '') + '.engine')

    print(f"Loading YOLOv11 model: {model_name}...")
    try:
        model = YOLO(model_name)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model name is correct or the .pt file exists.")
        return

    print(f"Exporting model to TensorRT engine: {output_path}")
    print(f"  - Image size: {img_size}")
    print(f"  - Precision: {'FP16' if half_precision else 'FP32'}")
    print(f"  - Dynamic batch: {dynamic_batch}")
    print(f"  - Workspace size: {workspace_size} GB")

    try:
        model.export(
            format="engine",
            imgsz=img_size,
            half=half_precision,
            dynamic=dynamic_batch,
            workspace=workspace_size
        )
        print(f"Successfully exported {model_name} to TensorRT engine: {output_path}")
    except Exception as e:
        print(f"Error during TensorRT export: {e}")
        print("Please ensure you have TensorRT installed and configured correctly.")
        print("Also check CUDA toolkit compatibility and Ultralytics version.")

if __name__ == "__main__":
    MODEL_TO_EXPORT = "yolo11s.pt"
    OUTPUT_DIRECTORY = "./tensorrt_engines"
    IMAGE_SIZE = 640
    USE_FP16 = True
    ENABLE_DYNAMIC_BATCH = False
    MAX_WORKSPACE_GB = 8

    export_yolov11_to_tensorrt(
        model_name=MODEL_TO_EXPORT,
        output_dir=OUTPUT_DIRECTORY,
        img_size=IMAGE_SIZE,
        half_precision=USE_FP16,
        dynamic_batch=ENABLE_DYNAMIC_BATCH,
        workspace_size=MAX_WORKSPACE_GB
    )
