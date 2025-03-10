import cv2
import numpy as np
import time
from ultralytics import YOLO
from hailo_platform import (HEF, Device, VDevice, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType, HailoStreamInterface)

# ✅ Load YOLOv8 model for CPU processing
cpu_model = YOLO("yolov8n.pt")

# ✅ Load AI Hat Model
devices = Device.scan()
hef_path = "/home/queen/Downloads/yolov8n_seg.hef"  
hef = HEF(hef_path)

# ✅ Load Image
image_path = "/home/queen/test_image.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image at {image_path}")
    exit()

# ✅ Process Image on CPU
cpu_start = time.time()
cpu_results = cpu_model(image)
cpu_end = time.time()
cpu_fps = 1 / (cpu_end - cpu_start)

# ✅ Draw bounding boxes & labels from CPU
for result in cpu_results:
    for box, class_id in zip(result.boxes.xyxy, result.boxes.cls):
        x_min, y_min, x_max, y_max = map(int, box)
        class_name = cpu_model.names[int(class_id)]  # Get object name

        # ✅ Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # ✅ Display object name
        cv2.putText(image, f"{class_name}", (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

# ✅ Resize image for AI Hat input
input_data = cv2.resize(image, (640, 640))
input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

# ✅ Process Image on AI Hat
try:
    with VDevice(device_ids=devices) as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_vstreams_params = InputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32)

        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            ai_start = time.time()

            # ✅ Run inference with AI Hat
            with network_group.activate(network_group_params):
                infer_pipeline.infer({"yolov8n_seg/input_layer1": input_data})

            ai_end = time.time()
            ai_fps = 1 / (ai_end - ai_start)

# ✅ Handle Errors if AI Hat fails
except Exception as e:
    ai_fps = 0.00  # Default to 0 FPS if AI Hat fails
    print(f"AI Hat processing error: {e}")

# ✅ Display FPS values
cv2.putText(image, f"CPU FPS: {cpu_fps:.2f}", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(image, f"AI Hat FPS: {ai_fps:.2f}", (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

# ✅ Show the final comparison image
cv2.imshow("Comparison Between CPU and AI Hat Speed", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
